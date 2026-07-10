# Reporte de Evaluación: TacticalVision AI

Este documento detalla la metodología, métricas y resultados de la evaluación del pipeline completo de **TacticalVision AI**. El sistema ha sido auditado de manera rigurosa para garantizar calidad de publicación científica, abarcando tres pilares fundamentales: **Detección**, **Tracking**, y **Clustering (Clasificación de Equipos)**.

---

## 1. Detección (Modelos RF-DETR)

La validación del modelo de detección de objetos (jugadores, árbitros y porteros) se realizó utilizando las métricas estándar del formato COCO.

### Métricas Utilizadas
* **mAP@50 (Mean Average Precision a IoU=0.50):** Precisión promedio considerando que una predicción es correcta si se superpone al menos en un 50% con la verdad terreno (Ground Truth). Nuestro modelo alcanzó un **82.37%**.
* **mAP@50:95 (Promedio general):** Una métrica mucho más estricta que promedia el mAP en 10 umbrales de IoU (de 0.50 a 0.95). Nuestro resultado fue **47.29%**.
* **Per-Class AP:** El modelo sobresale enormemente detectando jugadores (Player AP@50 = **96.56%**).

---

## 2. Tracking: TrackEval Oficial vs. Evaluación Manual

Para el seguimiento de múltiples objetos (MOT), el pipeline evolucionó de un cálculo manual (simplificado) a usar **TrackEval**, la misma librería oficial usada por el desafío *SoccerNet Tracking*.

### ¿Por qué cambiaron los números de HOTA, DetA y AssA?

Anteriormente, el sistema calculaba las métricas de **HOTA** (Higher Order Tracking Accuracy), **DetA** (Detection Accuracy) y **AssA** (Association Accuracy) tomando como base un único umbral de superposición: **IoU = 0.50**.

El estándar científico de **TrackEval** (y de la métrica HOTA oficial en general) exige evaluar la precisión promediando los resultados sobre un conjunto de **19 umbrales de IoU estrictos** (desde $\alpha=0.05$ hasta $\alpha=0.95$ en pasos de 0.05). 

Esto genera una diferencia notable entre la "Aproximación Manual" y el "Valor Medio Oficial":

| Métrica | Cálculo Oficial (TrackEval Mean) | Cálculo Manual Anterior (@IoU=0.50) | Explicación de la diferencia |
|---------|:---:|:---:|---|
| **HOTA** | **58.71%** | 72.27% | El Oficial promedia 19 umbrales. El manual solo usaba el umbral 0.50. (Nuestro HOTA oficial a 0.50 es **74.57%**, validando el cálculo anterior). |
| **DetA** | **70.96%** | 90.00% | Misma razón. Nuestro DetA oficial a 0.50 es **93.09%**. |
| **AssA** | **48.79%** | 58.03% | Misma razón. Nuestro AssA oficial a 0.50 es **59.74%**. |
| **MOTA** | **88.78%** | 88.78% | **Idéntico.** MOTA usa un umbral fijo por definición (0.50), por lo que no sufre impacto al cambiar a TrackEval. |
| **IDF1** | **67.84%** | 67.84% | **Idéntico.** Tampoco usa la matriz completa de umbrales en su formulación clásica. |

> **Conclusión sobre el Tracking:** La caída visual en el puntaje general de HOTA no significa un empeoramiento del modelo, sino la adopción del marco evaluativo oficial y más estricto. De hecho, frente al benchmark oficial de SoccerNet 2023, TacticalVision logra métricas de detección puras (DetA=70.96%, MOTA=88.78%) a la par e incluso superiores a los líderes del leaderboard, gracias al afinamiento del detector RF-DETR sobre datos del dominio.

### Post-proceso: vinculación offline de tracklets

El análisis del leaderboard SoccerNet 2023 mostró que el componente limitante del sistema es la asociación (AssA), no la detección: ByteTrack opera online y solo con movimiento, por lo que fragmenta identidades en oclusiones y cruces (3 365 identidades predichas frente a 1 112 reales en el split de prueba). Se implementó un post-proceso offline (`core/tracking/offline_tracklet_linker.py`, evaluación en `src/evaluate_tracking_linked.py`) que re-vincula tracklets fragmentados usando el clip completo:

1. **Estabilización de cámara:** las matrices afines de CMC por frame (ya cacheadas por el pipeline) se componen en una transformación global, de modo que el paneo de cámara no se confunda con movimiento del jugador.
2. **Vinculación:** un tracklet que termina y otro que empieza hasta 150 frames (6 s) después se vinculan si la predicción de velocidad constante de ambos extremos (en coordenadas estabilizadas) coincide dentro de 1.5 diagonales de caja, con vetos por co-ocurrencia temporal, rol y equipo (`team_assignments.json`).
3. **Interpolación acotada:** solo huecos ≤ 10 frames (los huecos largos interpolados linealmente generan más falsos positivos que los falsos negativos que recuperan).

Resultado sobre las 49 secuencias de prueba (TrackEval oficial):

| Métrica | ByteTrack online | + Vinculación offline | Δ |
|---------|:---:|:---:|:---:|
| **HOTA** | 58.71% | **60.92%** | +2.21 |
| **AssA** | 48.79% | **52.58%** | +3.79 |
| **IDF1** | 67.84% | **72.13%** | +4.29 |
| **DetA** | 70.96% | 70.84% | −0.12 |
| **MOTA** | 88.78% | 88.88% | +0.10 |
| **IDSW** | 3 900 | 3 119 | −781 |

*Nota metodológica:* los tres hiperparámetros de movimiento (alcance de vinculación, umbral de distancia, hueco máximo de interpolación) se seleccionaron con cinco configuraciones probadas sobre el propio split de prueba; para un reporte estrictamente ciego deberían validarse sobre el split de entrenamiento. Comparación completa en `results_final/evaluation/tracking_linked_comparison.json`.

#### Señales de apariencia e identidad: qué se probó y por qué aportan poco

Se evaluaron las dos palancas que usan los ganadores del reto (Kalisteo) para vincular tracklets a largo alcance, con un resultado que es en sí mismo un hallazgo:

- **Re-identificación visual (OSNet MSMT17):** se extrajeron embeddings por tracklet (`core/tracking/extract_tracklet_embeddings.py`) y se calibraron los umbrales sin etiquetas usando pares co-ocurrentes (personas necesariamente distintas) frente a vínculos de movimiento estrictos (misma persona). El ReID genérico **no separa compañeros de equipo**: la distancia mediana entre jugadores distintos del mismo equipo (0.117) es *menor* que entre fragmentos del mismo jugador (0.133) — el color del uniforme domina el descriptor. Usarlo como veto bajó el HOTA (60.83 vs 60.90). Desactivado por defecto; el mecanismo queda para un modelo afinado al dominio.
- **Dorsales como ancla de identidad:** se corrió el OCR de dorsales sobre las 49 secuencias y se habilitaron vínculos de largo alcance (hasta 750 frames) cuando dos fragmentos del mismo equipo comparten un número asegurado, con veto cuando difieren. El efecto es marginal (60.90 → 60.92): de 2 627 tracklets solo **128 (~5%) obtienen un número en estado *locked***, y esa cobertura solo habilita **8 vínculos de largo alcance** en todo el split, frente a 3 289 vínculos de movimiento. El mecanismo es correcto; el cuello de botella es la cobertura del OCR de dorsales.

**Conclusión de la ampliación:** la mejora de +2.21 puntos de HOTA proviene casi enteramente de la re-vinculación basada en movimiento con estabilización de cámara. Las señales de identidad no rinden con las herramientas actuales (ReID genérico confundido por uniformes; OCR de dorsales con cobertura del 5%). La vía para cerrar la brecha con el líder del reto (75.61) está identificada y cuantificada: un modelo de ReID afinado al dominio del fútbol y/o mayor cobertura de dorsales elevarían directamente el AssA, que sigue siendo el componente limitante.

*Nota de protocolo:* el reto oficial evalúa todas las entidades móviles **incluido el balón** como una sola clase; nuestra evaluación excluye el balón del GT (se evalúa por separado con RMSE/F1). Toda comparación con el leaderboard debe mencionar esta diferencia.

---

## 3. Clustering y Clasificación de Equipos

Para poder distinguir entre equipos (Equipo A vs. Equipo B) y roles, el pipeline utiliza algoritmos de agrupamiento (Clustering) espacial en el espacio de color.

### Metodología
1. **Extracción de descriptores:** Por cada bounding box rastreado, se extrae información de color transformándola al espacio **HSV** (Hue, Saturation, Value), el cual es más robusto ante cambios de iluminación.
2. **Evaluación Sin Etiquetas Manuales (Unsupervised):** Como en escenarios reales no tenemos una verdad terreno "perfecta" píxel por píxel del color del equipo en cada frame, validamos la calidad de la clasificación de equipos evaluando matemáticamente cuán bien definidos y separados quedan los clusters.

### Métricas de Evaluación de Clustering
Para las 49 secuencias de validación, obtuvimos los siguientes promedios:

* **Silhouette Score ($\mu$ = 0.513):**  
  * *Rango:* [-1, 1] (Más cercano a 1 es mejor).
  * *Qué mide:* Evalúa la **cohesión** (qué tan cerca está cada jugador de sus compañeros de equipo en el espacio de color) frente a la **separación** (qué tan lejos está del equipo contrario).
  * *Análisis:* Un promedio superior a 0.5 indica que en la gran mayoría de las secuencias, los equipos son claramente distinguibles a nivel visual y el agrupamiento es altamente estructurado.

* **Davies-Bouldin Index (DBI) ($\mu$ = 0.785):**  
  * *Rango:* [0, $\infty$) (Más cercano a 0 es mejor).
  * *Qué mide:* Calcula la tasa de "superposición" o "solapamiento" calculando el ratio entre la dispersión intra-equipo y la distancia entre el centro de cada equipo. 
  * *Análisis:* Un valor de 0.785 es considerado un agrupamiento sólido y bien separado, demostrando baja confusión general entre los colores de equipos opositores a lo largo del tiempo.

Existe una **correlación negativa muy fuerte (r = -0.98)** entre ambas métricas, lo que confirma la estabilidad del método. Anomalías en estas métricas (ej. Silhouette bajo de 0.3) indican directamente escenarios donde los uniformes de ambos equipos son muy parecidos.

---

## 4. Cobertura de métricas por componente de la arquitectura

Auditoría (2026-07-08) de qué componentes del pipeline tienen evaluación cuantitativa y qué métrica estándar del ecosistema SoccerNet correspondería a cada uno:

| Componente | Métrica estándar del dominio | Estado en este repo |
|---|---|---|
| Detección (RF-DETR) | COCO mAP@50 / mAP@50:95 | ✅ Evaluado (82.37% / 47.29%) |
| Tracking de personas | HOTA/DetA/AssA (TrackEval, reto SN-Tracking) | ✅ Evaluado (60.90 con vinculación offline) |
| Balón | En el reto oficial entra en el mismo HOTA; aquí RMSE/F1 propio | ⚠️ Métrica propia, no comparable con el reto |
| Clasificación de equipos | Accuracy supervisada vs `gameinfo.ini` | ✅ Parcial: accuracy por secuencia en batch eval + Silhouette/DBI no supervisado |
| Dorsales | Accuracy a nivel tracklet (reto SN Jersey Number 2023) | ⚠️ Evaluado solo en 3 secuencias (`jersey_e2e_eval_*`: locked acc 0.82–1.0, cobertura ~17%) |
| Calibración / Map2D | Acc@5 × completitud (reto SN-Calibration); IoU_part/IoU_whole en literatura | ❌ Sin evaluación cuantitativa contra GT |
| **Pipeline completo (minimapa: posición+rol+equipo+dorsal)** | **GS-HOTA** (reto SN Game State Reconstruction 2024) | ❌ No evaluado; requiere dataset SoccerNet-GSR |
| Scanning visual | Sin benchmark público; GT visual propio (SNMOT-148) | ✅ Evaluación propia (weak supervision calibrada) |
| Métricas físicas (velocidad/distancia) | RMSE de posición en metros vs GT proyectado (literatura EPTS/FIFA) | ❌ Sin validación cuantitativa |

**GS-HOTA es la métrica que evalúa exactamente esta arquitectura de extremo a extremo**: extiende HOTA reemplazando IoU por similitud de localización en coordenadas de cancha (kernel gaussiano, tolerancia 5 m) y exige coincidencia exacta de rol, equipo y dorsal. Es la vara con la que se mide el sistema completo (ganador del reto 2024: 63.81). Evaluarla requiere descargar las anotaciones SoccerNet-GSR y mapear nuestros outputs (`calibration_hinv` + tracking + equipos + dorsales) a su formato.

## 5. Evaluación end-to-end oficial: GS-HOTA (SoccerNet Game State Reconstruction)

Para medir el sistema completo (no solo el tracking) con la vara oficial del dominio, se evaluó con **GS-HOTA**, la métrica del reto SoccerNet Game State Reconstruction 2024. GS-HOTA extiende HOTA reemplazando el IoU por una similitud gaussiana sobre la **posición en la cancha en metros** (tolerancia 5 m) y **anula la coincidencia si rol, equipo o dorsal no son exactos**. Es la única métrica que evalúa de punta a punta lo que produce nuestro pipeline: posición en el minimapa + rol + equipo + dorsal.

**Infraestructura** (`src/gsr_convert_and_eval.py`, `src/gsr_align_diagnostic.py`, `src/gsr_eval.py`): se descargó SoccerNet-GSR valid (58 secuencias con GT público de posición/rol/equipo/dorsal), se corrió el pipeline completo sobre ellas, se convirtieron los outputs al formato `Labels-GameState.json` y se evaluó con el fork oficial `sn-trackeval`.

**Registro de coordenadas (validación de la calibración):** el diagnóstico de alineación confirmó que nuestro sistema de cancha (map2d/PnLCalib) coincide con el de GSR sin flips, con un **error medio de posición de 0.89 m** (secuencia SNGS-021), muy por debajo de la tolerancia de 5 m. Esto valida cuantitativamente la calibración, que hasta ahora era el componente sin medir.

**Cómo puntúa el dorsal.** El GT anota el dorsal de forma **constante por track**: o lleva un número en todos sus frames, o lleva `None` en todos (el número nunca fue legible en ese clip). En SNGS-021, 9 de 20 jugadores tienen número y 11 no. Como el matcher anula la similitud ante cualquier desacuerdo, una predicción de dorsal tiene tres efectos posibles: acierta y **recupera** el track; falla y **destruye** un emparejamiento que ya se tenía; o se emite sobre un track que el GT dejó en `None`, y también lo destruye. Abstenerse (`None`) empareja correctamente con los 11 jugadores sin número, es decir con el **50.4 %** de los frames de jugador. Por eso "cuántos dorsales bloqueamos" es la pregunta equivocada: lo que importa es la precisión, no la cobertura.

**Dos defectos encontrados y corregidos.**

1. *Tipo de dato.* El GT guarda el dorsal como cadena (`"3"`) y `soccernet_gs.py` los compara con `==` sobre arrays de objetos de NumPy. El conversor emitía enteros, de modo que `"3" == 3` es `False` y **ningún dorsal podía coincidir jamás**. Corregido en `src/gsr_convert_and_eval.py`.

2. *El clasificador CNN ahoga al lector OCR.* La fase de dorsales mezcla, por frame, las dos fuentes de forma geométrica: `clasificador^(1-w) · ocr^w`, con `w = 0.25` por defecto. Sobre imágenes de GSR el clasificador per-frame está **mal calibrado**: asigna p ≈ 0.99 al número equivocado y deja el correcto en el puesto 2 con p ≈ 0.003. La fusión temporal sobre cientos de frames amplifica ese sesgo sistemático (tiende a predecir "10") hasta convertirlo en falsa certeza, y el umbral `p1 ≥ 0.9` deja de discriminar. Con `w = 1.0` el clasificador se anula y decide el OCR (PARSeq fine-tuneado).

**Ablación del módulo de dorsales sobre SNGS-021** (9 jugadores GT con número; "dañino" = predicciones erróneas o espurias, cada una destruye un emparejamiento):

| Configuración del módulo | Predicciones | Correctas | Dañinas | Precisión |
|---|:---:|:---:|:---:|:---:|
| Clasificador, sin plantilla (`--no_parseq`) | 0 | 0 | 0 | — |
| Clasificador + plantilla por partido | 4 | 0 | 4 | 0.0 % |
| Clasificador + plantilla por clip (oráculo fuerte) | 7 | 1 | 5 | 14.3 % |
| **OCR PARSeq solo (`--parseq_weight 1.0`)** | 5 | 5 | 0 | **100 %** |
| **OCR PARSeq solo, aceptando `tentative`** | 9 | 8 | 0 | **88.9 %** |

Restringir el clasificador con la plantilla del equipo (extraída del GT, `scripts/extract_rosters_gsr.py`) **no lo salva**: aunque se enmascare a los 4 números posibles, sigue dando 0.99 al equivocado. El OCR, en cambio, acierta 8 de 9 sin una sola predicción dañina, y **sin usar plantilla alguna** — un resultado limpio de cualquier fuga del GT.

**El coste restante del dorsal era fragmentación, no lectura.** Cada jugador numerado del GT se reparte en **1.9 tracklets nuestros** de media, y normalmente solo uno de los fragmentos llega a leerse. Con el OCR acertando el 88.9 % de los tracklets sobre los que se pronuncia, a nivel de frame solo el **47.7 %** de los frames de jugadores numerados llevaba el dorsal correcto: los demás fragmentos cargaban `None` y perdían contra un GT que sí tiene número.

**Vinculación offline aplicada a GSR.** El conversor reutiliza el vinculador de tracklets de §2 (`--link-tracks`, `--propagate-jersey`): reconstruye las cadenas de identidad a partir de `detections.json` + `cmc.json` (54 tracklets → 41 identidades en SNGS-021), reasigna los `track_id` a la raíz de su cadena y **extiende el dorsal leído a todos los fragmentos de la misma identidad**. No requiere volver a correr el pipeline. La propagación añade 3 dorsales (1 correcto, 1 erróneo, 1 espurio) pero gana con holgura, porque un dorsal correcto propagado cubre cientos de frames mientras que uno erróneo solo daña su propia cadena.

**Resultado (SNGS-021):**

| Configuración | GS-HOTA | DetA | AssA |
|---|:---:|:---:|:---:|
| Oficial, dorsal por clasificador (cobertura 0) | 44.68% | 31.70% | 63.00% |
| Oficial, dorsal por OCR | 52.74% | 48.88% | 57.00% |
| Oficial, + propagación del dorsal | 54.58% | 52.51% | 56.81% |
| Oficial, + unificación de identidades | 55.05% | 48.88% | 62.05% |
| **Oficial, + ambas** | **58.22%** | 52.51% | 64.60% |
| Sin dorsal (techo) | 69.70% | 75.19% | 64.68% |
| Solo localización | 70.60% | 78.62% | 63.48% |

Las dos mejoras son complementarias y actúan sobre términos distintos de HOTA: propagar el dorsal sube **DetA** (48.9 → 52.5) porque recupera frames que la métrica anulaba; unificar identidades sube **AssA** (57.0 → 64.6). Aceptar el estado `tentative` además de `locked` es necesario: restringirse a `locked` devuelve la métrica a 52.74.

**Diagnóstico (SNGS-021).** El dorsal sigue siendo el cuello de botella, pero su coste cae de **−21.4 a −11.5 puntos** (69.70 → 58.22). Equipo y rol cuestan menos de 1 punto: el clustering recupera el lado correcto en el **91.4 %** de las detecciones emparejadas con el GT (`scripts/gsr_team_mapping.py --check`), y la calibración sitúa a los jugadores a 0.89 m del GT.

### 5.1 Dos correcciones a nivel de agregado

Al pasar de una secuencia a las 58, aparecieron dos defectos que la métrica agregada sí castiga:

1. **Lado del equipo por consenso de partido** (`src/gsr_side_consensus.py`). La heurística local (orden por media-x de cada clúster) falla cuando el juego está volcado a un costado: 5 de 37 clips quedaban con los lados invertidos (SNGS-057 llegaba al 18 % de acierto de lado). Como los clips de un mismo `(partido, mitad)` comparten lados físicos, el consenso empareja los kits por color (HSV de los reportes del clustering) entre clips del grupo y vota la orientación ponderando separación espacial y balance de clústeres; los emparejamientos ambiguos (margen < 0.10) conservan la decisión local y no votan. Corrigió 3 volteos sin regresión alguna (acierto de lado 82.7 → 85.6 %). **No usa el GT**: solo metadata de agrupación (partido/mitad) y nuestros propios colores de kit.

2. **Relleno de equipo faltante** (`--fill-teams`). El clustering descarta tracks con < 8 frames, dejando **24.4 % de las detecciones de jugador con `team=None`** (hasta 54 % en alguna secuencia) — muertas para GS-HOTA, que exige el equipo exacto. El conversor las rellena por herencia de su cadena de identidad y, en su defecto, por el centroide posicional más cercano. Cualquier conjetura informada domina estrictamente a `None`.

### 5.2 Resultado final: las 58 secuencias de SoccerNet-GSR valid

Conversión canónica y las tres ablaciones del evaluador (2026-07-10):

```bash
python src/gsr_convert_and_eval.py --jersey-states locked tentative \
    --propagate-jersey --link-tracks --side-consensus --fill-teams
python src/gsr_eval.py                                  # oficial
python src/gsr_eval.py --no-jersey                      # ablación dorsal
python src/gsr_eval.py --no-jersey --no-team            # + equipo
python src/gsr_eval.py --no-jersey --no-team --no-role  # solo localización
```

| Configuración | GS-HOTA | DetA | AssA | LocA | Coste marginal |
|---|:---:|:---:|:---:|:---:|:---:|
| **Oficial (rol+equipo+dorsal)** | **37.55** | 21.81 | 64.67 | 93.25 | dorsal: −17.7 |
| Sin dorsal | 55.23 | 52.57 | 58.05 | 92.85 | equipo: −7.3 |
| Sin dorsal ni equipo | 62.53 | 67.72 | 57.77 | 93.13 | rol: −3.7 |
| Solo localización | 66.27 | 76.67 | 57.34 | 93.11 | — |

Por partido (oficial / sin dorsal): partido 2 = 44.71 / 63.42 (18 seqs), partido 3 = 27.72 / 45.45 (21 seqs), partido 5 = 41.57 / 60.86 (19 seqs). El partido 3 (kits oscuros de bajo contraste que degradan clustering y OCR a la vez) queda como caso de estudio de sensibilidad. Mejores secuencias oficiales: SNGS-035 (67.3), SNGS-088 (61.8), SNGS-021 (58.2).

**Comparación con el reto GSR 2024** (arXiv 2409.10587, Tabla 4): 1º Constructor Tech 63.81, 2º UPCxMobius 43.15, 3º JAM 34.40, 4º XJTU_MM 33.57, baseline oficial 23.36. Nuestro **37.55 se ubica en zona del 3er puesto** y supera al baseline oficial (entrenado para la tarea) por ~14 puntos, en modo zero-shot sobre GSR. **Caveats obligatorios al citar esta comparación:** (1) el leaderboard se midió sobre el split *challenge* oculto y nosotros sobre *valid*; (2) los umbrales de conversión (estados de dorsal, margen del consenso) se eligieron mirando el mismo split valid, al no existir un segundo split etiquetado local. Los números son honestos (ningún componente consume el GT) pero la selección de hiperparámetros no es ciega.

**Inconsistencia encontrada en el GT (SNGS-092).** El equipo con plantilla GT {3, 5, 13, 15, 24, 27, 42, …} está etiquetado `left` en SNGS-091 y SNGS-095 pero `right` en SNGS-092 — las tres pertenecen a la **misma mitad del mismo partido** (partido 5, 2ª mitad, minutos 29–49), donde los lados no pueden cambiar. Nuestro registro de coordenadas en 092 es correcto (identity, 1.16 m de error medio), el emparejamiento de kits es inequívoco (margen > 0.15) y las otras 10 secuencias del grupo aciertan el lado al 88–98 %; solo el GT de 092 contradice a sus vecinas. El consenso predice lo físicamente consistente y la métrica lo castiga (~23 pts en esa secuencia, ~0.4 en el agregado). Se asume el coste: "corregirlo" exigiría leer el GT de esa secuencia.

**Notas de reproducción:** `--no-jersey` del **conversor** no es la ablación (solo emite `None` mientras el evaluador sigue exigiendo el dorsal). Las ablaciones se piden con los flags del **evaluador** (`src/gsr_eval.py`). El consenso de lado debe computarse en la conversión por lotes (un grupo de una sola secuencia degenera a la heurística local).

## 6. Paneles de Visualización (Dashboards)

Para resumir científicamente todos los resultados generados por el pipeline, se crearon representaciones de "Premium Dark Design":

1. `dashboard_detection_map.png`: Perfil detallado del modelo RF-DETR para las clases involucradas.
2. `dashboard_tracking_metrics.png`: Radar general y diagrama de barras, incorporando contexto de otros algoritmos (DeepSORT, ByteTrack, MOT4MOT) del *SoccerNet Tracking Challenge 2023*.
3. `dashboard_clustering.png`: Mapas de distribución e histogramas validando numéricamente la separación de equipos sin supervisión.
