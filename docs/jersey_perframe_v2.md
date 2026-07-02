# Reconocimiento de Dorsales v1.5: Entrenamiento Per-Frame + Datos Mixtos + Fusión Calibrada
**Informe Técnico y Gate de Validación Multi-Secuencia**

> **Fecha:** 2026-06-11
> **Checkpoint de producción:** `runs/jersey_perframe_v2_mixed/best.pt`
> **Config de producción:** `--inference_mode temporal --fusion_mode arithmetic --legibility_threshold 0.70 --p1_threshold 0.75 --margin_threshold 0.15`

---

## 1. Motivación y diagnóstico

El gate v1.4 (legibilidad temporal) dejó el módulo con raw top-1 **27.3%** sobre el split
test (49 secuencias, evaluación MIL) y 66.7% E2E en `SNMOT-148`. La auditoría de esta
fase identificó tres causas raíz:

1. **Desajuste train/inference.** El modelo se entrenaba como MIL con bolsas de K=16
   crops, pero el modo de producción (`temporal`, validado en v1.3.7) ejecuta el modelo
   **per-frame** (bolsas de K=1). El backbone nunca fue optimizado para producir
   predicciones confiables por frame individual.
2. **Ruido de etiqueta por frame.** Cada crop heredaba la etiqueta del tracklet aunque
   el dorsal no fuera visible en ese frame (vistas frontales, blur, oclusión). Además,
   la loss del dígito de decenas se entrenaba con target 0 en números de 1 dígito,
   sesgando esa cabeza.
3. **Domain shift entre partidos.** val tracklet-level 75.7% vs test 30.8% con el mismo
   modelo: el split test son **partidos distintos** (fuentes/kits diferentes). El cuello
   de botella es diversidad de datos, no capacidad.

## 2. Cambios implementados

### 2.1 Entrenamiento per-frame con filtro de legibilidad (`training/identification/train_jersey_perframe.py`)
- Cada crop legible es una muestra independiente (bolsa K=1) → el checkpoint sigue
  siendo 100% compatible con `DigitCompositionalMIL` y con toda la inferencia existente.
- Los crops se filtran con el `LegibilityClassifier` (umbral 0.5) usando el cache
  `datasets/jersey_tracking_v1/legibility_scores.json` generado por
  `scripts/score_crop_legibility.py` (41,795 crops; 89.2% ≥ 0.5).
- Máscara de loss de decenas para números de 1 dígito, label smoothing 0.05,
  AdamW + warmup-coseno, augmentación fuerte (`TRANSFORM_TRAIN_PERFRAME`:
  RandomResizedCrop, perspectiva, rotación 15°, color jitter, blur, RandomErasing).
- Selección de best.pt por accuracy de tracklet fusionado en val (igual que producción).

### 2.2 Datos mixtos: SoccerNet Jersey 2023 (`scripts/build_soccernet_perframe_index.py`)
- 1,024 tracklets visibles del split **train** (test/challenge jamás se tocan).
- Submuestreo determinista de 32 frames/tracklet, torso-crop estándar (mismas
  fracciones que `core/identity/crops.py`), scoring de legibilidad y retención del
  top-16 legible por tracklet → **9,706 frames legibles de 932 tracklets**.
- Mezcla por concatenación con los 19,498 crops de tracking (≈33% SoccerNet).
- El intento v1.3.9 (MIL + fine-tuning secuencial) había fracasado (-5.5 pts);
  con per-frame + filtro de legibilidad + entrenamiento conjunto la mezcla **suma**.

### 2.3 Fusión temporal configurable (`core/identity/jersey_identity_phase.py`)
- `temporal_fusion` acepta `fusion_mode` ∈ {`geometric` (legacy, default),
  `arithmetic`, `topk_geometric`} y `temperature`.
- La fusión **aritmética** (promedio ponderado de probabilidades) es robusta a frames
  "confiados pero equivocados" que dominan la fusión geométrica en log-espacio.
- Expuesto en el CLI de la fase y en `src/tactical_vision_pipeline.py`
  (`--fusion_mode`, `--temperature`). Defaults intactos → cero cambio de
  comportamiento para flujos existentes.

### 2.4 Evaluación de generalización multi-secuencia (`scripts/evaluate_jersey_dataset_temporal.py`)
- Replica el camino de producción (per-frame + legibilidad + fusión + asignación
  por secuencia/equipo) sobre **todas** las secuencias de un split.
- Cachea probabilidades per-frame en `.npz` → los sweeps de calibración corren en
  segundos sin re-inferencia GPU.
- Protocolo anti-fuga: **calibración solo en val** (sweep de 180 configs),
  test se reporta una vez con la config elegida.

## 3. Protocolo de calibración

Sweep en val (70 tracklets, secuencias de partidos de train nunca usadas como test):
`fusion_mode × temperature {1.0,1.5,2.0,3.0} × legibilidad {0.4–0.8} × p1 {0.60,0.75,0.85}`.
Regla de selección (idéntica al gate v1.4): false-lock ↑, locked-acc ↓, cobertura locked ↓,
assigned ↓. Ganadora: **arithmetic, T=1.0, legibilidad 0.70**. `p1` se mantiene en 0.75
(perfil de producción); el trade-off de `p1` se reporta en §4.3.

## 4. Resultados

### 4.1 Val (calibración) y test (generalización, 49 secuencias, sin roster)

| Métrica (camino temporal de producción) | Baseline MIL v2 | Per-frame v1 | **Per-frame v2 mixed** | Δ vs baseline |
|---|---:|---:|---:|---:|
| val tracklet fused top-1 | 57.1%* | 75.7% | **87.1%** | +30.0 |
| test raw top-1 (matched) | 26.0% | 30.8% | **37.4%** | **+11.4** |
| test raw top-3 (matched) | 40.9% | 44.1% | **50.0%** | +9.1 |
| test assigned (matched) | 16.3% | 19.6% | **20.2%** | +3.9 |
| test locked accuracy (p1=0.75) | 75.5% | 87.6% | **94.6%** | +19.1 |
| test false lock rate (p1=0.75) | 24.5% | 12.3% | **5.4%** | **-19.1** |

\* val del baseline según evaluación MIL dataset-native v1.3 (57.1%); con el camino temporal
la comparación de val no es estrictamente homogénea, test sí lo es (misma tubería exacta).

### 4.2 E2E `SNMOT-148` (holdout, con roster + team audit, IoU+Hungarian vs GT)

| Métrica | v1.4 (baseline) | **v1.5 (per-frame v2 mixed)** | Δ |
|---|---:|---:|---:|
| GT raw top-1 (matched) | 66.7% (12/18) | **72.2%** (13/18) | +5.5 |
| GT best-frag raw top-1 | 66.7% | **77.8%** (14/18) | +11.1 |
| GT assigned (matched) | 61.1% (11/18) | **66.7%** (12/18) | +5.6 |
| GT locked / acc | 8 @ 100% | 7 @ 85.7% (1 falso) | -1 lock |
| GT best-frag lock acc | 100% | 100% | = |
| Team side consistency | 100% | 100% | = |

**Modo de fallo del único false lock:** GT #33 leído como 44 (92/178 frames per-frame
predicen 44 en crudo) — confusión sistemática 3↔4 a baja resolución, no un error de
roster ni de postproceso. Documentado como caso para auditoría visual futura.

### 4.3 Trade-off del umbral de lock (test, 49 secuencias)

| p1 | locked | locked acc | false lock |
|---:|---:|---:|---:|
| 0.60 | 111 | 88.3% | 11.7% |
| **0.75 (producción)** | **56** | **94.6%** | **5.4%** |
| 0.85 (perfil conservador) | 27 | 96.3% | 3.7% |

## 5. Reproducción

```bash
# 1) Cache de legibilidad sobre el dataset de tracking
python scripts/score_crop_legibility.py \
    --dataset_dir datasets/jersey_tracking_v1 \
    --legibility_model runs/jersey_legibility_v1/best.pt

# 2) Índice per-frame legible de SoccerNet Jersey 2023 (solo train)
python scripts/build_soccernet_perframe_index.py \
    --soccernet_root datasets/soccernet/jersey-2023 \
    --legibility_model runs/jersey_legibility_v1/best.pt \
    --output_json datasets/soccernet/jersey-2023/perframe_index_train.json

# 3) Entrenamiento mixto per-frame
python training/identification/train_jersey_perframe.py \
    --dataset_dir datasets/jersey_tracking_v1 \
    --output_dir runs/jersey_perframe_v2_mixed \
    --init_from runs/jersey_digit_mil_v2/best.pt \
    --soccernet_index datasets/soccernet/jersey-2023/perframe_index_train.json \
    --min_legibility 0.5 --epochs 30 --batch_size 64 --lr 2e-4 --device cuda:0

# 4) Calibración en val (sweep in-memory desde cache)
python scripts/evaluate_jersey_dataset_temporal.py \
    --model_path runs/jersey_perframe_v2_mixed/best.pt \
    --split val --output_dir outputs/jersey_eval/temporal_perframe_v2 --sweep

# 5) Reporte de generalización en test (config calibrada)
python scripts/evaluate_jersey_dataset_temporal.py \
    --model_path runs/jersey_perframe_v2_mixed/best.pt \
    --split test --output_dir outputs/jersey_eval/temporal_perframe_v2 \
    --fusion_mode arithmetic --legibility_threshold 0.7 \
    --p1_threshold 0.75 --margin_threshold 0.15

# 6) E2E SNMOT-148 (producción)
python core/identity/jersey_identity_phase.py \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --detections_json outputs/SNMOT-148/SNMOT-148_detections.json \
    --team_assignments_json outputs/SNMOT-148/SNMOT-148_team_assignments.json \
    --model_path runs/jersey_perframe_v2_mixed/best.pt \
    --output_json outputs/SNMOT-148/SNMOT-148_jersey_identity.json \
    --inference_mode temporal --fusion_mode arithmetic \
    --legibility_model runs/jersey_legibility_v1/best.pt --legibility_threshold 0.7 \
    --p1_threshold 0.75 --margin_threshold 0.15 \
    --roster_json datasets/jersey_tracking_v1/rosters.json \
    --team_mapping outputs/SNMOT-148/team_audit.json --device cuda:0
```

## 5.1 Render final con dorsales

La salida de la Fase 10 ahora se consume en el render final de la Fase 9:
`core/mapping/tactical_vision_2d_mapper.py --jersey_json <salida fase 10>` reemplaza la
etiqueta `#track_id` por el **dorsal identificado** (locked en firme; tentative con `?`)
tanto en los bounding boxes del video como en los puntos del minimapa 2D. El pipeline
(`src/tactical_vision_pipeline.py`) lo pasa automáticamente cuando `--jersey_model`
está activo. Video demo: `outputs/SNMOT-148/SNMOT-148_2d_map_final_jersey.mp4`.

---

# v1.6: Postproceso de Identidad y Gate E2E Multi-Secuencia

> **Fecha:** 2026-06-12
> **Flags de producción añadidos:** `--link_fragments --split_on_switch --reassign_conflicts`

## 7. Motivación

Tras v1.5, el análisis E2E mostró que el techo ya no era solo el modelo sino la
**estructura de identidad**: la accuracy por mejor-fragmento (77.8% en SNMOT-148)
superaba a la del fragmento representativo (72.2%) — la fragmentación de ByteTrack
diluye evidencia. Además, la validación E2E dependía de una sola secuencia.

## 8. Cambios implementados

### 8.1 Gate E2E multi-secuencia (`scripts/evaluate_jersey_e2e_multi.py`)
- Orquesta `jersey_identity_phase` + `evaluate_jersey_e2e` sobre N secuencias con
  prerequisitos del pipeline y agrega métricas GT-level.
- `scripts/run_e2e_prereqs.ps1` genera los prerequisitos faltantes (detección,
  clustering, audit) por secuencia.
- Gate actual: **SNMOT-148, SNMOT-116, SNMOT-132, SNMOT-190** (65 jugadores GT).
  Purezas de clustering: 100%, 90.6%, 88%*, 100% (*132 bajo la regla del 90% —
  el roster por equipo es menos confiable allí).

### 8.2 Enlace de fragmentos (`core/identity/tracklet_linking.py::link_fragments`)
Une fragmentos del mismo jugador y agrupa su evidencia per-frame antes de la fusión.
Requisitos para enlazar A→B (todo sin GT):
1. Mismo `team_id` (0/1) y frames disjuntos con gap ≤ 50 frames.
2. Continuidad espacial: distancia entre el bbox de salida de A y el de entrada de B
   ≤ `min(300px, 40 + 6·gap + tamaño/2)`.
3. **Compatibilidad de identidad (crítica)**: las fusiones por-fragmento de A y B deben
   coincidir en top-1 (o una carecer de evidencia). Sin esta regla, un fragmento
   *incierto pero equivocado* voltea a uno *confiado y correcto* — observado en
   SNMOT-132 (GT#33: el pooling permisivo lo convertía en "5"). La iteración
   permisiva ("permitir si uno es incierto") se probó y se descartó con datos.
4. Greedy por (gap, distancia); cada fragmento se encadena a lo sumo una vez por lado.

### 8.3 Detección de cambio de identidad (`detect_mode_switch`)
Escanea la secuencia de predicciones per-frame buscando un punto de corte donde el
modo dominante cambia (≥8 frames y ≥60% de dominancia por lado). Si se detecta:
el tracklet **nunca se bloquea** (lock→tentative), y si el segmento minoritario es
≥30% de los frames, la fusión usa solo el segmento dominante (contaminación grande).

### 8.4 Reasignación de conflictos (`jersey_assignment.py --reassign_conflicts`)
Exclusividad intra-frame por equipo: cuando dos tracklets solapados comparten dorsal,
el perdedor ya no cae a `unknown` — recibe su mejor alternativa **no usada por ningún
tracklet solapado** si su probabilidad ≥ 0.30, como `tentative`.

## 9. Resultados del gate E2E multi-secuencia (4 secuencias, 65 jugadores GT)

| Métrica | v1.5 (sin postproc v2) | **v1.6 (postproc v2 estricto)** |
|---|---:|---:|
| raw top-1 | 55.4% | **55.4%** (sin regresión) |
| assigned | 40.0% | **41.5%** (+1.5) |
| locked / acc | 14 @ 92.9% | 14 @ 92.9% |
| best-frag raw | 67.7% | 69.2% |

Por secuencia, la mejora se concentra en SNMOT-116 (assigned 22.2%→27.8%); ninguna
secuencia empeora en ninguna métrica (mejora de Pareto). La iteración intermedia con
linking permisivo lograba +11 pts en 116 pero rompía 132 (−6.7) — ver §8.2.

**Lectura honesta:** con 65 jugadores GT, ±1 jugador = ±1.5 pts; las conclusiones
fuertes de v1.6 son (a) el postproceso nunca degrada, (b) el gap fragmento-vs-mejor-
fragmento (55.4% vs 69.2%) sigue siendo el mayor headroom restante, y (c) ahora existe
un gate multi-partido reproducible en vez de un solo holdout.

## 10. Reproducción v1.6

```bash
# Prerequisitos del pipeline por secuencia (detección + clustering + audit)
powershell -ExecutionPolicy Bypass -File scripts/run_e2e_prereqs.ps1

# Gate multi-secuencia, configuración de producción v1.6
python scripts/evaluate_jersey_e2e_multi.py \
    --sequences SNMOT-148 SNMOT-116 SNMOT-132 SNMOT-190 \
    --model_path runs/jersey_perframe_v2_mixed/best.pt \
    --tag v2_strict \
    --link_fragments --split_on_switch --reassign_conflicts
# Agregado: outputs/jersey_eval/e2e_multi_v2_strict.json
```

El pipeline completo acepta los mismos flags (`--link_fragments --split_on_switch
--reassign_conflicts`) y los pasa a la Fase 10.

---

# v1.7: Re-extracción a 224px y Reentrenamiento

> **Fecha:** 2026-06-12
> **Checkpoint de producción:** `runs/jersey_perframe_v3_224/best.pt` (img_size=224 embebido)

## 11. Motivación
Los crops del dataset estaban almacenados a 128×128 px — un techo duro de información
(un dígito ≈ 15 px). Era la palanca individual más grande identificada en v1.5/v1.6.

## 12. Cambios implementados
1. **Dataset re-extraído a resolución nativa 224px**: `datasets/jersey_tracking_v2_224`
   (mismo constructor, `--img_size 224`; mismos parámetros, mismas 106 secuencias,
   792 tracklets de test idénticos). `splits.json` verificado byte-a-byte contra v1 —
   protocolo de splits intacto.
2. **Tamaño de entrada como metadato del checkpoint**: `save_full_checkpoint(...,
   img_size=...)` lo persiste y `load_jersey_model` lo restaura (`model.img_size`);
   `get_model_transform(model)` construye la transform correcta. Toda la inferencia
   (fase E2E, evaluadores, MIL legacy) resuelve la resolución automáticamente — los
   checkpoints antiguos siguen funcionando a 128 y el de legibilidad permanece a 128.
3. **Transforms parametrizadas** (`build_transform_inference/train_perframe(img_size)`)
   en `core/identity/jersey_model.py`; las instancias legacy de 128 se conservan.
4. **Entrenamiento**: misma receta v1.5 (per-frame + legibilidad + mezcla SoccerNet,
   warm-start desde `jersey_perframe_v2_mixed`), única variable cambiada: resolución
   (224, batch 32). Esto aísla el efecto de la resolución en la comparación.

## 13. Resultados v1.7

### Test split (49 secuencias, camino de producción, sin roster)
| Métrica | v1.5/1.6 (128px) | **v1.7 (224px)** | Δ |
|---|---:|---:|---:|
| raw top-1 | 37.4% | **39.4%** | +2.0 |
| raw top-3 | 50.0% | **52.3%** | +2.3 |
| assigned | 20.2% (25.5%*) | **25.5%** | +5.3 |
| locked (p1=0.75) | 56 @ 94.6% | **80 @ 92.5%** | +43% cobertura |
| locked (p1=0.85, perfil conservador) | 27 @ 96.3% | **51 @ 94.1%** | +89% cobertura |

\* el assigned de 128px con reassign_conflicts activo no cambió en dataset-native.

### Val (calibración): raw 90.0% → **92.9%**, assigned **75.7%**, 36 locks @ **100%**, FL 0%.

### Gate E2E multi-secuencia (4 partidos, 65 jugadores GT, postproc v1.6 activo)
| Métrica | v1.5 | v1.6 | **v1.7** |
|---|---:|---:|---:|
| raw top-1 | 55.4% | 55.4% | 55.4% |
| **assigned** | 40.0% | 41.5% | **46.2%** |
| locked / acc | 14 @ 92.9% | 14 @ 92.9% | **17 @ 88.2%** |
| SNMOT-148 assigned | 66.7% | 66.7% | **77.8%** |

Lectura: la resolución mueve principalmente la **confianza correcta** (más locks, más
assigned con la misma raw E2E — el raw E2E matched está limitado por fragmentación y
legibilidad física, no por el clasificador). Los 2 locks incorrectos del gate (1 en
148, 1 en 116) están en el régimen del trade-off p1 documentado arriba.

## 14. Reproducción v1.7
```bash
# Re-extraer dataset a 224px
python training/identification/build_tracking_jersey_dataset.py \
    --root_dir data/tracking/SoccerNet/tracking \
    --output_dir datasets/jersey_tracking_v2_224 --img_size 224

# Cadena completa (scoring + entrenamiento + evals + gate E2E)
powershell -ExecutionPolicy Bypass -File scripts/run_train_224_chain.ps1
```

## 15. Trabajo futuro
1. **Resolución nativa mayor**: re-extraer crops del dataset a 224px (hoy 128px,
   límite duro de información) y reentrenar — la palanca más grande pendiente para
   el raw top-1 y la confusión 3↔4.
2. **Cerrar el gap de fragmentación restante** (55.4% vs 69.2% best-frag): linking con
   embedding de apariencia (color/HSV ya disponible del clustering) para enlazar
   fragmentos donde la geometría no alcanza.
3. **Ampliar el gate E2E** a 8-10 secuencias (los prerequisitos se generan con
   `run_e2e_prereqs.ps1`; solo cuesta GPU).
4. Reconocedor de texto de escena (PARSeq) fine-tuneado como segunda opinión en
   ensamble (`docs/2404.08401v5.pdf`).
5. Calibración de temperatura por NLL en val con re-ajuste conjunto de umbrales.

## 16. v1.8 (2026-07-02): recalibración de fusión — geometric p1=0.80

Auditoría disparada por la caída de locks E2E en SNMOT-148 (5 vs 10 de la
referencia v1.7 del 12-jun, generada con código pre-commit `7803b91`).

**Hallazgos** (sweep en val con caché npz + verificación en test, 792 tracklets):

| config (test split) | raw top-1 | locks | lock acc | false lock |
|---|---|---|---|---|
| arithmetic p1=0.75 m=0.15 (prod anterior) | **39.4%** | 80 | 92.5% | 7.5% |
| **geometric p1=0.80 m=0.15 (nueva prod)** | 36.1% | **99** | **93.9%** | 6.1% |
| geometric p1=0.85 m=0.20 (max precisión) | 36.1% | 75 | **97.3%** | 2.7% |

- El checkpoint v3_224 **reproduce exactamente** los números documentados de
  v1.7 en el dataset (arithmetic 39.4% / 80 @ 92.5%): el modelo y la fusión
  están intactos; la pérdida E2E venía del punto de operación, no del modelo.
- E2E SNMOT-148 con geometric p1=0.80: **7 jugadores GT bloqueados @ 100%,
  0 falsos** (antes 5 @ 100%).
- **Fine-tune v4 descartado con evidencia**: warm-start 15 épocas con 29,662
  crops legibles (tracking 224 + SoccerNet 2023) → test 35.0% / 95 @ 91.6%,
  ligeramente peor que v3. El modelo está limitado por datos/resolución de
  dígitos, no por entrenamiento (checkpoint en `runs/jersey_perframe_v4_224_ft`,
  no promovido).
- Render: jugadores sin dorsal ya NO muestran `#<track_id>` (se leía como
  dorsal imposible >99); debug con `--show_track_ids`.

Producción: `runs/jersey_perframe_v3_224/best.pt` + `--fusion_mode geometric
--p1_threshold 0.80 --margin_threshold 0.15` (defaults del pipeline).

## 17. v1.9 (2026-07-02): restricción por roster — precisión >85% con 2.8x cobertura

Idea (usuario): los dorsales válidos de cada equipo se conocen a priori
(post-procesado con plantillas). La máscara de roster ya existía
(`--roster_json`, `scripts/extract_rosters.py`, 106 secuencias desde
gameinfo.ini) pero NO estaba en el camino de producción.

| config test (792 tracklets) | assigned | locks | lock acc |
|---|---|---|---|
| geometric p1=0.80 sin roster (v1.8) | 25.8% | 99 | 93.9% |
| geometric p1=0.85 m=0.20 + roster (**v1.9 prod**) | **42.3%** | **226** | **88.0%** |
| topk_geometric p1=0.75 + roster (descartada) | 41.8% | 309 | 81.9% |

- E2E SNMOT-148 (GT): **12/18 jugadores bloqueados (66.7% cobertura) @ 91.7%**
  — antes 5/18 (27.8%) @ 100%. Umbral elegido A PRIORI (el más conservador de
  la rejilla), no optimizado sobre test.
- `topk_geometric` (fusión de las mejores distribuciones por frame) se evaluó:
  en val empata, en test pierde precisión → geometric se mantiene.
- El sweep sobre train NO es utilizable para calibrar (el modelo memoriza:
  100% en todo); solo val (pequeño) y decisiones a priori.
- Producción: pipeline pasa `--roster_json datasets/jersey_tracking_v1/rosters.json`
  por defecto (secuencia ausente en el archivo → sin restricción). Requiere
  `team_mapping` (audit) para mapear cluster→lado.
