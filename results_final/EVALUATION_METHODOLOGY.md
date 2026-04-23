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

## 4. Paneles de Visualización (Dashboards)

Para resumir científicamente todos los resultados generados por el pipeline, se crearon representaciones de "Premium Dark Design":

1. `dashboard_detection_map.png`: Perfil detallado del modelo RF-DETR para las clases involucradas.
2. `dashboard_tracking_metrics.png`: Radar general y diagrama de barras, incorporando contexto de otros algoritmos (DeepSORT, ByteTrack, MOT4MOT) del *SoccerNet Tracking Challenge 2023*.
3. `dashboard_clustering.png`: Mapas de distribución e histogramas validando numéricamente la separación de equipos sin supervisión.
