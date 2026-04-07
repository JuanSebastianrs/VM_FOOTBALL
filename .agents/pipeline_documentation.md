# Documentación del Pipeline de TacticalVision AI

Este documento describe la arquitectura técnica, las fases de procesamiento y las optimizaciones recientes aplicadas al pipeline de seguimiento de balón (ball tracking) basado en el Grafo Oculto de Markov (HMM) con decodificación de Viterbi, así como la segmentación con SAM2.

## Arquitectura del Pipeline (End-to-End)

El pipeline de resolución se divide en 7 fases principales gestionadas por el orquestador principal (`src/tactical_vision_pipeline.py`):

### Fase 1: Extracción de Características (Feature Extraction)
*   **Modelos:** Utiliza **YOLOv26** (pesos optimizados) para la detección de candidatos de balón y **RT-DETR** para la detección de jugadores en la cancha.
*   **Salida:** Un archivo `_detections.json` que almacena las coordenadas espaciales `[x, y, w, h]` y los scores de confianza (conf) para cada frame.

### Fase 2: Compensación de Movimiento de Cámara (CMC - Camera Motion Compensation)
*   **Descripción:** Estima el movimiento afín/homográfico de la cámara entre frames consecutivos extrayendo puntos clave (keypoints) del campo.
*   **Salida:** Un archivo `_cmc.json` que contiene las matrices de transformación, fundamental para alinear espacialmente el grafo de Viterbi a través del tiempo.

### Fases 3, 4 y 5: Grafo Dinámico, Manejo de Oclusiones y Decodificación Viterbi
*   **Descripción:** Es el núcleo algorítmico del seguimiento temporal. En lugar de confiar ciegamente en YOLO, construye un Grafo HMM:
    *   **Nodos de Observación:** Detecciones reales entregadas por YOLO de la Fase 1.
    *   **Nodo Fantasma (Dummy Node):** Un estado artificial que el modelo puede transicionar si considera que el balón está ocluido o el tracking se perdió. Si transiciona al Dummy, intenta "pegarse" al bounding box de un jugador cercano basándose en un umbral de intersección (`foot_pct`).
*   **Cálculo de Transiciones:** 
    *   **Costo Cinemático:** Castiga saltos espaciales irreales basándose en las matrices CMC (distancia Euclidiana compensada).
    *   **Costo de Apariencia / Confianza:** Recompensa anclarse a detecciones de alta confianza (`score_reward`).
    *   **Costo de Oclusión:** Recompensa entrar al Dummy Node si no hay detecciones lógicas, para mantener la coherencia temporal.
*   **Decodificación:** El algoritmo de **Viterbi** minimiza la penalización global para encontrar el camino (trayectoria) óptimo a lo largo del video.
*   **Salida:** Un archivo `_trajectory.json` que indica las coordenadas corregidas en cada frame y un booleano indicando si es un estado Dummy u observación real.

### Fase 6: Segmentación Zero-Shot y Renderizado de Videos
*   **Descripción:** Utiliza **SAM2 (Segment Anything Model 2)** (versión `hiera_small`). Con la trayectoria optimizada de Viterbi funcionando como "bounding box prompt" dinámico, SAM2 genera las máscaras de segmentación píxel a píxel del balón.
*   **Salida Visual:** Genera dos vídeos representativos:
    1.  **Detecciones Crudas:** El vídeo sin refinamiento (`YOLOv26 + RT-DETR`).
    2.  **TacticalVision Refined:** El vídeo de seguimiento temporal óptimo (Viterbi) más la máscara destacada del balón por SAM2, coloreado dependiendo de su estado (visible u ocluído).

### Fase 7: Evaluación y Métricas (Thesis Analytics)
*   **Descripción:** Toma la trayectoria final y efectúa validaciones de métricas con el Ground Truth (dataset original manual).
*   **Métricas Evaluadas:**
    *   F1-Score, Precisión y Recall.
    *   Center Location Error (CLE) y perfiles espaciales.
    *   Métricas de Recuperación de Oclusiones (ORR - Occlusion Recovery Rate).
    *   Curvas mAP y Gráficos de Éxito (Success Plot).

---

## Organización de Modelos y Pesos

Para mantener la limpieza del repositorio, todos los archivos de pesos (.pt) se encuentran centralizados en la carpeta `models/`:

1.  **YOLO (Balón):** `models/yolo26.pt`
2.  **RT-DETR (Jugadores):** `models/rtdetr-l.pt`
3.  **SAM2 (Segmentación):** `models/sam2.1_hiera_small.pt`
4.  **Base Models (Otros):** `models/yolov8n.pt`, `models/yolo11n.pt`

---

## Modificaciones Recientes y Tunning Computacional

### 1. Optimización Bayesiana (Optuna)
Se construyó el script `tactical_vision_optimize.py` que emplea un optimizador TPE paramétrico en Optuna para ajustar de manera probabilística **7 hiperparámetros core** de la Fase 3-5:
*   `w1, w2, w3`: Pesos de costo para transiciones del grafo.
*   `delta_P`: Distancia máxima cinemática en píxeles permitida.
*   `lambda_cost, alpha`: Controles subyacentes de la transición al estado de Oclusión.
*   `score_reward`: Bonus al costo temporal por score en bounding box.
*   `foot_pct`: Umbral de solapamiento para adherirse a un jugador en estados ocluidos.

Para no hacer *overfitting*, la evaluación de la función de pérdida (`Loss = (1 - F1) + CLE/1000`) se realizó solo para un ensamble muestral representativo de los videos de Training, permitiendo luego inyectar la configuración validada en los secuencias Test (`best_hmm_params.yaml`).

### 2. Error Arquitectónico Conocido: Re-Adquisición del Balón desde el Dummy Node
*   **Síntoma:** Tras varios frames ocluido/perdido, Viterbi entra al Dummy y proyecta la trayectoria. Con el paso de los frames, la proyección en cadena diverge del estado real. Cuando YOLO vuelve a ver el balón en otro punto y la fase de inferencia lo reporta, la distancia física proyectada vs la real supera el hiperparámetro `delta_P` de barrera cinemática. En lugar de recuperarla, asume que es un error y permanece atrapado en el Dummy.
*   **Impacto de la caída en SAM2:** Al quedar trabado, la máscara temporal intentará adivinar "restos visuales" generando inferencias extrañas en partes de publicidad o tribunas.
*   **Solución pendiente:** Implementar un **Gating Relajado de Re-adquisición**. Es necesario indicar a la matriz probabilística que las barreras umbral espaciales deben penalizar fuertemente las continuidades reales (Real → Real) para no hacer track a ruido, pero que la ventana de la transición (Dummy → Real) **debe estar muy relajada**, para permitir saltos cinemático agresivos.
