# Documentación del Pipeline de TacticalVision AI

Este documento describe la arquitectura técnica, las fases de procesamiento y las optimizaciones del pipeline TacticalVision AI. Incluye la detección con YOLO26 y RT-DETR, el seguimiento temporal con Grafo HMM/Viterbi, la segmentación con SAM2 y la reconstrucción 2D del campo con YOLOv11-Pose.

## Arquitectura del Pipeline (End-to-End)

El pipeline de resolución se divide en 7 fases principales gestionadas por el orquestador principal (`src/tactical_vision_pipeline.py`):

### Fase 1: Extracción de Características (Feature Extraction)
*   **Modelos:** 
    *   **YOLO26 Nano:** (SOTA) Especializado en detección de balón. Utiliza una capa **P2 (Stride 4)** y **STAL** (Small-Target-Aware Label Assignment) para capturar el balón incluso a resoluciones bajas. Reemplaza al anterior YOLO11.
    *   **RT-DETR (Real-Time DEtection TRansformer):** Utilizado para la detección robusta de jugadores y porteros.
*   **Salida:** Un archivo `_detections.json` con coordenadas `[x, y, w, h]` y scores de confianza.

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
### Fase Extra: Reconstrucción 2D y Calibración Geométrica (PnLCalib Minimap)
*   **Modelo de Extracción:** **Arquitectura Dual HRNet-W48 (PnLCalib)**.
*   **Descripción Arquitectónica:** En un salto de precisión sobre la heurística de estimaciones pasadas (como YOLOv11-Pose), este módulo integra PnLCalib (Perspective-n-Line/Point Calibration) enfocado al ámbito deportivo. Usa dos redes neuronales estriadas **HRNet** (`SV_kp` y `SV_lines`) funcionando en paralelo. Una predice *mapas de calor (heatmaps)* espaciales correspondientes a 29 puntos clave de la FIFA, y la otra aísla las aristas (líneas) demarcatorias del campo verde.
*   **Decodificación e Intersección:** Se combinan las matrices de confusión de ambas redes, completando los puntos claves ciegos u ocluidos mediante un cálculo algebraico de las intersecciones de las líneas proyectadas de la cancha, garantizando resistencia a oclusiones topológicas formadas por los propios jugadores.
*   **Matriz de Proyección (P) y Homografía Pura (H_inv):** 
    En lugar de calcular una homografía inestable por Direct Linear Transform (DLT) frame a frame, el sistema ejecuta una votación heurística con optimización tipo RANSAC y LM (Levenberg-Marquardt) para deducir el **Espacio 3D completo de la Cámara Broadcast**. 
    Infiere 8 grados de libertad (DOF): Pan, Tilt, Roll, las Distancias Focales ($f_x, f_y$) y el vector de traslación Posicional ($x,y,z$) con respecto al centro oficial de la cancha.
    Con esto, se ensambla la **Matriz de Proyección de Cámara $3 \times 4$ ($P = K[R|t]$)** que mapea matemáticamente cualquier sistema de coordenadas Mundiales al Plano de Imagen. Dado que nos interesa el mapeo 2D rasante (`Pitch Ground Plane`), aplicamos la restricción de altura $Z=0$ sobre $P$, lo que nos permite descartar la tercera columna de covarianzas de altura y aislar una estricta **Matriz de Homografía $3 \times 3$ ($H_{inv}$)**. Esta matriz resultante efectúa una transformación proyectiva bidireccional entre el espacio de píxeles (plano de la imagen) y el **sistema de coordenadas locales del campo en metros 2D**, permitiendo la reconstrucción fidedigna del posicionamiento táctico.
*   **Suavizado Temporal Matemático (Bidirectional SO(3) Lie Algebra & ESKF-Lite):**
    Los modelos frame-a-frame puros producen "flickering" ocasionado por variaciones probabilísticas de RANSAC. Inicialmente se incorporaron filtros predictivos tipo Kalman, pero su ejecución bidireccional generaba un problema de oscilación severa al chocar las inercias físicas (adelante vs atrás).
    Para solucionar esto definitivamente, se reemplazó por un **Suavizador Bidireccional de Álgebra de Lie (BidirectionalLieSmoother)** trabajando offline en dos direcciones:
    1. **Pasada Hacia Adelante (Forward Pass):** Utiliza un enfoque predictivo de estado de error (ESKF-Lite), donde la velocidad angular ($\omega$) es rastreada por un promedio móvil para predecir la posición rotacional de la cámara.
    2. **Pasada Hacia Atrás (Backward Pass):** Utiliza un **EMA puro sobre el colector SO(3)** prescindiendo *intencionalmente* de la predicción de velocidad. Omitir la velocidad aquí fue el cambio clave que detuvo las oscilaciones fantasmas del minimapa.
    3. **Fusión Geodésica:** Ambas trayectorias se unen calculando la interpolación esférica (Geodesic Midpoint) sobre la variedad $SO(3)$ para evitar el temido Gimbal Lock y problemas de no-ortogonalidad matricial.
    El resultado otorga una suavidad panorámica imbatible sin el retraso causal ("lag") latente en los filtros tradicionales.
*   **Script Core:** `core/mapping/tactical_vision_2d_mapper.py`


---

## Organización de Modelos y Pesos

1.  **YOLO26 (Balón):** `models/yolo26.pt` (Arquitectura SOTA con P2 y STAL).
2.  **RT-DETR (Jugadores):** `models/rtdetr-l.pt` (Basado en Transformer).
3.  **HRNet PnLCalib (Keypoints):** `models/SV_kp` (Genera mapas de calor de puntos de cancha).
4.  **HRNet PnLCalib (Lines):** `models/SV_lines` (Inferencia paralela de aristas).
5.  **SAM2 (Segmentación):** `models/sam2.1_hiera_small.pt`.


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
