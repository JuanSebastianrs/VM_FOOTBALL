# Visual Scanning Module (`core/scanning`)

Estimación de **orientación visual aproximada** y detección de **scanning** (barrido
visual) de jugadores en video broadcast de fútbol — el comportamiento de jugadores
como Xavi, que escanean el campo constantemente antes de recibir el balón.

> ⚠️ **Esto NO es eye-tracking / gaze exacto.** En broadcast los ojos casi nunca son
> visibles con suficiente resolución. El objetivo realista es estimar una
> **orientación-proxy** (hacia dónde está orientado el jugador) a partir de cabeza,
> hombros, torso, dirección de carrera y contexto temporal, y detectar *cambios* de
> esa orientación antes de recibir.

---

## 1. Qué problema resuelve

Dado un video ya procesado por el pipeline táctico (detección + tracking + balón +
calibración de cancha), para cada jugador y frame estima:

- `theta_visual ∈ [-π, π]` — orientación visual aproximada (en imagen y en cancha).

Y para cada **recepción** de balón detecta si el jugador hizo **scanning** en la
ventana previa (`t_recepción − 3.0 s … t_recepción − 0.2 s`), reportando:

`scan_count`, `max_orientation_change_deg`, `mean_orientation_change_deg`,
`orientation_entropy`, `looked_away_from_ball_count`, `looked_towards_ball_count`,
`quality_score`.

## 2. Qué es "visual scanning"

Un jugador hace scanning si, en la ventana previa a recibir, **cambia de forma clara
su orientación** (cabeza/torso/cuerpo) para observar zonas del campo distintas a la
dirección inmediata del balón o de su carrera. Se mide por la *variación temporal* de
la orientación-proxy, no por su valor absoluto.

## 3. Por qué no se estima gaze exacto

- Resolución insuficiente de los ojos en broadcast.
- Oclusiones frecuentes y jugadores de espaldas a cámara.
- No existe un dataset público que combine *video de fútbol + crops + etiqueta exacta
  de mirada + scanning antes de recibir*.

Por eso se usa una **jerarquía de robustez** (cabeza → torso → carrera → anterior) y un
modelo temporal entrenable sobre anotaciones propias.

## 4. Arquitectura

```
video/crop → keypoints → orientación → suavizado temporal
           → recepciones → scanning → exporter → visualización
```

| Módulo | Archivo | Responsabilidad |
|---|---|---|
| Recorte | `player_cropper.py` | bbox expandida → crop redimensionado + calidad |
| Keypoints | `keypoint_extractor.py` | backends YOLO-Pose (default) / MediaPipe → landmarks canónicos |
| Orientación | `orientation_estimator.py` | jerarquía cabeza/torso/carrera/anterior + proyección a cancha |
| Suavizado | `orientation_smoother.py` | EMA **circular** por `track_id`, α adaptativo a confianza |
| Recepciones | `reception_detector.py` | posesión por frame (`attached_player_id` + heurística) |
| Scanning | `scanning_detector.py` | métricas de barrido en la ventana previa |
| Vision map | `vision_map.py` | mapa probabilístico de campo visible (opcional) |
| Export | `exporter.py` | `player_orientation.parquet`, `scanning_events.parquet` |
| Visualización | `visualization.py` | overlay de video + minimapa |
| Orquestador | `pipeline.py` | `ScanningPipeline` (coordina los módulos) |
| I/O | `data_io.py` | adapta los JSON del pipeline táctico a estructuras tipadas |
| Circular | `circular.py` | matemática de ángulos (wrap, delta, media, entropía, EMA) |
| Modelo | `models/` | TCN/BiLSTM entrenable + dataset + train/eval |

**Convención de ángulos:** matemática estándar (0 = derecha, +90 = arriba, antihorario).
Como el eje *y* de imagen apunta hacia abajo, los ángulos en imagen niegan la componente
*y* (`circular.image_vector_to_angle`). Todo combinado/suavizado se hace sobre
`[cos θ, sin θ]` para evitar el salto en ±π.

## 5. Datos que necesita

Por secuencia, los productos del pipeline táctico:

- `*_detections.json` — jugadores por frame (`track_id`, bbox) + candidatos de balón.
- `*_trajectory.json` — balón refinado por frame (incluye `attached_player_id`).
- `*_team_assignments.json` — `track_id → team_id, role` (opcional).
- `calibration_hinv.json` — homografía por frame (imagen→cancha, opcional pero
  recomendado para orientación y posiciones en cancha).
- `img1/` — frames de la secuencia (requerido para keypoints y overlay de video).

### Datasets recomendados (estrategia híbrida)
- **Fútbol (detección/tracking/contexto):** SoccerNet-Tracking, SoccerNet-GSR, videos propios.
- **Head pose / gaze (pre-entrenamiento de referencia):** 300W-LP, AFLW2000, BIWI,
  Gaze360, ETH-XGaze. *No son broadcast de fútbol → no usar como única fuente.*
- **Dataset propio de orientación:** crops anotados (ver §6). Inicial sugerido:
  500–1000 crops de orientación + 100–300 ventanas pre-recepción etiquetadas scan/no-scan.

## 6. Cómo construir anotaciones

```bash
python scripts/scanning/build_orientation_annotation_dataset.py \
    --video_id SNMOT-148 \
    --detections   outputs/SNMOT-148/SNMOT-148_detections.json \
    --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json \
    --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
    --calibration  outputs/SNMOT-148/calibration_hinv.json \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --stride 5
```

Genera:
- `data/annotations/<video>_orientation_features.parquet` — features por frame
  (keypoints + contexto), consumido por el dataset de entrenamiento.
- `data/annotations/orientation_labels.csv` — **plantilla pre-llenada** con la
  orientación heurística como *semilla*. El anotador corrige
  `orientation_angle_deg` / `orientation_bin_8` y marca `scan_label`.

Columnas del CSV: `sample_id, video_id, frame_id, track_id, crop_path,
context_frame_path, x_field, y_field, ball_x, ball_y, time_to_reception,
orientation_bin_8, orientation_angle_deg, scan_label, visibility, confidence,
annotator_notes`. (`orientation_bin_8`: 0=derecha, 2=arriba, 4=izquierda, 6=abajo, …)

## 7. Cómo correr el pipeline

```bash
python scripts/scanning/run_scanning_pipeline.py \
    --video_id SNMOT-148 \
    --detections   outputs/SNMOT-148/SNMOT-148_detections.json \
    --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json \
    --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
    --calibration  outputs/SNMOT-148/calibration_hinv.json \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --visualize
```

Salidas en `outputs/scanning/<video_id>/`:
`player_orientation.parquet`, `scanning_events.parquet`, `reception_events.parquet`,
`debug_video.mp4`, `minimap_scanning.mp4`.

Flags útiles: `--no_pose` (orientación sólo por movimiento, sin keypoints — rápido),
`--max_frames N`, `--format csv`.

#### Columnas de `player_orientation`

La orientación se separa por **marco de referencia** y por **crudo vs suavizado**
(todos los ángulos en grados, convención matemática):

- `theta_visual_img` / `theta_visual_smooth_img` — orientación en **imagen**.
- `theta_visual_field` / `theta_visual_smooth_field` — orientación en **cancha**
  (vía homografía). El minimapa y el `vision_map` usan la versión `*_field`.
- `ball_relative_angle_img` / `ball_relative_angle_field` — dirección jugador→balón
  en cada marco. `looked_towards/away_from_ball` se calcula en **cancha** cuando hay
  homografía (compara `theta_visual_smooth_field` con `ball_relative_angle_field`).
- `theta_source` ∈ {head, torso, movement, previous, invalid}.
- `head_angle` (encaramiento del rostro), `shoulder_angle` (**eje** de hombros, sólo
  diagnóstico — el encaramiento del torso se calcula **perpendicular** a este eje),
  `torso_angle` (eje caderas→hombros), `run_angle`.
- `keypoint_backend_used`, `keypoint_backend_attempted`, `pose_valid`, `crop_quality`.

> **Geometría de la orientación (corregido):** la línea de hombros es un *eje*, no la
> dirección de mirada. El encaramiento se toma **perpendicular** al eje de hombros y
> el frente/espalda se desambigua con (en orden) la cabeza, la carrera, la dirección
> al balón y la orientación previa.

### Comparación de backends en el pipeline completo

```bash
python scripts/scanning/full_pipeline_comparison.py \
    --video_id SNMOT-148 \
    --detections outputs/SNMOT-148/SNMOT-148_detections.json \
    --trajectory outputs/SNMOT-148/SNMOT-148_trajectory.json \
    --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json \
    --calibration outputs/SNMOT-148/calibration_hinv.json \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148
```

Corre el pipeline end-to-end para `yolo_pose`, `mediapipe` y `hybrid`, y escribe
`outputs/scanning/backend_benchmark/full_pipeline_comparison.md` (pose_valid_rate,
head/torso_angle_rate, jitter, ms/frame, recepciones, eventos de scanning y la
distribución de backends realmente usados).

### Clips de eventos de scanning (auditoría visual)

```bash
python scripts/scanning/render_scanning_events.py \
    --video_id SNMOT-148 \
    --detections outputs/SNMOT-148/SNMOT-148_detections.json \
    --trajectory outputs/SNMOT-148/SNMOT-148_trajectory.json \
    --calibration outputs/SNMOT-148/calibration_hinv.json \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --orientation outputs/scanning/SNMOT-148/player_orientation.parquet \
    --scanning    outputs/scanning/SNMOT-148/scanning_events.parquet \
    --top 10
```

Renderiza los N eventos más fuertes como `<tag>_video.mp4` + `<tag>_minimap.mp4`
(ventana de 3 s antes de la recepción, flecha en imagen y cono en cancha).

### Backends de keypoints

`configs/scanning.yaml → keypoint_backend`: `"yolo_pose"` | `"mediapipe"` | `"hybrid"`.

- **`yolo_pose`** (default) — `ultralytics`, ya dependencia; descarga `yolo11n-pose.pt`
  la primera vez. Mejor **cobertura**, sobre todo en crops pequeños (jugadores lejanos).
- **`mediapipe`** — Pose Landmarker (Tasks API, `full`/`heavy`/`lite`); requiere
  `pip install mediapipe` y descarga `pose_landmarker_<c>.task`. Cabeza más limpia y
  **menos jitter** cuando detecta, pero **falla mucho en crops pequeños**. Entrega
  además `world_landmarks` (3D en metros).
- **`hybrid`** — usa MediaPipe si `bbox_height >= mediapipe_min_crop_height` (default
  110 px) y YOLO-Pose en el resto. **Si MediaPipe falla la detección, reintenta con
  YOLO-Pose** (fallback). Cada fila exportada registra `keypoint_backend_used`
  (`yolo_pose`/`mediapipe`/`invalid`) y `keypoint_backend_attempted`
  (p.ej. `mediapipe+yolo_pose`). Es lo recomendado por el benchmark.

**Comparación objetiva** (mismos crops, doble inferencia):

```bash
python scripts/scanning/benchmark_keypoint_backends.py \
    --video_id SNMOT-148 \
    --detections outputs/SNMOT-148/SNMOT-148_detections.json \
    --trajectory outputs/SNMOT-148/SNMOT-148_trajectory.json \
    --calibration outputs/SNMOT-148/calibration_hinv.json \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --n_random 100 --n_reception 50 --jitter_tracks 6
```

Genera `outputs/scanning/backend_benchmark/`: `report.md`, `metrics.json`,
`per_crop.csv` y muestras visuales (`yolo_pose_samples/`, `mediapipe_samples/`,
`side_by_side/`). En SNMOT-148: YOLO-Pose `pose_valid≈0.93` (small `0.90`) vs
MediaPipe `0.57` (small `0.37`); MediaPipe gana en confianza de keypoints (`0.99` vs
`0.76`), jitter (`38.9°` vs `45.9°`) y cabeza en crops medianos (`0.81` vs `0.64`).
**Conclusión: `hybrid` (YOLO por defecto, MediaPipe en crops grandes).**

## 7b. Cómo ver los resultados de SNMOT-148 (Windows PowerShell)

Resultados en `D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning`.

```powershell
# Abrir la carpeta de resultados
explorer "D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning"

# Listar todos los videos generados
Get-ChildItem "D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning" -Recurse -Filter *.mp4 | Select-Object FullName

# Abrir la carpeta de eventos
explorer "D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning\events"

# Abrir el primer clip de evento (video)
$videos = Get-ChildItem "D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning" -Recurse -Filter "*_video.mp4"
start $videos[0].FullName

# Abrir el primer minimapa de evento
$maps = Get-ChildItem "D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning" -Recurse -Filter "*_minimap.mp4"
start $maps[0].FullName

# Ver los eventos de scanning con Python
python -c "import pandas as pd; p=r'D:\sebastian\Tesis\VM_FOOTBALL\outputs\SNMOT-148\scanning\scanning_events.parquet'; df=pd.read_parquet(p); print(df[['event_id','receiver_track_id','frame_reception','scan_count','max_orientation_change_deg','quality_score']])"
```

## 8. Cómo entrenar el modelo temporal

El baseline heurístico es transparente y sirve de semilla. El **modelo temporal**
(TCN/BiLSTM) aprende un mapeo más fino sobre los mismos features + tus etiquetas:

```bash
python scripts/scanning/train_orientation_model.py \
    --labels   data/annotations/orientation_labels.csv \
    --features data/annotations/SNMOT-148_orientation_features.parquet \
    --model tcn --epochs 50
```

Entradas por frame (39-D): keypoints normalizados (nose/ojos/orejas/hombros/caderas
× x,y,visibilidad), bbox w/h, velocidad, `run_angle` (sin/cos), `ball_relative_angle`
(sin/cos), `x_field`/`y_field`, distancia al balón, `time_to_reception`.
Salidas: regresión angular (sin,cos), `orientation_bin_8`, confianza.
Loss: `(1 − cos Δθ) + 0.5·CE(bin) + 0.2·BCE(conf)`.
Checkpoint: `outputs/scanning/checkpoints/best_orientation_tcn.pt`.

## 9. Cómo evaluar

```bash
# A. Orientación
python scripts/scanning/evaluate_orientation.py \
    --checkpoint outputs/scanning/checkpoints/best_orientation_tcn.pt \
    --labels   data/annotations/orientation_labels.csv \
    --features data/annotations/SNMOT-148_orientation_features.parquet

# B. Scanning (requiere GT por recepción con columna scan_label)
python scripts/scanning/evaluate_orientation.py \
    --scanning_pred outputs/scanning/SNMOT-148/scanning_events.parquet \
    --scanning_gt   data/annotations/scanning_gt.csv
```

Orientación: MAE angular, accuracy@22.5°/45°, accuracy por bin, errores por
visibilidad/tamaño de crop. Scanning: precision/recall/F1 + matriz de confusión.

## 10. Limitaciones

- Baja resolución en broadcast → la orientación es **aproximada**.
- Oclusiones y aglomeraciones degradan los keypoints.
- Jugadores de espaldas → la cabeza no resuelve front/back desde 2D; se cae a torso/carrera.
- Keypoints inestables frame a frame (mitigado por el suavizado circular).
- No hay dataset público perfecto → la calidad depende de las anotaciones propias.
- Las métricas de scanning del baseline son heurísticas; el bin/ángulo "verdadero"
  necesita anotación humana (la plantilla viene con semilla heurística, **no** es GT).

## 11. Trabajo futuro

- Fine-tuning del modelo temporal con anotaciones propias a escala.
- Backends de pose más fuertes (ViTPose, RTMPose) y super-resolución de crops.
- Estimación explícita de yaw front/back de cabeza.
- Integración con pitch control / pitch value y con el vision map para "espacio observado".
- GT de scanning por recepción para cerrar el bucle de evaluación.

## 12. Principio de diseño

Separación estricta: **(A) visión** (frame/crop → keypoints/orientación),
**(B) modelado táctico** (orientación + balón + recepción + cancha → scanning),
**(C) visualización**. Código modular, testeable (`tests/test_scanning.py`) y extensible
— sin scripts monolíticos.
