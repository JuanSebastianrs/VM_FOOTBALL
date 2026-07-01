# Comparacion de pipeline completo por backend — SNMOT-148

> Orientacion visual **aproximada** basada en keypoints. **No es gaze real.** Cada backend corre el `ScanningPipeline` end-to-end sobre la MISMA secuencia.

- Dispositivo: YOLO-Pose en GPU (CUDA), MediaPipe en CPU (XNNPACK) — el tiempo no es comparable 1:1

| Metrica | yolo_pose | mediapipe | hybrid |
|---|---|---|---|
| total_frames | 750 | 750 | 750 |
| player_frame_samples | 9065 | 9065 | 9065 |
| pose_valid_rate | 0.926 | 0.685 | 0.933 |
| head_angle_rate | 0.616 | 0.685 | 0.829 |
| torso_valid_rate | 0.909 | 0.683 | 0.922 |
| mean_orientation_confidence | 0.654 | 0.744 | 0.809 |
| angular_jitter_deg | 38.28 | 31.53 | 31.26 |
| receptions_detected | 14 | 14 | 14 |
| scanning_events_detected | 6 | 4 | 1 |
| mean_scan_count | 0.50 | 0.29 | 0.07 |
| mean_max_orientation_change_deg | 76.20 | 27.26 | 10.21 |
| crops_to_mediapipe | 0 | 8993 | 7086 |
| crops_to_yolo | 8993 | 0 | 3690 |
| fallbacks_mp_to_yolo | 0 | 0 | 1783 |
| elapsed_s | 151.5 | 205.1 | 241.4 |
| ms_per_frame | 202.0 | 273.5 | 321.9 |

## theta_source (%)

| Fuente | yolo_pose | mediapipe | hybrid |
|---|---|---|---|
| head | 58.4 | 68.4 | 81.2 |
| torso | 32.7 | 0.0 | 11.2 |
| movement | 7.5 | 28.4 | 6.5 |
| previous | 1.0 | 2.7 | 0.7 |
| invalid | 0.4 | 0.5 | 0.4 |

## Backend realmente usado por player-frame

| Backend usado | yolo_pose | mediapipe | hybrid |
|---|---|---|---|
| invalid | 670 | 2857 | 609 |
| mediapipe | 0 | 6208 | 5303 |
| yolo_pose | 8395 | 0 | 3153 |

## Distribucion por bbox_height

| Bucket | yolo_pose | mediapipe | hybrid |
|---|---|---|---|
| small | 2922 | 2922 | 2922 |
| medium | 5491 | 5491 | 5491 |
| large | 652 | 652 | 652 |

## Lectura

- `hybrid` debe igualar o mejorar la cobertura (`pose_valid_rate`) de `yolo_pose` y acercarse a la limpieza de `mediapipe` en crops grandes.
- `angular_jitter_deg` menor = orientacion mas estable.
- El conteo de recepciones es independiente del backend (depende del balon).
