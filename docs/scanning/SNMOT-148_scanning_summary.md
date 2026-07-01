# Resumen de scanning — SNMOT-148

> **No es gaze real.** Es orientacion visual *aproximada* basada en keypoints (cabeza/torso/carrera). En broadcast los ojos rara vez son visibles; `theta_source` indica de donde sale cada estimacion.

> `scan_count` y las metricas de scanning son **heuristicas**: miden *cambios* de orientacion, no mirada verificada. Su validacion de accuracy requiere **anotacion humana** (los seed labels NO son GT).

## Que se genero

Pipeline completo con **keypoint_backend = hybrid** sobre SNMOT-148.

- player-frame samples: **9065**, pose_valid_rate: **0.933**
- orientacion separada en imagen (`*_img`) y cancha (`*_field`), cruda y suavizada.
- backend por fila en `keypoint_backend_used` / `keypoint_backend_attempted`.
- recepciones detectadas: **14**
- eventos de scanning (scan_count>0): **1**

## Como interpretarlo

- El **video overlay** muestra la orientacion en imagen; el **minimapa** la orientacion en cancha (`theta_visual_smooth_field`) con cono de vision aproximado.
- `looked_towards/away_from_ball` se mide en cancha cuando hay homografia.
- Los clips por evento estan en `events/` (3 s antes de cada recepcion con scan).

## Donde estan los resultados (no versionados, pesados)

`outputs/SNMOT-148/scanning/` — parquet/csv, mp4, events/, backend_benchmark/.
Estos archivos siguen en `.gitignore`; aqui en `docs/` solo hay reportes ligeros.
