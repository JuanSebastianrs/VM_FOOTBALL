# Resumen de benchmark de backends — SNMOT-148

> **No es gaze real.** Es orientacion visual *aproximada* basada en keypoints (cabeza/torso/carrera). En broadcast los ojos rara vez son visibles; `theta_source` indica de donde sale cada estimacion.

| backend | pose_valid | head_angle | jitter° | conf | scan_ev | ms/frame |
|---|---|---|---|---|---|---|
| yolo_pose | 0.926 | 0.616 | 38.28 | 0.654 | 6 | 202.0 |
| mediapipe | 0.685 | 0.685 | 31.53 | 0.744 | 4 | 273.5 |
| hybrid | 0.933 | 0.829 | 31.26 | 0.809 | 1 | 321.9 |

## Recomendacion

**`hybrid` (YOLO-Pose por defecto, MediaPipe en crops grandes con fallback a YOLO).**

### Por que hybrid mejora la orientacion
- Mayor `pose_valid_rate` que yolo_pose y mucho mayor que mediapipe (combina cobertura de YOLO en crops pequenos + cabeza limpia de MediaPipe en grandes).
- Mayor `head_angle_rate` (mas frames resueltos por cabeza, la fuente mas fiable).
- Menor `angular_jitter_deg` => orientacion mas estable temporalmente.
- El fallback MediaPipe->YOLO evita perder el frame cuando MediaPipe no detecta.

> Nota: hybrid suele detectar **menos** eventos de scanning que yolo_pose. No es peor: su menor jitter elimina 'scans' espurios por ruido de keypoints. Esto confirma que `scan_count` es sensible al ruido del backend y **necesita GT humano**.

## Limitaciones
- Baja resolucion y oclusiones en broadcast => orientacion aproximada.
- Jugadores de espaldas: la cabeza no resuelve frente/espalda en 2D; se cae a torso/carrera.
- MediaPipe colapsa en crops pequenos (por eso hybrid usa YOLO ahi).
- El tiempo YOLO(GPU) vs MediaPipe(CPU) no es comparable 1:1.

> `scan_count` y las metricas de scanning son **heuristicas**: miden *cambios* de orientacion, no mirada verificada. Su validacion de accuracy requiere **anotacion humana** (los seed labels NO son GT).
