# Reporte SNMOT-148 — Visual Scanning V2

> Estimación de orientación visual **aproximada** + head-turn **heurístico**.
> **No es gaze real.**

## Resumen

| Métrica | Valor |
|---|---|
| frames | 750 |
| tracks | 53 (39 player, 12 unknown, 2 referee) |
| árbitros identificados | track 6, track 77 |
| tracks rol `unknown` (no candidatos) | 12 |
| homografía | sí |
| **recepciones detectadas** | **5** (todas `player`) |
| candidatos rechazados | 30 |
| — rechazos por árbitro | **23** (todos track 6) |
| — rechazos por ambigüedad | 4 |
| — rechazos por baja confianza | 3 |
| eventos de head-turn (scan_label_pred=1) | **0** (heads diminutas; ver abajo) |
| head_pose rows | 232 |
| backend usado | yolo_pose_body 160, body_orientation 72 |
| head_crop_valid_rate | ≈ 0.013 |

## El bug de la V1, corregido

La V1 eligió al **árbitro (track 6)** como receptor (`ev_t6_f141`). En la V2 el
track 6 es `is_referee=True`, **nunca** aparece como `receiver_track_id` y fue
**rechazado 23 veces** (ver `rejected_reception_candidates.csv`). Las 5
recepciones detectadas son todas de rol `player`.

Además, los 12 tracks sin asignación explícita de rol ahora son `unknown`
(antes se asumían `player`) y quedan **excluidos** como posibles receptores.

## Eventos detectados

5 recepciones (`pass_reception_events.parquet`), confianza 0.46–0.58, `source =
heuristic`. Conservador por diseño: preferimos pocas recepciones correctas a
muchas falsas.

## Head pose y por qué 0 head-turns

Las cabezas en broadcast son diminutas (`head_crop_valid_rate ≈ 0.013`), por lo
que la orientación proviene del **proxy de keypoints de cuerpo** (160 frames) y
del fallback corporal (72 frames); 6DRepNet no está instalado y MediaPipe-Face
casi siempre falla en caras tan pequeñas. El `yaw` es cámara-relativo (para
*cambios*, no absoluto).

El fallback corporal (`head_pose_backend_used="body_orientation"`) está
**excluido** del conteo de head-turn por config (`allow_body_fallback_for_scan:
false`): un giro de **cuerpo** no debe contar como giro de **cabeza**. Por eso el
sistema reporta honestamente **0 head-turns confiables** en lugar de un falso
positivo (la V2 anterior reportaba 1, originado en orientación corporal). Esto es
intencional: *pocos eventos pero confiables*. Para recuperar head-turns reales hay
que instalar 6DRepNet y/o super-resolución de cabezas, o permitir el fallback
corporal explícitamente por config (menos defendible).

## Evaluación

No hay ground truth humano todavía → `reports/evaluation_gt.md` reporta
*"Ground truth not available. Only heuristic outputs generated."* (no se inventan
métricas). Para evaluar: completar `annotation_pack/annotation_template.csv`
(ver `README_annotation_guidelines.md`) y correr `evaluate_scanning_groundtruth.py`.

## Cómo correr (PowerShell Windows)

```powershell
python scripts/scanning_v2/run_scanning_v2.py `
  --config configs/scanning_v2.yaml `
  --video_id SNMOT-148 `
  --detections   outputs/SNMOT-148/SNMOT-148_detections.json `
  --trajectory   outputs/SNMOT-148/SNMOT-148_trajectory.json `
  --team_assignments outputs/SNMOT-148/SNMOT-148_team_assignments.json `
  --calibration  outputs/SNMOT-148/calibration_hinv.json `
  --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 `
  --output_dir   outputs/SNMOT-148/scanning `
  --head-pose-backend sixdrepnet --render --build-annotation-pack
```

## Cómo abrir los clips

```powershell
explorer "D:\sebastian\Tesis\VM_FOOTBALL\outputs\scanning_v2\SNMOT-148\event_clips"
$v = Get-ChildItem "D:\sebastian\Tesis\VM_FOOTBALL\outputs\scanning_v2\SNMOT-148" -Recurse -Filter "*_video.mp4"
start $v[0].FullName
```
