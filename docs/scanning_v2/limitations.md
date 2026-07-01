# Limitaciones — Visual Scanning V2

> Estimación de orientación visual **aproximada** + head-turn **heurístico**.
> **No es gaze real / mirada exacta.**

## Datos y eventos
- Sin ground truth de eventos público para SNMOT-148: las recepciones son
  **heurísticas** (`source = heuristic`). Pueden faltar recepciones reales
  (el detector es conservador a propósito) o incluir alguna dudosa.
- `attached_player_id` del balón es escaso (≈24/750 frames en SNMOT-148): la
  posesión depende sobre todo de cercanía + estabilidad + desaceleración.
- Tracks sin equipo (`team_id` ausente/-2) bajan la confianza del evento.

## Head pose
- En broadcast la cabeza suele medir **< 32 px**: los head crops dedicados rara
  vez son válidos (en SNMOT-148, `head_crop_valid_rate ≈ 0.03`). Por eso domina
  el **proxy de yaw desde keypoints de cuerpo** y el fallback a orientación
  corporal, no 6DRepNet/MediaPipe-Face.
- **6DRepNet no está instalado** en este entorno → la cadena cae a MediaPipe
  (que casi siempre falla en caras diminutas) y luego al proxy YOLO. Documentado.
- El `yaw` es **cámara-relativo**: sirve para detectar *cambios* (head-turn), no
  como orientación absoluta. No se convierte a cancha (no fiable en broadcast).

## Scanning
- `scan_label_pred` es **heurístico**, no verdad de campo. Mide cambios de
  orientación, no mirada verificada.
- Distinguir "giró la cabeza" de "giró el cuerpo" es difícil cuando solo hay
  keypoints de cuerpo. Por eso el fallback corporal
  (`head_pose_backend_used="body_orientation"`) está **excluido** del conteo de
  head-turn por defecto (`allow_body_fallback_for_scan: false`). Consecuencia: en
  broadcast (cabezas <32 px, `head_crop_valid_rate ≈ 0.013`) el sistema reporta
  **0 head-turns confiables** en SNMOT-148 en lugar de falsos positivos. Para
  recuperar head-turns reales se necesita 6DRepNet instalado y/o super-resolución
  de head crops.
- La validación de precision/recall/F1 **requiere anotación humana**
  (`annotation_pack/` + `scanning_windows_gt.csv`). El evaluador desglosa además
  por `visibility`, `event_source`, `head_pose_backend_used` y `crop_quality`, y
  lista FP/FN por `event_id` cuando hay GT.

## Geometría / referencia
- Orientación de **imagen** y de **cancha** se mantienen separadas; el minimapa
  usa la de cancha (cuerpo) y, si no existe, **no dibuja flecha** (muestra el
  aviso "orientation field unavailable").

## Reproducibilidad
- Tiempos no comparables 1:1 entre backends (GPU vs CPU).
- Resultados dependen de la calidad del tracking/calibración previos.
