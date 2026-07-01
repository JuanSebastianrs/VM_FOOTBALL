# Metodología — Visual Scanning V2

> **Esto NO es detección exacta de mirada (gaze).** Es *estimación de orientación
> visual aproximada y detección heurística de head-turn/scanning antes de la
> recepción*, usando orientación de cabeza/cuerpo y eventos de recepción.

## 1. Por qué falló la V1

La V1 decidía el receptor con una heurística de "jugador más cercano al balón".
En SNMOT-148 eso seleccionó al **árbitro (track 6)** como receptor: el único
evento de scanning de la V1 fue `ev_t6_f141`, es decir, el árbitro. El problema
no era el backend de pose (YOLO/MediaPipe), sino la **definición de evento/receptor**.

## 2. Principio: separar tres problemas

La V2 separa estrictamente:

- **A. Game state** (`core/game_state/`): quién es jugador/arquero/árbitro, dónde
  está, qué equipo tiene, dónde está el balón. Aquí —y solo aquí— se decide qué
  track es candidato a receptor (`is_candidate_receiver = role ∈ {player, goalkeeper}`)
  y cuál es árbitro (`is_referee`).
- **B. Evento/receptor** (`core/events/`): quién recibe y cuándo. Nunca un árbitro.
- **C. Head-turn/scanning** (`core/scanning_v2/`): si ese receptor giró la cabeza
  antes de recibir. No redefine quién recibe.

## 3. Detección de recepción (conservadora)

Prioridad de fuente: `ground_truth` → `manual` → `model` → `heuristic`
(`event_ground_truth_adapter.py` mapea datasets externos: SoccerTrack v2,
SoccerNet Ball-Action, FOOTPASS, anotación propia). Toda `source` desconocida se
normaliza a `heuristic` (la más baja) para no sobre-confiar en datos opacos.

**`player_id` NO es receptor.** El adaptador ya **no** mapea `player_id` a
`receiver_track_id`: un `player_id` puede ser el ejecutor del pase o el dueño de
la acción. Si el GT no trae un receptor explícito, el evento se marca
`missing_receiver_in_gt` y **no** se usa para scanning. Además, un evento GT se
**rechaza** si el receptor es árbitro, rol no candidato, `NaN`, o no existe en el
`game_state`.

Sin GT, la heurística (`pass_reception_detector.py`) **no** usa solo distancia:

- la posesión se estima **solo entre candidatos** (`possession_estimator.py`);
  un árbitro cerca del balón nunca genera posesión y se registra como rechazado;
- `attached_player_id` (balón "atado") **solo** se acepta si apunta a un candidato
  presente en el frame, dentro de la distancia y sin ambigüedad fuerte; si no, se
  registra `invalid_attached_player` y se cae a la lógica de distancia;
- exige cercanía durante varios frames + posesión estable (`min_possession_frames`)
  + **track estable** (el receptor aparece en casi todo el run);
- usa desaceleración del balón cerca del receptor;
- verifica una transición pasador→receptor con `frame_pass` razonable
  (`max_pass_gap_frames`); si no, el pase no se atribuye y baja la confianza;
- si los equipos de pasador y receptor son incoherentes, baja fuerte la confianza
  (`team_mismatch_penalty`) o descarta (`team_coherence_required`);
- ante ambigüedad fuerte (dos candidatos casi a la misma distancia) baja la
  confianza o descarta;
- exige `event_confidence ≥ min_event_confidence`; si no, **no crea evento**.

Todo candidato rechazado se registra en `rejected_reception_candidates.csv` con
su razón: `referee`, `unknown_role`, `ball_too_far`, `unstable_track`,
`ambiguous_receiver`, `low_confidence`, `missing_ball`, `missing_homography`,
`missing_receiver_in_gt`, `invalid_attached_player`.

## 4. Head pose (orientación aproximada)

Cadena de backends (`head_pose_estimator.py`), configurable en
`configs/scanning_v2.yaml`: `sixdrepnet` → `mediapipe` (FaceLandmarker) →
`yolo_pose_body` (proxy de yaw desde la geometría de los keypoints de cabeza) →
fallback a orientación de cuerpo (marcado `head_pose_backend_used="body_orientation"`,
confianza baja).

**Device y modelos configurables (no hardcodeados):** `pose_device`
(`cpu`/`cuda:N`), `sixdrepnet_gpu_id`, `face_model_path` y
`allow_model_download` salen del YAML. Por defecto **no** se descargan modelos en
runtime (`allow_model_download: false`); si 6DRepNet no está instalado o el modelo
de cara no existe localmente, la cadena sigue al siguiente backend sin romper.

**Sistema de referencia (clave):** el `yaw` es **cámara-relativo**, no orientación
en cancha. Se usa para detectar *cambios* de cabeza (head-turn), no como
orientación absoluta. La orientación de **cuerpo** (`theta_body_img/field`) sí
tiene marco imagen/cancha (reusa el estimador V1 corregido). **No se mezclan**:
`theta_head_field` queda vacío salvo conversión explícita (no disponible en
broadcast).

## 5. Head-turn / scanning

`head_turn_detector.py` toma `yaw_smooth` (suavizado circular) en la ventana
`[t_recepción − 3.0 s, t_recepción − 0.2 s]` y cuenta **giros sostenidos**
(deltas consecutivos del mismo signo sobre un piso de ruido; un solo frame
ruidoso no dispara). Usa la **confianza suavizada** (`yaw_confidence_smooth`,
persistida en `head_pose.parquet`) cuando existe. `scan_label_pred = 1` requiere
`head_turn_count > 0`, `valid_pose_ratio` y confianza media sobre umbral.

**Un giro de cuerpo NO es un head-turn.** Las filas cuyo backend es
`body_orientation` se **excluyen** del conteo de giros salvo que
`allow_body_fallback_for_scan: true`. Así un mero giro corporal no dispara
`scan_label_pred`. En broadcast, donde las cabezas son diminutas y domina el
fallback corporal, esto produce **0 head-turns confiables** en vez de falsos
positivos — coherente con la prioridad "pocos eventos pero confiables".

`look_away/back_from_ball` se computan con la **orientación de cuerpo en cancha**
vs la dirección al balón en cancha (mismo marco), no con el yaw cámara-relativo.

## 5b. Vision map (espacio observado)

`vision_map_wog.py` se integra al pipeline: en el frame de recepción de cada
evento estima un campo de visibilidad (FOV con decaimiento) usando
`theta_head_field` si existe, si no `theta_body_field` (confianza baja, 0.3). Si
**no** hay orientación en cancha, `vision_map_available=False` (no se inventa una
flecha). Las métricas por evento se exportan en `vision_map_metrics.csv`.

## 6. Significado de `scan_label_pred`

Es una **predicción heurística** de si hubo head-turn observable antes de recibir.
No es verdad de campo. Su validación de accuracy requiere **anotación humana**
(`annotate_scanning_windows.py` → `scanning_windows_gt.csv`), evaluada con
`evaluate_scanning_groundtruth.py`. Sin GT humano, el sistema **no inventa
métricas**.
