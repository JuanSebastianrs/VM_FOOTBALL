# Scanning V2 — Fase Supervisada

> **Esto NO es eye-gaze ni mirada exacta.** Es *detección supervisada/heurística de
> head-turn o visual scanning **aproximado** antes de la recepción*, apoyada en
> estimaciones aproximadas de orientación de cabeza/cuerpo.

## 1. Por qué una capa supervisada

La V2 heurística decide head-turn con reglas sobre el `yaw` suavizado. Esta fase
añade un **clasificador supervisado** que aprende de **ventanas de recepción
anotadas por humanos**. El objetivo es pasar de "heurística sin validar" a un
sistema **evaluable y refinable** con datos propios.

El modelo supervisado **no** corrige quién recibe el balón: opera **solo** sobre
eventos de recepción ya validados por `core/events/` (sin árbitros, sin unknown).

## 2. Tres capas separadas

| Capa | Qué hace | Dónde |
|---|---|---|
| A. Evento/receptor | quién recibe y cuándo | `core/events/` |
| B. Señales visuales aproximadas | yaw / head / body / crop quality / backend / confidence | `core/scanning_v2/` |
| C. Clasificador de scanning | predice si hubo head-turn en la ventana previa | `core/scanning_v2/supervised/` |

**Head pose backend ≠ scanning classifier.** El *backend* (6DRepNet / MediaPipe /
proxy YOLO / fallback corporal) estima una orientación aproximada por frame. El
*clasificador* consume features agregadas de esas orientaciones para predecir
head-turn a nivel de evento. Son cosas distintas.

## 3. Qué datos se necesitan

- Outputs V2 por video: `pass_reception_events.parquet`, `head_pose.parquet`,
  `scanning_events.parquet`, `vision_map_metrics.csv`.
- **Anotación humana**: `data/annotations/scanning_windows_gt.csv` con
  `scan_label_gt` (0/1) por `event_id`. Sin esto, **no se entrena**.

## 4. Cómo anotar

1. Corre la V2 con `--build-annotation-pack` → `annotation_pack/` (clips + plantilla).
2. Llena `scan_label_gt` (0/1), `visibility`, `confidence`, etc. (ver
   `README_annotation_guidelines.md`). Vacío = *unlabeled* (no se usa).
3. Vuelca/concentra las filas en `data/annotations/scanning_windows_gt.csv`.

## 5. Construir dataset → entrenar → predecir → evaluar

```powershell
python scripts/scanning_v2/build_scanning_dataset.py --config configs/scanning_v2_supervised.yaml `
  --video_ids SNMOT-148 --outputs_root outputs `
  --annotations data/annotations/scanning_windows_gt.csv `
  --output_dir outputs/scanning_training_gt/dataset

python scripts/scanning_v2/train_scanning_model.py --config configs/scanning_v2_supervised.yaml `
  --features outputs/scanning_training_gt/dataset/features.parquet `
  --labels   outputs/scanning_training_gt/dataset/labels.parquet `
  --output_dir outputs/scanning_training_gt/models `
  --model-type logistic_regression --overwrite

python scripts/scanning_v2/predict_scanning_model.py --config configs/scanning_v2_supervised.yaml `
  --features outputs/scanning_training_gt/dataset/features.parquet `
  --model   outputs/scanning_training_gt/models/scanning_classifier.pkl `
  --output_dir outputs/scanning_training_gt/predictions

python scripts/scanning_v2/evaluate_scanning_model.py --config configs/scanning_v2_supervised.yaml `
  --predictions outputs/scanning_training_gt/predictions/scanning_model_predictions.parquet `
  --labels      outputs/scanning_training_gt/dataset/labels.parquet `
  --heuristic_scanning outputs/SNMOT-148/scanning/scanning_events.parquet `
  --output_dir  outputs/scanning_training_gt/reports
```

## 6. Features (una fila por `event_id`)

Resumen de la ventana previa a la recepción: calidad (`valid_pose_ratio`,
`mean_head_pose_confidence`, `mean_yaw_confidence_smooth`, ratios por backend,
calidad de crop), yaw (rango, std, deltas, cambios de dirección, entropía, giros
sostenidos a 20/30/40°), temporal (`scan_candidate_before_1s/2s/3s`), balón/cuerpo
(`look_away_ratio`, distancias) y vision map (`observed_space_score`, visibles).
Versionadas con `feature_version`.

**Reglas:** `scan_label_gt`, `confidence`, `visibility`, `notes` son
etiquetas/eval → **nunca** features. La predicción heurística V2 entra como feature
**solo** si `use_heuristic_pred_as_feature: true`.

## 7. Modelos (fábrica de arquitecturas)

`core/scanning_v2/supervised/models.py` expone seis arquitecturas, de más simple
a más expresiva:

| tipo | naturaleza | cuándo usarla |
|---|---|---|
| `logistic_regression` | lineal, interpretable | baseline; GT pequeño |
| `random_forest` | ensamble de árboles | tabular no lineal |
| `gradient_boosting` | boosting clásico | tabular; sin class_weight |
| `hist_gradient_boosting` | boosting con NaN nativo + class_weight | default con cientos de labels |
| `mlp` | perceptrón multicapa (sklearn) | tabular denso |
| `gru_sequence` | **GRU temporal** (torch) sobre la serie de yaw re-muestreada + rama tabular | requiere `dataset.include_sequence_features: true` |

`gru_sequence` (ver `sequence_model.py`) consume las columnas
`seq_cos_XX/seq_sin_XX/seq_valid_XX` que el FeatureExtractor genera al activar
`include_sequence_features`: la serie de yaw se re-muestrea a `sequence_length`
pasos sobre los últimos `sequence_seconds` antes de la recepción, representada
como (cos, sin) por ser variable circular, con máscara de validez (no se
interpola pose inexistente). Es sklearn-compatible y picklable (joblib), así que
`train/predict/evaluate` funcionan igual para todas las arquitecturas.

Con GT pequeño los modelos simples siguen siendo el default defendible.
`class_weight: balanced`. Splits reproducibles (`random_state`), opcionalmente
agrupados por video (`group_split_by_video`).

`scripts/scanning_v2/train_model_zoo.py` entrena TODAS las arquitecturas con el
mismo split, las evalúa en test held-out, compara con la heurística V2 y copia
la mejor a `models/best/` con un reporte comparativo.

## 7b. Weak supervision (multi-video, sin GT humano a escala)

Cuando aún no hay GT humano suficiente, `generate_weak_labels.py` produce
pseudo-etiquetas con reglas EXPLÍCITAS y más estrictas que la heurística de
runtime (`core/scanning_v2/supervised/weak_labeler.py`): solo eventos
inequívocos (positivo = giros sostenidos repetidos y amplios; negativo = cabeza
esencialmente quieta; zona gris = sin etiqueta), con gates de calidad de pose.
Se escriben en un archivo SEPARADO (`scanning_windows_weak_v1.csv`,
`label_source=weak_rules_v1`) — nunca en el GT humano, que siempre tiene
prioridad. Config: `configs/scanning_v2_supervised_weak.yaml`.

**Advertencia metodológica:** un modelo entrenado con etiquetas débiles es una
destilación suavizada de las reglas; sus métricas contra esas etiquetas miden
*consistencia*, no validez frente a percepción humana. Sirve para tener un
modelo operativo y una arquitectura validada mientras se anota GT humano.

## 8. Qué pasa si hay pocos labels

Si `n_labeled < min_labeled_samples` o `n_positive < min_positive_samples` o hay
una sola clase, **no se entrena**: `training_report.md` dice *"Not enough labeled
samples to train supervised scanning model."* y se mantiene la heurística V2 como
fallback. No se inventan métricas.

## 9. Interpretar `scan_probability_model`

Probabilidad estimada [0,1] de que hubo head-turn/scanning observable antes de
recibir. **No** es probabilidad de mirada; es una señal aproximada calibrada con
los pocos datos disponibles. `model_confidence = max(p, 1−p)`.

## 10. Integración

Las predicciones se guardan **aparte** (`scanning_model_predictions.parquet`,
`prediction_source=model`); **no** sobrescriben `scanning_events.parquet`. Para
combinarlas explícitamente se puede crear `scanning_events_with_model.parquet`. Sin
modelo entrenado, el sistema usa la heurística y lo reporta.

## 11. Limitaciones

- Validez limitada por el tamaño del GT humano; con pocos eventos las métricas son
  orientativas.
- Las features heredan el límite de la orientación aproximada (cabezas pequeñas en
  broadcast); no es gaze real.
- Un solo video (SNMOT-148) no permite generalización; se necesita anotar varios.
