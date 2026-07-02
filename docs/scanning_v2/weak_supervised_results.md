# Scanning V2 — Resultados fase supervisada con weak supervision (2026-07-01)

> Orientación visual **APROXIMADA** de cabeza/cuerpo, **NO gaze real**. Métricas
> contra pseudo-etiquetas = *consistencia con las reglas*; la validez se mide
> aparte contra el GT de revisión visual.

## 1. Datos

- **42 secuencias** SoccerNet test con pipeline táctico completo. La homografía
  por frame (`calibration_hinv.json`) se generó para las 48 secuencias con el
  mapper PnLCalib (`--output_calibration`, suavizado SO(3) bidireccional);
  el ajuste alternativo desde `tracking_2d.csv` reproduce la calibración de
  referencia con error mediano de 0.0002 m (SNMOT-148).
- **115 eventos de recepción** detectados (detector conservador de
  `core/events/`, con distancias métricas gracias a la homografía).
- **16 pseudo-etiquetas débiles** inequívocas (4 scanning / 12 no-scanning),
  `weak_rules_v1` (ver `weak_labeler.py`): gate de calidad por cobertura de yaw
  suavizado (≥50%, ≥20 frames), positivo = giro sostenido ≥30° + alternancia
  repetida (o ≥2 giros amplios) con confianza media ≥0.30, negativo = cabeza
  esencialmente quieta. 63/115 eventos quedan fuera por cobertura de pose
  insuficiente (broadcast: jugadores pequeños/de espaldas), 28 en zona gris,
  8 reservados por tener GT visual.
- **GT independiente**: 8 eventos anotados por revisión visual de crops
  (`annotator=claude_visual_review` en `scanning_windows_gt.csv`; 3 positivos).
  Estos eventos se EXCLUYEN del entrenamiento débil.

## 2. Comparativa de arquitecturas (CV estratificada agrupada por video, k=4)

| modelo | F1@0.5 | precision | recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.667 | 1.000 | 0.500 | 1.000 | 1.000 |
| random_forest | 0.400 | 1.000 | 0.250 | 1.000 | 1.000 |
| **gradient_boosting** (best) | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| hist_gradient_boosting | 0.286 | 0.333 | 0.250 | 0.271 | 0.521 |
| mlp | 0.545 | 0.429 | 0.750 | 0.604 | 0.854 |
| **gru_sequence** (temporal) | 0.889 | 0.800 | 1.000 | 0.679 | 0.917 |

Lectura: `gradient_boosting` replica exactamente las reglas débiles (esperable:
son umbrales sobre las mismas features tabulares — esta métrica mide
consistencia, no validez). Lo relevante metodológicamente es que
**`gru_sequence` alcanza F1 0.889 leyendo la serie temporal cruda de yaw**
(cos/sin/validez re-muestreada), sin acceso directo a los conteos de giros que
definen las reglas: la señal secuencial contiene la información de scanning.

## 3. Validación independiente (GT visual, 8 eventos NUNCA vistos en train)

| sistema | accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **modelo best (gradient_boosting)** | **0.875** | **1.000** | **0.667** | **0.800** | 0.967 | 0.917 |
| heurística V2 runtime (40°) | 0.625 | 0.000 | 0.000 | 0.000 | — | — |

- El modelo detecta 2 de los 3 scannings reales sin ningún falso positivo.
- Único fallo: `SNMOT-117_rcp_0000`, un giro de cabeza visible al ojo pero con
  cobertura de pose de solo 21% (la señal de yaw casi no existe en esa ventana)
  → el límite es la estimación de pose upstream, no el clasificador.
- La heurística V2 de runtime (umbral 40°, gates estrictos) no detecta ninguno.

## 4. Artefactos

- Modelo final: `outputs/scanning_v2_supervised_weak/models/best/`
  (`scanning_classifier.pkl` + `model_metadata.json` con CV, gates y límites;
  re-entrenado con los 16 labeled tras la CV).
- Comparativa: `outputs/scanning_v2_supervised_weak/models/model_zoo_report.md`.
- Evaluación GT: `outputs/scanning_v2_supervised_weak/reports_gt/`.
- Dataset: `outputs/scanning_v2_supervised_weak/dataset/` (115 eventos,
  `scanning_v2_features_v2_seq` con serie temporal para el GRU).
- Reproducir todo: `python scripts/scanning_v2/run_supervised_weak_e2e.py
  --config configs/scanning_v2_supervised_weak.yaml --overwrite`.

## 5. Limitaciones y siguiente paso

1. n pequeño (16 weak + 8 GT): las métricas son orientativas; los intervalos de
   confianza son anchos. Prioridad: **anotación humana** de los 115 eventos
   (packs de anotación ya generables con `--build-annotation-pack`).
2. La cobertura de pose (mediana 13% con confianza ≥0.35) es el cuello de
   botella real: 63/115 ventanas no son pseudo-etiquetables. Mejorar el head
   pose en crops pequeños (p. ej. 6DRepNet instalado, super-resolución del
   crop) subiría el rendimiento de TODO el sistema.
3. El GT de revisión visual fue anotado por IA sobre contact sheets; debe ser
   validado/reemplazado por anotador humano experto para la tesis.
