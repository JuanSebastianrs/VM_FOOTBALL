# VM_FOOTBALL ⚽🤖

**Sistema de Visión por Computadora para Análisis Táctico de Fútbol**

> Proyecto de tesis adaptado al contexto colombiano y latinoamericano.

---

## 🎯 Estado Actual

| Módulo | Estado | Archivos |
|--------|--------|----------|
| **Detección** | ✅ Funcional (RF-DETR 3 clases entrenado) | `cloud/rfdetr_player_detection/*`, `core/detection/*` |
| **Tracking** | ✅ Funcional | ByteTrack (Ultralytics) |
| **Equipos + Portero** | ✅ Funcional (evaluación) | `core/clustering/team_classifier.py`, `eval_team_clustering.py` |
| **Dorsales** | 📋 Planificado | `core/identity/` |

### RF-DETR 3 clases (completado)

- Taxonomía: `player`, `goalkeeper`, `referee` (balón ignorado)
- Checkpoint principal (GCS): `gs://vm-football-data/models/rfdetr_player_gk_ref/rfdetr/rfdetr_base_448_3class/checkpoint_best_ema.pth`
- Resultados (GCS): `gs://vm-football-data/models/rfdetr_player_gk_ref/rfdetr/rfdetr_base_448_3class/results.json`
- Métricas test (all classes):
	- mAP@50: **0.7819**
	- mAP@50:95: **0.4473**
	- Precision: **0.8576**
	- Recall: **0.7604**
	- F1: **0.8035**

### Validación de evaluación role-aware

- Clips validados con el checkpoint 3 clases: SNMOT-116, SNMOT-117 y SNMOT-143.
- Modo por defecto para evaluación: `python eval_team_clustering.py`.
- Ese default ya ejecuta el flujo role-aware fused: GK por clase, árbitros excluidos del clustering y fallback legacy disponible por flags.
- Debug / overrides útiles: `--no-use-gk-class`, `--gk-assignment-mode legacy`, `--cluster-referee`.

---

## 📂 Estructura del Proyecto

```
VM_FOOTBALL/
├── core/                  # Pipeline de IA (TODOs estructurados)
│   ├── detection/         # YOLO, RF-DETR
│   ├── tracking/          # ByteTrack, SAM2
│   ├── clustering/        # TeamClassifier (HSV/DBSCAN)
│   ├── identity/          # JerseyReader, CropGenerator
│   └── pipeline.py        # Orquestador
│
├── training/              # Scripts de entrenamiento (Cloud)
│   ├── detection/         # train_rfdetr.py
│   ├── segmentation/      # finetune_sam2.py
│   └── identification/    # train_resnet_jersey.py
│
├── cv_model/              # Scripts legacy funcionales
│   ├── team_clustering.py # ✓ K-Means + HSV
│   ├── video_test.py      # ✓ Visualización
│   └── train_yolo*.py     # ✓ Entrenamiento YOLO
│
├── outputs/
│   ├── visualizations/     # Videos anotados (team clustering)
│   └── team_clustering_debug/ # Post-reports y diagnósticos CSV
│
├── models/                # Pesos .pt (NO en Git)
├── datasets/              # SoccerNet-MOT (NO en Git)
└── outputs/               # Resultados
```

---

## 🚀 Quickstart

```bash
# 1. Instalar dependencias
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Ejecutar evaluación role-aware por defecto (K=2 recomendado)
python eval_team_clustering.py --mode hsv --k 2

# 2.1 Ablation rápida (post-training)
python eval_team_clustering.py --mode hsv --k 2 --gk-assignment-mode legacy
python eval_team_clustering.py --mode hsv --k 2 --no-use-gk-class

# 2.2 Debug de referee en clustering
python eval_team_clustering.py --mode hsv --k 2 --cluster-referee

# 2.3 Cambiar el clustering por claridad visual
python eval_team_clustering.py --mode dbscan --k 2

# 2.4 Smoke test de entrenamiento RF-DETR 3 clases (Vertex AI)
.\cloud\submit_rfdetr_job.ps1 -SmokeTest

# 3. Revisar reporte post-run y diagnóstico
type outputs\team_clustering_debug\SNMOT-116_post_report.txt
python -c "import pandas as pd; print(pd.read_csv('outputs/team_clustering_debug/SNMOT-116_gk_diagnostics.csv').head())"
```

---

## �️ Stack Tecnológico

- **Detección**: YOLO (Ultralytics), RF-DETR
- **Segmentación**: SAM2
- **Embeddings**: (Planificado para identificación, no en clustering actual)
- **OCR**: SmolVLM2, ResNet
- **Clustering**: scikit-learn (K-Means, DBSCAN)
- **Portero (rol)**: heurística temporal con ventana deslizante + histéresis sobre tracks
- **RF-DETR producción actual**: 3 clases (`player`, `goalkeeper`, `referee`)

---

## 📖 Documentación

- [Plan RF-DETR Role-Aware](docs/rfdetr_gk_separation_implementation_plan.md)
- [Pipeline detallado de Team Clustering](docs/team_clustering_pipeline_detailed.md)
- [Master Prompt del Proyecto](MASTER_PROMPT.md)
- [Short Paper (LaTeX)](docs/VM_FOOTBALL_Short_Paper_updated.md)
