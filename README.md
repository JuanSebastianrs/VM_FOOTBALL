# VM_FOOTBALL ⚽🤖

**Sistema de Visión por Computadora para Análisis Táctico de Fútbol**

> Proyecto de tesis de maestría enfocado en la reconstrucción analítica del juego.

---

## 📖 Documentación Centralizada

Toda la documentación técnica profunda ha sido unificada y actualizada en el directorio [`.agents/`](file:///d:/sebastian/Tesis/VM_FOOTBALL/.agents/):

- **[Pipeline Principal](file:///d:/sebastian/Tesis/VM_FOOTBALL/.agents/pipeline_documentation.md)**: Explicación detallada de las 7 fases (YOLO26, RT-DETR, Viterbi HMM, SAM2 y **Mapping 2D**).
- **[Arquitectura YOLO26](file:///d:/sebastian/Tesis/VM_FOOTBALL/.agents/architecture/yolo26_architecture.md)**: Detalles sobre el detector de balón SOTA (STAL, MuSGD, ProgLoss).
- **[Reporte de Entrenamiento](file:///d:/sebastian/Tesis/VM_FOOTBALL/.agents/reports/ball_detection_report.md)**: Histórico y configuración de entrenamiento (GCP Vertex AI).
- **[Guía de Desarrollo](file:///d:/sebastian/Tesis/VM_FOOTBALL/.agents/vm-football.md)**: Workflow, entorno y comandos útiles.
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

## 🎯 Estado del Pipeline

| Componente | Modelo / Algoritmo | Propósito |
|------------|-------------------|-----------|
| **Balón** | YOLO26 Nano (P2) | Detección de precisión SOTA |
| **Jugadores** | RT-DETR Large | Detección robusta (Transformer) |
| **Tracking** | Viterbi HMM | Suavizado y manejo de oclusiones |
| **Segmentación**| SAM2 Hiera Small | Máscaras de píxel zero-shot |
| **Mapping 2D** | YOLOv11-Pose | Reconstrucción métrica del campo |

---

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar modelos
# Descargue los pesos .pt en la carpeta /models/

# 3. Ejecutar pipeline completo
python src/tactical_vision_pipeline.py --sequence_dir datasets/SNMOT-197 --yolo_weights models/yolo26.pt --rtdetr_weights models/rtdetr-l.pt --sam2_weights models/sam2.1_hiera_small.pt
```

---

## 📂 Organización del Repositorio

- `core/`: Implementación de los módulos de IA (detection, tracking, mapping).
- `src/`: Orquestadores y scripts de ejecución principal.
- `training/`: Scrips de entrenamiento y notebooks para Cloud.
- `.agents/`: Centro de documentación técnica y contexto del proyecto.
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
