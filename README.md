# VM_FOOTBALL ⚽🤖

**Sistema de Visión por Computadora para Análisis Táctico de Fútbol**

> Proyecto de tesis adaptado al contexto colombiano y latinoamericano.

---

## 🎯 Estado Actual

| Módulo | Estado | Archivos |
|--------|--------|----------|
| **Detección** | ✅ Funcional | `cv_model/train_yolo*.py` |
| **Tracking** | ✅ Funcional | ByteTrack (Ultralytics) |
| **Equipos** | ✅ Prototipo | `cv_model/team_clustering.py` |
| **Dorsales** | 📋 Planificado | `core/identity/` |

---

## 📂 Estructura del Proyecto

```
VM_FOOTBALL/
├── core/                  # Pipeline de IA (TODOs estructurados)
│   ├── detection/         # YOLO, RF-DETR
│   ├── tracking/          # ByteTrack, SAM2
│   ├── clustering/        # TeamClassifier, SigLIP2
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

# 2. Ejecutar clustering de equipos
python cv_model/team_clustering.py

# 3. Generar video de visualización
python cv_model/video_test.py
```

---

## �️ Stack Tecnológico

- **Detección**: YOLO (Ultralytics), RF-DETR
- **Segmentación**: SAM2
- **Embeddings**: SigLIP2
- **OCR**: SmolVLM2, ResNet
- **Clustering**: scikit-learn (K-Means)

---

## 📖 Documentación

- [Plan de Implementación](docs/implementation_plan.md)
- [Tesis (PDF)](docs/tesis.pdf)
