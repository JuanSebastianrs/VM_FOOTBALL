# Training - VM_FOOTBALL

Scripts y notebooks para entrenamiento de modelos en **Google Colab / Kaggle**.

## 🚀 Notebook Principal

### `detection/kaggle_training_notebook.py`

Notebook completo para entrenar detectores en Kaggle:

1. **Análisis Exploratorio** - Distribución de clases, visualización de muestras
2. **YOLO v11** - Entrenamiento con Ultralytics, hiperparámetros optimizados
3. **RF-DETR** - Conversión YOLO→COCO + entrenamiento con Roboflow
4. **Comparación** - Métricas y visualización lado a lado
5. **Exportación** - Pesos listos para producción

## 📋 Uso en Kaggle

```bash
# 1. Crear nuevo notebook en Kaggle
# 2. Añadir dataset "tactical-vision" desde tus datasets
# 3. Copiar contenido de kaggle_training_notebook.py
# 4. Ejecutar con GPU P100
# 5. Descargar modelos desde /kaggle/working/outputs/
```

## ⚙️ Hiperparámetros Clave

| Param | YOLO | RF-DETR |
|-------|------|---------|
| Epochs | 100 | 50 |
| Batch | 16 | 8 |
| LR | 0.01 | 1e-4 |
| ImgSize | 640 | 560 |

## 📂 Estructura

```
training/
├── detection/
│   ├── kaggle_training_notebook.py  # ⭐ Principal
│   ├── train_rfdetr.py
│   └── convert_to_coco.py
├── segmentation/
│   └── finetune_sam2.py
└── identification/
    └── train_resnet_jersey.py
```
