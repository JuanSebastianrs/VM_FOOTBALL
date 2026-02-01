# VM_FOOTBALL - Carpeta de Entrenamiento

Esta carpeta contiene los scripts para **entrenar** los modelos.
El entrenamiento debe realizarse en la nube (Colab/Kaggle).

## Estructura

```
training/
├── detection/         # Entrenar detectores (YOLO, RF-DETR)
├── segmentation/      # Fine-tuning SAM2
└── identification/    # OCR de dorsales (ResNet, SmolVLM2)
```

## Uso

1. Modifica los scripts según el modelo a entrenar
2. Sube el dataset a Google Drive
3. Ejecuta en Colab con GPU
4. Descarga los pesos `.pt` resultantes a `models/`
