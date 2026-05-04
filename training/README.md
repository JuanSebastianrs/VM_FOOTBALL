# Training - VM_FOOTBALL

Scripts y notebooks para entrenamiento de modelos en **Kaggle/Colab** y en **Vertex AI (GCP)**.

## Estado Actual RF-DETR (2026-04-05)

- RF-DETR role-aware (3 clases) completado: `player`, `goalkeeper`, `referee`.
- Checkpoint principal (GCS): `gs://vm-football-data/models/rfdetr_player_gk_ref/rfdetr/rfdetr_base_448_3class/checkpoint_best_ema.pth`
- Resultados (GCS): `gs://vm-football-data/models/rfdetr_player_gk_ref/rfdetr/rfdetr_base_448_3class/results.json`
- Smoke test de Vertex validado end-to-end (descarga, conversión, train, verificación, upload).

## Validación de evaluación

- El checkpoint 3 clases ya fue validado en SNMOT-116, SNMOT-117 y SNMOT-143 con `eval_team_clustering.py`.
- La ruta por defecto ya es role-aware fused: GK por clase, árbitros excluidos del clustering de equipos.
- Debug / overrides disponibles: `--no-use-gk-class`, `--gk-assignment-mode legacy`, `--cluster-referee`.

## Entrenamiento en Vertex AI (recomendado)

```powershell
# Smoke test (2 epochs)
pwsh .\cloud\submit_rfdetr_job.ps1 -SmokeTest

# Entrenamiento completo
pwsh .\cloud\submit_rfdetr_job.ps1
```

Notas:
- Se mantiene el mismo proyecto/cuenta/bucket de GCS.
- El linaje nuevo se guarda en subcarpetas versionadas bajo `models/rfdetr_player_gk_ref`.
- Dependencias sensibles en cloud: `transformers<5` y `numpy<2`.

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

## ⚙️ Hiperparámetros Clave (referencia)

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
        ├── train_resnet_jersey.py
        └── preprocess_sn_jersey.py
    └── train_sn_jersey.py

## Jersey Number - Preprocessing

Preprocess sn-jersey tracklets and select top frames per tracklet:

```bash
python training/identification/preprocess_sn_jersey.py \
    --input-root datasets/sn_jersey_2023/jersey-2023 \
    --output-root datasets/sn_jersey_2023/processed \
    --keep-ratio 0.2 --min-keep 4 --max-keep 12
```

## Jersey Number - Training

Baseline using selected frames (processed mode):

```bash
python training/identification/train_sn_jersey.py \
    --mode processed \
    --dataset-root datasets/sn_jersey_2023/jersey-2023 \
    --processed-root datasets/sn_jersey_2023/processed \
    --run-name processed_baseline
```

Baseline using all frames (full mode):

```bash
python training/identification/train_sn_jersey.py \
    --mode full \
    --dataset-root datasets/sn_jersey_2023/jersey-2023 \
    --run-name full_baseline
```
```
