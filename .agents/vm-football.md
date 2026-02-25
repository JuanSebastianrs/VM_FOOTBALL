---
description: Flujo de trabajo para desarrollar VM_FOOTBALL - Sistema de visión por computadora para análisis de fútbol
---

# VM_FOOTBALL Development Workflow

## Prerequisitos
- Python 3.10+
- CUDA compatible GPU (recomendado)
- Git configurado

---

## Paso 1: Configurar Entorno
```bash
cd d:\sebastian\Tesis\VM_FOOTBALL
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Paso 2: Descargar Modelos Pre-entrenados
Los pesos `.pt` NO se suben a Git. Descargarlos de Google Drive y colocar en `models/`:
- `yolo11n_football.pt` - Detector principal
- `ball_detector.pt` - Detector especializado de balón
- `rf_detr_football.pt` - Detector basado en transformadores (por entrenar)

---

## Paso 3: Ejecutar Clustering de Equipos
// turbo
```bash
python cv_model/team_clustering.py
```
**Output esperado:** 
- `outputs/team_clustering/SNMOT-116_track_to_team.csv`
- `outputs/team_clustering/SNMOT-116_frame_assignments.json`

---

## Paso 4: Generar Video de Visualización
// turbo
```bash
python cv_model/video_test.py
```
**Output esperado:** `outputs/visualizations/SNMOT-116_viz.mp4`

---

## Paso 5: Entrenar Modelos (En la Nube)

### Opción A: Google Colab
1. Subir dataset comprimido a Google Drive
2. Abrir notebook en Colab
3. Montar Drive: `from google.colab import drive; drive.mount('/content/drive')`
4. Ejecutar entrenamiento
5. Los pesos se guardan automáticamente en Drive

### Opción B: Kaggle (para +12 horas)
1. Crear notebook en Kaggle
2. Subir dataset como Dataset de Kaggle
3. Configurar GPU P100
4. Usar "Save Version" para entrenamientos largos

---

## Paso 6: Evaluar Modelos
// turbo
```bash
python runs/compare_models.py
```

---

## Comandos Útiles

### Ver estructura del dataset
// turbo
```bash
dir datasets\reorganized_dataset /s
```

### Ver resultados de entrenamiento
// turbo
```bash
dir runs\train /s | findstr "best.pt"
```

### Verificar GPU disponible
// turbo
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Estructura de Salidas

```
outputs/
├── team_clustering/     # CSV y JSON con asignaciones de equipos
├── visualizations/      # Videos anotados .mp4
├── crops/               # Recortes de jugadores/dorsales
└── evaluations/         # Métricas y comparativas
```
