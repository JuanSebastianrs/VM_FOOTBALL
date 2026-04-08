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
