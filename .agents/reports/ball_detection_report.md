# Reporte Integrado de Entrenamiento: YOLO Detector de Balón

Este reporte consolida las dos fases principales de entrenamiento para el detector de balón en el proyecto TacticalVision AI: la fase inicial con **YOLO11** y la evolución final a **YOLO26**.

---

## Fase 1: Enfoque YOLOv11 (Legacy)

### Arquitectura del Modelo
El modelo base inicial fue **YOLOv11** (Nano y Small).
- **YOLO11n**: Optimizado para tiempo real.
- **YOLO11s**: Mayor capacidad de aprendizaje para rasgos complejos.

### Estrategia de Objetos Pequeños
- **Resolución (1280px)**: Para evitar que el balón sea "borrado" en las capas de Pooling.
- **Capa P2 (Custom YAML)**: Se añadió una capa de Stride 4 para analizar el balón a nivel microscópico.
- **Data Augmentation**: Se prohibió el `mixup` y se incentivó el `copy_paste` (0.3) para balancear la clase balón.

---

## Fase 2: Evolución a YOLO26 (SOTA - Actual)

Tras los resultados de YOLOv11, el pipeline migró a **YOLO26 Nano**, que representa el estado del arte actual para este proyecto.

### ¿Por qué YOLO26?
| Característica | YOLOv11 (Legacy) | YOLO26 (Actual) |
|---|---|---|
| **Arquitectura** | YOLO11n + custom P2 | **YOLO26n-p2 (Nativo)** |
| **Optimizador** | AdamW / SGD | **MuSGD** (Inspirado en Muon) |
| **Pérdida (Loss)** | Standard | **ProgLoss** (Progresiva/Currículum) |
| **Asignación** | TAL | **STAL** (Small-Target-Aware) |
| **NMS** | Post-procesado | **NMS-free** (End-to-end) |

### Optimizaciones Clave en YOLO26
1. **MuSGD Optimizer**: Aplica actualizaciones tipo Muon a parámetros de alta dimensión (convoluciones) y SGD estándar a bajas dimensiones. Mayor estabilidad de convergencia.
2. **ProgLoss**: Estrategia de currículum que enfoca objetos fáciles al inicio y transiciona a objetos difíciles (balones ocluidos o borrosos) gradualmente.
3. **STAL**: Asignación de etiquetas optimizada matemáticamente para objetos que representan <30 píxeles en cuadros 1080p.
4. **NMS-free**: Elimina la necesidad de filtrado post-procesamiento, reduciendo la latencia de exportación y ejecución.

### Configuración Final de Entrenamiento (A100 GPU)
- **Modelo**: `yolo26n.pt` ( Nano P2-STAL).
- **Epochs**: 300.
- **ImgSize**: 1280px.
- **Optimizer**: MuSGD (lr0=0.001).
- **Patience**: 50.

---

## Resultados y Pesos
Los pesos definitivos se encuentran en `models/yolo26.pt`. Estos pesos son los integrados en el pipeline de seguimiento temporal (Viterbi HMM) para garantizar la máxima precisión en la detección de candidatos.
