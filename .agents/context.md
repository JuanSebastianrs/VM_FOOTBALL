# Contexto del Proyecto VM_FOOTBALL para Antigravity

¡Hola Antigravity! Este documento sirve para ponerte en contexto rápidamente sobre el estado del proyecto para que puedas ayudar a Sebastián o a su compañero de equipo sin necesidad de leer todo el historial.

## 📌 Estado Actual (\u00DAltimo Hito)
- **Proyecto:** Entrenamiento de modelos de detección de objetos en videos de fútbol (VM_FOOTBALL).
- **\u00DAltimo logro:** Se completó exitosamente el entrenamiento del modelo **YOLO** dedicado específicamente a la detección del **balón**.
- **Objetivo Inmediato:** Entrenar un modelo **RF-DETR** enfocado **exclusivamente en los jugadores**. Para esto, las distintas variantes de jugadores y porteros deben evaluarse juntas como **un solo label** (clase `player`).

## 📊 Estructura del Dataset
- El dataset local/original contiene 6 clases: 
  `0: player_left`, `1: player_right`, `2: goalkeeper_left`, `3: goalkeeper_right`, `4: referee`, `5: ball`.
- **Estrategia para Jugadores:** A través de la configuración del pipeline (ver `cloud/config.yaml` u opciones de unificación), las clases `0` al `3` se mapean a una única clase `0` (`player`), mientras que las clases de árbitro (`4`) y balón (`5`) se ignoran o filtran.
- Existe un archivo de ejemplo (`datasets/reorganized_dataset/data_rfdetr_players_example.yaml`) que muestra cómo queda la configuración con 1 sola clase `nc: 1`.

## ☁\uFE0F Infraestructura de Datos (GCP)
- **Almacenamiento:** El dataset se aloja en un bucket de Google Cloud Storage (GCS), por ejemplo `gs://vm-football-data`.
- **Ingesta de Datos:** Durante el entrenamiento en la nube (Vertex AI o Compute Engine), las máquinas virtuales descargan automáticamente el dataset desde el bucket usando comandos como `gsutil -m cp -r`.
- **Trabajo en equipo:** El bucket se comparte gestionando los **Permisos IAM** desde Google Cloud. No es necesario enviar los archivos físicos; basta con agregar el correo del compañero como `Storage Object Viewer` o `Storage Object Admin`.

## 🚀 Próximos Pasos Sugeridos
1. Validar la lectura y el mapeo correcto del dataset para que RF-DETR reciba solo 1 clase continua de jugadores.
2. Iniciar / Monitorear los trabajos de entrenamiento de RF-DETR en los GPUs T4.
3. Evaluar las métricas de Average Precision (AP) centralizadas solo en la detección de jugadores.
