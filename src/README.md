# src - Estado Actual

Este directorio se conserva como placeholder.

En el estado actual del repositorio, la implementacion activa vive en:
- `core/` para modulos de pipeline (deteccion, tracking, clustering, identidad).
- scripts de evaluacion en la raiz (`eval_team_clustering.py`, `eval_rfdetr_video.py`).
- `cloud/` para entrenamiento y ejecucion en GCP/Vertex.

Si en el futuro se migra a una estructura de aplicacion API/servicio, `src/` sera el punto de entrada de ese layout.
