# VM_FOOTBALL - Código Fuente de Producción

Este directorio contiene el código de la aplicación en producción.

## Estructura

```
src/
├── main.py              # FastAPI entrypoint
├── config.py            # Configuración global
├── api/                 # Endpoints REST
│   ├── routes.py
│   └── schemas.py
├── core/                # Pipeline de IA
│   ├── pipeline.py      # Orquestador maestro
│   ├── motion/          # Detección y Tracking
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   └── segmenter.py
│   └── identity/        # Equipos y Dorsales
│       ├── team_classifier.py
│       └── jersey_reader.py
├── workers/             # Tareas asíncronas (Celery)
│   └── video_processor.py
└── utils/               # Utilidades
    ├── video_io.py
    └── visualization.py
```

## Ejecución Local

```bash
cd src
uvicorn main:app --reload --port 8000
```
