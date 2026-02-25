FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install gcloud CLI for GCS access within Vertex AI
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /workspace

# Python deps
COPY cloud/requirements-cloud.txt /workspace/requirements-cloud.txt
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy training code
COPY training/ /workspace/training/
COPY cloud/ /workspace/cloud/
COPY core/ /workspace/core/

# Vertex AI uses AIP_* environment variables
# AIP_MODEL_DIR = GCS dir for output model
# AIP_CHECKPOINT_DIR = GCS dir for checkpoints
# CLOUD_ML_PROJECT_ID = project ID

# Default: run the pipeline orchestrator
ENTRYPOINT ["python", "cloud/run_pipeline.py"]
CMD ["--config", "cloud/config.yaml"]
