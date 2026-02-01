# finetune_sam2.py
"""
Fine-tuning de SAM2 para segmentación de jugadores de fútbol.
"""

from pathlib import Path
from typing import Optional
import argparse


def main(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    model_size: str = "large",
    freeze_encoder: bool = True,
):
    """
    Fine-tune SAM2 con máscaras de jugadores de fútbol.
    
    Dataset esperado:
        dataset_path/
        ├── images/
        │   └── *.jpg
        └── masks/
            └── *.png  # Mismo nombre que imagen, máscara binaria
    
    TODO:
        1. Instalar segment-anything-2
        2. Cargar modelo SAM2 base
        3. Congelar image encoder (opcional)
        4. Entrenar mask decoder con datos de fútbol
        5. Guardar pesos fine-tuneados
    
    Args:
        dataset_path: Path al dataset con máscaras
        output_dir: Directorio de salida
        epochs: Número de épocas
        batch_size: Tamaño de batch
        learning_rate: Learning rate
        model_size: "tiny", "small", "base", "large"
        freeze_encoder: Si congelar el image encoder
    """
    # TODO: Implementar
    #
    # from sam2 import build_sam2, SAM2ImagePredictor
    # import torch
    # 
    # # Cargar modelo base
    # checkpoint = f"sam2_hiera_{model_size}.pt"
    # model = build_sam2(checkpoint)
    # 
    # if freeze_encoder:
    #     for param in model.image_encoder.parameters():
    #         param.requires_grad = False
    # 
    # # Solo entrenar mask decoder y prompt encoder
    # trainable_params = list(model.mask_decoder.parameters())
    # trainable_params += list(model.prompt_encoder.parameters())
    # 
    # optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    # 
    # # Dataset y entrenamiento...
    # # Ver documentación de SAM2 para detalles de fine-tuning
    
    raise NotImplementedError(
        "TODO: Implementar fine-tuning SAM2.\n"
        "Requisitos:\n"
        "1. pip install segment-anything-2\n"
        "2. Dataset con máscaras de jugadores\n"
        "3. GPU con suficiente VRAM (>16GB recomendado)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM2")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/train/sam2_football"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--size", choices=["tiny", "small", "base", "large"], default="large")
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_size=args.size,
        freeze_encoder=args.freeze_encoder,
    )
