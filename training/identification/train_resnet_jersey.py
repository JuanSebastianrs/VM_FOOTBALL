# train_resnet_jersey.py
"""
Entrenar clasificador ResNet para números de camiseta (0-99).
"""

from pathlib import Path
from typing import Optional, Tuple
import argparse


def main(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    model_type: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """
    Entrenar ResNet para clasificación de dorsales.
    
    Dataset esperado:
        dataset_path/
        ├── train/
        │   ├── 00/  # Dorsal 0
        │   ├── 01/  # Dorsal 1
        │   ...
        │   └── 99/ # Dorsal 99
        └── val/
            └── ...
    
    TODO:
        1. Cargar dataset con ImageFolder
        2. Crear modelo ResNet con 100 clases de salida
        3. Data augmentation
        4. Entrenar con CrossEntropyLoss
        5. Guardar mejor modelo
    
    Args:
        dataset_path: Path al dataset de crops
        output_dir: Directorio para guardar modelo
        epochs: Número de épocas
        batch_size: Tamaño de batch
        learning_rate: Learning rate
        model_type: "resnet18" o "resnet34"
        pretrained: Usar pesos ImageNet
        freeze_backbone: Congelar backbone (solo entrenar FC)
    """
    # TODO: Implementar
    #
    # import torch
    # import torch.nn as nn
    # from torch.utils.data import DataLoader
    # from torchvision import models, transforms, datasets
    # from torch.optim import Adam
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # 
    # # Transforms
    # train_transforms = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.RandomHorizontalFlip(p=0.3),  # Algunos dorsales son simétricos
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # 
    # val_transforms = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # 
    # # Datasets
    # train_dataset = datasets.ImageFolder(dataset_path / "train", transform=train_transforms)
    # val_dataset = datasets.ImageFolder(dataset_path / "val", transform=val_transforms)
    # 
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 
    # # Modelo
    # if model_type == "resnet18":
    #     model = models.resnet18(pretrained=pretrained)
    # else:
    #     model = models.resnet34(pretrained=pretrained)
    # 
    # if freeze_backbone:
    #     for param in model.parameters():
    #         param.requires_grad = False
    # 
    # # Cambiar última capa para 100 clases
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 100)
    # 
    # model = model.cuda()
    # 
    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    # 
    # best_acc = 0.0
    # for epoch in range(epochs):
    #     # Train
    #     model.train()
    #     for images, labels in train_loader:
    #         images, labels = images.cuda(), labels.cuda()
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #     
    #     # Validate
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for images, labels in val_loader:
    #             images, labels = images.cuda(), labels.cuda()
    #             outputs = model(images)
    #             _, predicted = outputs.max(1)
    #             total += labels.size(0)
    #             correct += predicted.eq(labels).sum().item()
    #     
    #     acc = correct / total
    #     if acc > best_acc:
    #         best_acc = acc
    #         torch.save(model.state_dict(), output_dir / "best.pt")
    #     
    #     scheduler.step()
    #     print(f"Epoch {epoch+1}/{epochs}: Acc = {acc:.4f}")
    
    raise NotImplementedError(
        "TODO: Implementar entrenamiento ResNet para dorsales.\n"
        "Requisitos:\n"
        "1. Dataset de crops organizado por número\n"
        "2. GPU para entrenamiento\n"
        "3. torch, torchvision instalados"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar ResNet para dorsales")
    parser.add_argument("--dataset", type=Path, required=True, help="Path al dataset de crops")
    parser.add_argument("--output", type=Path, default=Path("runs/train/resnet_jersey"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    main(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
