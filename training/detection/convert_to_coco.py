# convert_to_coco.py
"""
Convertir dataset SoccerNet-MOT a formato COCO para RF-DETR.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import argparse


def convert_yolo_to_coco(
    yolo_dataset_path: Path,
    output_path: Path,
    class_names: List[str],
) -> None:
    """
    Convertir dataset formato YOLO a formato COCO.
    
    Formato YOLO (por imagen):
        clase x_center y_center width height (normalizados)
    
    Formato COCO:
        {
            "images": [{"id": int, "file_name": str, "width": int, "height": int}],
            "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x,y,w,h]}],
            "categories": [{"id": int, "name": str}]
        }
    
    TODO:
        1. Leer estructura YOLO (images/, labels/)
        2. Para cada imagen:
            - Obtener dimensiones
            - Leer anotaciones
            - Convertir a formato COCO
        3. Guardar JSON
    
    Args:
        yolo_dataset_path: Path al dataset YOLO
        output_path: Path para guardar annotations.json
        class_names: Lista de nombres de clases
    """
    # TODO: Implementar
    # 
    # coco = {
    #     "images": [],
    #     "annotations": [],
    #     "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)]
    # }
    # 
    # images_dir = yolo_dataset_path / "images"
    # labels_dir = yolo_dataset_path / "labels"
    # 
    # annotation_id = 0
    # for img_id, img_path in enumerate(sorted(images_dir.glob("*.jpg"))):
    #     # Leer imagen para dimensiones
    #     img = cv2.imread(str(img_path))
    #     h, w = img.shape[:2]
    #     
    #     coco["images"].append({
    #         "id": img_id,
    #         "file_name": img_path.name,
    #         "width": w,
    #         "height": h
    #     })
    #     
    #     # Leer anotaciones YOLO
    #     label_path = labels_dir / f"{img_path.stem}.txt"
    #     if label_path.exists():
    #         for line in label_path.read_text().strip().split("\n"):
    #             cls, xc, yc, bw, bh = map(float, line.split())
    #             # Convertir de normalizado a píxeles
    #             x = (xc - bw/2) * w
    #             y = (yc - bh/2) * h
    #             bbox_w = bw * w
    #             bbox_h = bh * h
    #             
    #             coco["annotations"].append({
    #                 "id": annotation_id,
    #                 "image_id": img_id,
    #                 "category_id": int(cls),
    #                 "bbox": [x, y, bbox_w, bbox_h],
    #                 "area": bbox_w * bbox_h,
    #                 "iscrowd": 0
    #             })
    #             annotation_id += 1
    # 
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # output_path.write_text(json.dumps(coco, indent=2))
    
    raise NotImplementedError("TODO: Convertir YOLO a COCO")


def main():
    parser = argparse.ArgumentParser(description="Convertir YOLO a COCO")
    parser.add_argument("--input", type=Path, required=True, help="Dataset YOLO")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["player_left", "player_right", "goalkeeper_left", "goalkeeper_right", "referee", "ball"]
    )
    
    args = parser.parse_args()
    convert_yolo_to_coco(args.input, args.output, args.classes)


if __name__ == "__main__":
    main()
