from pathlib import Path
import cv2
import numpy as np
import shutil
import yaml
import os

def crop_center(img, crop_width_ratio=0.6, crop_height_ratio=0.6):
    """
    Recorta la parte central de la imagen según las proporciones especificadas.
    
    Args:
        img: Imagen en formato numpy array (BGR)
        crop_width_ratio: Proporción del ancho a mantener (0-1)
        crop_height_ratio: Proporción del alto a mantener (0-1)
    """
    h, w = img.shape[:2]
    
    # Calcular dimensiones del recorte
    new_width = int(w * crop_width_ratio)
    new_height = int(h * crop_height_ratio)
    
    # Calcular coordenadas para el recorte central
    start_x = (w - new_width) // 2
    start_y = (h - new_height) // 2
    
    # Realizar el recorte
    cropped = img[start_y:start_y+new_height, start_x:start_x+new_width]
    return cropped, (start_x, start_y, new_width, new_height)

def adjust_annotations(label_path, crop_info, img_width, img_height):
    """
    Ajusta las anotaciones YOLO para el nuevo tamaño recortado.
    
    Args:
        label_path: Ruta al archivo de etiquetas
        crop_info: Tupla (start_x, start_y, new_width, new_height)
        img_width: Ancho original de la imagen
        img_height: Alto original de la imagen
    """
    if not label_path.exists():
        return []
    
    start_x, start_y, new_width, new_height = crop_info
    new_lines = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            # Extraer valores originales
            class_id = parts[0]
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Verificar si el centro del objeto está dentro del recorte
            if (start_x <= x_center <= start_x + new_width and 
                start_y <= y_center <= start_y + new_height):
                
                # Ajustar coordenadas al nuevo sistema
                new_x = (x_center - start_x) / new_width
                new_y = (y_center - start_y) / new_height
                new_w = min(width / new_width, 1.0)  # Limitar a 1
                new_h = min(height / new_height, 1.0)  # Limitar a 1
                
                # Mantener track_id y team si existen
                extra_info = " ".join(parts[5:]) if len(parts) > 5 else ""
                
                new_line = f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f} {extra_info}".strip()
                new_lines.append(new_line)
    
    return new_lines

def process_dataset(root_path: Path, output_path: Path, crop_ratios=(0.6, 0.6)):
    """
    Procesa todo el dataset, creando versiones recortadas de imágenes y etiquetas.
    
    Args:
        root_path: Ruta al dataset original
        output_path: Ruta donde guardar el dataset recortado
        crop_ratios: Tupla (width_ratio, height_ratio) para el recorte
    """
    # Crear estructura de directorios
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Procesar cada split
    for split in ['train', 'val', 'test']:
        img_dir = root_path / 'images' / split
        label_dir = root_path / 'labels' / split
        out_img_dir = output_path / 'images' / split
        out_label_dir = output_path / 'labels' / split
        
        if not img_dir.exists():
            print(f"Carpeta no encontrada: {img_dir}")
            continue
        
        print(f"Procesando split: {split}")
        total_processed = 0
        
        # Procesar cada imagen
        for img_path in img_dir.glob('*.jpg'):
            # Leer y recortar imagen
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            cropped, crop_info = crop_center(img, crop_ratios[0], crop_ratios[1])
            
            # Procesar etiquetas correspondientes
            label_path = label_dir / f"{img_path.stem}.txt"
            new_labels = adjust_annotations(label_path, crop_info, img.shape[1], img.shape[0])
            
            # Si hay objetos en el recorte, guardar imagen y etiquetas
            if new_labels:
                cv2.imwrite(str(out_img_dir / img_path.name), cropped)
                with open(out_label_dir / f"{img_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(new_labels))
                total_processed += 1
        
        print(f"  Procesadas {total_processed} imágenes")
    
    # Crear archivo YAML de configuración
    yaml_content = {
        'path': output_path.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 6,  # Mantener número de clases
        'names': ['player team left', 'player team right', 
                 'goalkeeper team left', 'goalkeeper team right',
                 'referee', 'ball']
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def main():
    # Configuración de rutas
    root = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"
    output_root = root.parent / "center_crop_dataset"
    
    # Verificar dataset original
    if not root.exists():
        raise FileNotFoundError(f"Dataset original no encontrado en: {root}")
    
    # Procesar dataset
    crop_ratios = (0.6, 0.6)  # Proporción a mantener del ancho y alto
    process_dataset(root, output_root, crop_ratios)
    
    print(f"\nDataset recortado creado en: {output_root}")
    print(f"Proporciones de recorte: {crop_ratios[0]*100}% ancho, {crop_ratios[1]*100}% alto")

if __name__ == "__main__":
    main()