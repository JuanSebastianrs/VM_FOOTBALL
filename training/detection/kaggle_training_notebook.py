# %% [markdown]
# # VM_FOOTBALL - Entrenamiento RF-DETR
# 
# **Proyecto de Tesis: Sistema de Vision por Computadora para Analisis Tactico de Futbol**
# 
# Este notebook entrena **RF-DETR** con **multiples experimentos** automaticamente.
# 
# **Clases (2):** `player`, `goalkeeper`
# 
# ---

# %% [code]
# ============================================================
# 1. INSTALACION DE DEPENDENCIAS
# ============================================================
!pip install -q rfdetr>=1.4.0
!pip install -q supervision>=0.25.0

print("[OK] Dependencias instaladas")

# %% [code]
# ============================================================
# 2. IMPORTS Y CONFIGURACION
# ============================================================
import os
import json
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
import torch
import gc

# RF-DETR
from rfdetr import RFDETRBase, RFDETRLarge

# Supervision
import supervision as sv

# ============================================================
# REPRODUCIBILIDAD
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# VERIFICAR GPU
# ============================================================
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.1f} GB")
    
    # Optimizaciones
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Configurar expandable_segments para evitar fragmentacion
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
else:
    print("[WARN] No GPU detectada - usando CPU")

# %% [code]
# ============================================================
# 3. RUTAS (KAGGLE)
# ============================================================
KAGGLE_INPUT = Path("/kaggle/input/tactical-vision/reorganized_dataset")
WORKING_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORKING_DIR / "outputs"
COCO_DIR = WORKING_DIR / "dataset_coco"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
COCO_DIR.mkdir(exist_ok=True, parents=True)

# Dataset YOLO
YOLO_IMAGES_TRAIN = KAGGLE_INPUT / "images" / "train"
YOLO_IMAGES_VAL = KAGGLE_INPUT / "images" / "test"
YOLO_LABELS_TRAIN = KAGGLE_INPUT / "labels" / "train"
YOLO_LABELS_VAL = KAGGLE_INPUT / "labels" / "test"

print(f"Dataset: {KAGGLE_INPUT}")

# %% [code]
# ============================================================
# 4. CLASES (SOLO 2: player, goalkeeper)
# ============================================================
UNIFIED_CLASSES = ["player", "goalkeeper"]
NUM_CLASSES = len(UNIFIED_CLASSES)

# Mapeo
CLASS_MAPPING = {
    0: 0,   # player_left -> player
    1: 0,   # player_right -> player
    2: 1,   # goalkeeper_left -> goalkeeper
    3: 1,   # goalkeeper_right -> goalkeeper
    4: -1,  # referee -> IGNORAR
    5: -1,  # ball -> IGNORAR
}

COCO_CATEGORIES = [
    {"id": i, "name": name, "supercategory": "football"} 
    for i, name in enumerate(UNIFIED_CLASSES)
]

print(f"Clases: {UNIFIED_CLASSES}")

# %% [code]
# ============================================================
# 5. ANALISIS RAPIDO
# ============================================================
def count_annotations():
    stats = {"train": {c: 0 for c in UNIFIED_CLASSES}, "val": {c: 0 for c in UNIFIED_CLASSES}}
    
    for split, labels_dir in [("train", YOLO_LABELS_TRAIN), ("val", YOLO_LABELS_VAL)]:
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        orig = int(parts[0])
                        unified = CLASS_MAPPING.get(orig, -1)
                        if unified >= 0:
                            stats[split][UNIFIED_CLASSES[unified]] += 1
    return stats

stats = count_annotations()
print(f"Train: player={stats['train']['player']:,}, goalkeeper={stats['train']['goalkeeper']:,}")
print(f"Val: player={stats['val']['player']:,}, goalkeeper={stats['val']['goalkeeper']:,}")

# %% [code]
# ============================================================
# 6. CONVERSION YOLO -> COCO
# ============================================================
def convert_yolo_to_coco(images_dir, labels_dir, output_dir, split):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    coco = {
        "info": {"description": "VM_FOOTBALL", "version": "2.0"},
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    ann_id = 0
    images = list(images_dir.glob("*.jpg"))
    
    print(f"   Procesando {len(images)} imagenes para {split}...")
    
    for img_id, img_path in enumerate(tqdm(images, desc=split)):
        img = Image.open(img_path)
        w, h = img.size
        
        dest = split_dir / img_path.name
        if not dest.exists():
            shutil.copy(img_path, dest)
        
        coco["images"].append({"id": img_id, "width": w, "height": h, "file_name": img_path.name})
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig = int(parts[0])
                        unified = CLASS_MAPPING.get(orig, -1)
                        if unified < 0:
                            continue
                        
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x_min = (xc - bw/2) * w
                        y_min = (yc - bh/2) * h
                        
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": unified,
                            "bbox": [x_min, y_min, bw*w, bh*h],
                            "area": bw*w * bh*h,
                            "iscrowd": 0
                        })
                        ann_id += 1
    
    json_path = split_dir / "_annotations.coco.json"
    with open(json_path, "w") as f:
        json.dump(coco, f)
    
    print(f"   [OK] {split}: {len(coco['images'])} imgs, {len(coco['annotations'])} anns")
    return json_path

# Limpiar directorio anterior si existe
if COCO_DIR.exists():
    print("Limpiando directorio COCO anterior...")
    shutil.rmtree(COCO_DIR)
    COCO_DIR.mkdir(parents=True, exist_ok=True)

print("Convirtiendo a COCO...")

# Train
convert_yolo_to_coco(YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN, COCO_DIR, "train")

# Valid
convert_yolo_to_coco(YOLO_IMAGES_VAL, YOLO_LABELS_VAL, COCO_DIR, "valid")

# Test (RF-DETR lo requiere)
print("Creando test set...")
test_dir = COCO_DIR / "test"
test_dir.mkdir(parents=True, exist_ok=True)

with open(COCO_DIR / "valid" / "_annotations.coco.json") as f:
    valid_coco = json.load(f)

valid_imgs = list((COCO_DIR / "valid").glob("*.jpg"))
n_test = max(100, len(valid_imgs) // 10)
test_sample = valid_imgs[:n_test]

valid_img_map = {img["file_name"]: img for img in valid_coco["images"]}

coco_test = {
    "info": {"description": "VM_FOOTBALL Test", "version": "2.0"},
    "categories": COCO_CATEGORIES,
    "images": [],
    "annotations": []
}

ann_id = 0
for new_img_id, img_path in enumerate(tqdm(test_sample, desc="test")):
    shutil.copy(img_path, test_dir / img_path.name)
    
    orig_img = valid_img_map.get(img_path.name)
    if orig_img:
        coco_test["images"].append({
            "id": new_img_id,
            "width": orig_img["width"],
            "height": orig_img["height"],
            "file_name": img_path.name
        })
        
        for ann in valid_coco["annotations"]:
            if ann["image_id"] == orig_img["id"]:
                coco_test["annotations"].append({
                    "id": ann_id,
                    "image_id": new_img_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0
                })
                ann_id += 1

with open(test_dir / "_annotations.coco.json", "w") as f:
    json.dump(coco_test, f)

print(f"   [OK] test: {len(coco_test['images'])} imgs, {len(coco_test['annotations'])} anns")

# Verificar estructura
print("\nVerificando estructura COCO:")
for split in ["train", "valid", "test"]:
    split_dir = COCO_DIR / split
    json_exists = (split_dir / "_annotations.coco.json").exists()
    n_imgs = len(list(split_dir.glob("*.jpg")))
    status = "[OK]" if json_exists and n_imgs > 0 else "[ERROR]"
    print(f"   {status} {split}/: {n_imgs} imagenes, json={json_exists}")

# %% [code]
# ============================================================
# 7. CONFIGURACION DE EXPERIMENTOS
# ============================================================
# AJUSTADO PARA P100 (16GB VRAM)
# RF-DETR usa mucha memoria, batch_size debe ser conservador

EXPERIMENTS = {
    "exp1_base_conservative": {
        "description": "RF-DETR Base - batch=4 (conservador para P100)",
        "model_class": RFDETRBase,
        "epochs": 30,
        "batch_size": 4,            # Reducido de 8 a 4
        "grad_accum_steps": 4,      # Effective batch = 16
        "lr": 1e-4,
        "lr_encoder": 1e-5,
        "resolution": 560,
        "weight_decay": 1e-4,
        "use_ema": True,
    },
    
    "exp2_base_longer": {
        "description": "RF-DETR Base - mas epocas",
        "model_class": RFDETRBase,
        "epochs": 50,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "lr": 1e-4,
        "lr_encoder": 1e-5,
        "resolution": 560,
        "weight_decay": 1e-4,
        "use_ema": True,
    },
    
    "exp3_large_model": {
        "description": "RF-DETR Large (mas parametros)",
        "model_class": RFDETRLarge,
        "epochs": 30,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "lr": 5e-5,
        "lr_encoder": 5e-6,
        "resolution": 560,
        "weight_decay": 1e-4,
        "use_ema": True,
    },
}

print("Experimentos a ejecutar (optimizado para P100 16GB):")
for name, cfg in EXPERIMENTS.items():
    eff_batch = cfg["batch_size"] * cfg["grad_accum_steps"]
    print(f"   - {name}: {cfg['description']}")
    print(f"     batch={cfg['batch_size']}x{cfg['grad_accum_steps']}={eff_batch}, epochs={cfg['epochs']}")

# %% [code]
# ============================================================
# 8. EJECUTAR TODOS LOS EXPERIMENTOS
# ============================================================
results_all = {}

for exp_name, config in EXPERIMENTS.items():
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO: {exp_name}")
    print(f"   {config['description']}")
    print("=" * 70)
    
    # Limpiar memoria GPU agresivamente
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Directorio de salida
    exp_dir = OUTPUT_DIR / "rfdetr" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Crear modelo
        model = config["model_class"]()
        
        # Entrenar
        start_time = datetime.now()
        
        model.train(
            dataset_dir=str(COCO_DIR),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            grad_accum_steps=config["grad_accum_steps"],
            lr=config["lr"],
            lr_encoder=config["lr_encoder"],
            resolution=config["resolution"],
            weight_decay=config["weight_decay"],
            use_ema=config["use_ema"],
            output_dir=str(exp_dir),
            device="cuda:0",
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60
        
        # Guardar resultados
        results_all[exp_name] = {
            "status": "success",
            "training_time_min": training_time,
            "config": config["description"],
        }
        
        print(f"\n[OK] {exp_name} completado en {training_time:.1f} minutos")
        
        # Copiar mejor modelo
        best_ckpt = exp_dir / "checkpoint_best_ema.pth"
        if not best_ckpt.exists():
            best_ckpt = exp_dir / "checkpoint_best_regular.pth"
        
        if best_ckpt.exists():
            final_name = f"rfdetr_{exp_name}.pth"
            shutil.copy(best_ckpt, OUTPUT_DIR / final_name)
            print(f"   Guardado: {final_name}")
        
        # Liberar modelo
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"\n[ERROR] {exp_name}: {str(e)}")
        results_all[exp_name] = {"status": "error", "error": str(e)}
        torch.cuda.empty_cache()
        gc.collect()

# %% [code]
# ============================================================
# 9. RESUMEN DE TODOS LOS EXPERIMENTOS
# ============================================================
print("\n" + "=" * 70)
print("RESUMEN DE EXPERIMENTOS")
print("=" * 70)

for exp_name, result in results_all.items():
    status = "[OK]" if result["status"] == "success" else "[ERROR]"
    if result["status"] == "success":
        print(f"{status} {exp_name}: {result['training_time_min']:.1f} min")
    else:
        print(f"{status} {exp_name}: {result.get('error', 'Unknown')}")

# Guardar resumen
with open(OUTPUT_DIR / "experiments_summary.json", "w") as f:
    json.dump(results_all, f, indent=2, default=str)

# %% [code]
# ============================================================
# 10. VISUALIZACION COMPARATIVA
# ============================================================
print("\nGenerando visualizaciones comparativas...")

val_images = list((COCO_DIR / "valid").glob("*.jpg"))[:1]

if val_images:
    test_img_path = val_images[0]
    test_img = Image.open(test_img_path)
    
    n_experiments = len([r for r in results_all.values() if r["status"] == "success"])
    
    if n_experiments > 0:
        fig, axes = plt.subplots(1, n_experiments + 1, figsize=(6 * (n_experiments + 1), 6))
        if n_experiments == 0:
            axes = [axes]
        
        axes[0].imshow(np.array(test_img))
        axes[0].set_title("Original", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        idx = 1
        for exp_name, result in results_all.items():
            if result["status"] != "success":
                continue
            
            ckpt = OUTPUT_DIR / f"rfdetr_{exp_name}.pth"
            if ckpt.exists():
                try:
                    cfg = EXPERIMENTS[exp_name]
                    model = cfg["model_class"](pretrain_weights=str(ckpt))
                    detections = model.predict(test_img, threshold=0.5)
                    
                    img_np = np.array(test_img)
                    annotated = sv.BoxAnnotator().annotate(img_np.copy(), detections)
                    
                    axes[idx].imshow(annotated)
                    axes[idx].set_title(f"{exp_name}\n({len(detections)} det)", fontsize=10)
                    axes[idx].axis('off')
                    
                    del model
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"   Error visualizando {exp_name}: {e}")
                
                idx += 1
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "comparison_all_experiments.png", dpi=150, bbox_inches='tight')
        plt.show()

# %% [code]
# ============================================================
# 11. ARCHIVOS FINALES
# ============================================================
print("\nArchivos generados:")
for f in sorted(OUTPUT_DIR.rglob("*.pth")):
    size_mb = f.stat().st_size / 1e6
    print(f"   [MODEL] {f.name}: {size_mb:.1f} MB")

for f in sorted(OUTPUT_DIR.rglob("*.json")):
    print(f"   [CONFIG] {f.name}")

for f in sorted(OUTPUT_DIR.rglob("*.png")):
    print(f"   [IMAGE] {f.name}")

print("\nTODOS LOS EXPERIMENTOS COMPLETADOS")
print(f"Descarga los modelos desde: {OUTPUT_DIR}")
