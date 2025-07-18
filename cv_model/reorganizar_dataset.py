import os
import shutil
from pathlib import Path

def reorganizar_dataset(source_dir, split_name, output_base):
    output_img_dir = output_base / "images" / split_name
    output_label_dir = output_base / "labels" / split_name
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for seq_path in sorted(source_dir.glob("SNMOT-*")):
        seq_id = seq_path.name
        img_dir = seq_path / "img1"
        label_dir = seq_path / "labels"

        if not img_dir.exists() or not label_dir.exists():
            print(f"Omitido (faltan img1 o labels): {seq_id}")
            continue

        for img_file in sorted(img_dir.glob("*.jpg")):
            frame_num = img_file.stem
            new_img_name = f"{seq_id}_{frame_num}.jpg"
            new_label_name = f"{seq_id}_{frame_num}.txt"

            label_file = label_dir / f"{frame_num}.txt"
            if not label_file.exists():
                continue

            shutil.copy(img_file, output_img_dir / new_img_name)
            shutil.copy(label_file, output_label_dir / new_label_name)

        print(f"{split_name.upper()} procesado: {seq_id}")

if __name__ == "__main__":
    root_path = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "data" / "tracking" / "SoccerNet" / "tracking"
    output_base = root_path.parents[1] / "reorganized_dataset"

    for split_name in ["train", "test"]:
        split_dir = root_path / split_name / split_name
        reorganizar_dataset(split_dir, split_name, output_base)

    print("✅ Dataset reorganizado correctamente por split (train/test).")
