from pathlib import Path
import shutil
import yaml

def main():
    # Raíz del dataset reorganizado
    root = Path(__file__).resolve().parents[2] / "VM_FOOTBALL" / "datasets" / "reorganized_dataset"

    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Detectar el id de 'ball' en tu dataset actual (4 clases)
    names = data_cfg.get("names")
    if not names or "ball" not in names:
        raise ValueError("Tu data.yaml no contiene la clase 'ball' en 'names'.")
    ball_id = names.index("ball")

    # Carpetas fuente (dataset completo)
    imgs_train = root / "images" / "train"
    imgs_val   = root / "images" / "test"    # usamos test como validación
    lbls_train = root / "labels" / "train"
    lbls_val   = root / "labels" / "test"

    # Carpetas destino (subset de balón)
    ball_root  = root / "ball_only"
    out_imgs_train = ball_root / "images" / "train"
    out_imgs_val   = ball_root / "images" / "val"
    out_lbls_train = ball_root / "labels" / "train"
    out_lbls_val   = ball_root / "labels" / "val"
    for d in [out_imgs_train, out_imgs_val, out_lbls_train, out_lbls_val]:
        d.mkdir(parents=True, exist_ok=True)

    def process_split(img_dir: Path, lbl_dir: Path, out_img_dir: Path, out_lbl_dir: Path):
        kept = 0
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            img_file = (img_dir / lbl_file.name).with_suffix(".jpg")
            if not img_file.exists():
                # intenta .png por si acaso
                alt = img_file.with_suffix(".png")
                if alt.exists():
                    img_file = alt
                else:
                    continue

            # filtra SOLO las líneas de balón y remapea la clase a 0
            lines_out = []
            txt = lbl_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for ln in txt:
                if not ln.strip():
                    continue
                parts = ln.split()
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    # si algo raro, sáltalo
                    continue
                if cls == ball_id:
                    # YOLO: cls x y w h => remapeamos cls a 0
                    # mantenemos sólo los primeros 5 valores
                    if len(parts) < 5:
                        continue
                    x, y, w, h = parts[1:5]
                    lines_out.append(f"0 {x} {y} {w} {h}")

            if not lines_out:
                continue  # esta imagen no tiene balón -> no se copia

            # copiar imagen y escribir nuevo label con solo clase 0
            shutil.copy2(img_file, out_img_dir / img_file.name)
            (out_lbl_dir / lbl_file.name).write_text("\n".join(lines_out), encoding="utf-8")
            kept += 1
        return kept

    ntr = process_split(imgs_train, lbls_train, out_imgs_train, out_lbls_train)
    nva = process_split(imgs_val,   lbls_val,   out_imgs_val,   out_lbls_val)

    # Escribir el YAML de balón
    ball_yaml = ball_root / "ball_only.yaml"
    ball_yaml.write_text(
        f"path: {ball_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['ball']\n",
        encoding="utf-8"
    )

    # Borrar cachés antiguos que puedan confundir a Ultralytics
    for cache in [lbls_train, lbls_val, out_lbls_train, out_lbls_val]:
        for f in cache.glob("*.cache"):
            try:
                f.unlink()
            except Exception:
                pass

    print(f"Subset balón creado en: {ball_root}")
    print(f"Etiquetas filtradas: train={ntr}  val={nva}  (solo frames con balón)")
    print(f"YAML: {ball_yaml}")

if __name__ == "__main__":
    main()
