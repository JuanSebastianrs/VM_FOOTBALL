# scripts/generate_synthetic_jerseys.py
"""
Dataset SINTETICO de dorsales para atacar las confusiones de digitos del
modelo (3<->4, 1<->7, 6<->8, 9<->0) a la resolucion real del broadcast.

Genera crops tipo torso (fondo liso/gradiente color camiseta + numero con
fuente deportiva) con la degradacion del dominio: perspectiva, rotacion,
downscale-upscale (digitos de 12-40 px), motion blur, ruido y JPEG.

Emite un indice en el MISMO formato que build_soccernet_perframe_index
(image_path relativo a --soccernet_root para reutilizar el loader):

  python scripts/generate_synthetic_jerseys.py \
      --out_dir datasets/soccernet/jersey-2023/synth_digits --n_per_number 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# digitos que el modelo confunde -> numeros que los contienen se sobre-muestrean
CONFUSABLE = set("341768 90".replace(" ", ""))

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\impact.ttf", r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf", r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\ariblk.ttf",
]


def _rand_color(rng, lo=0, hi=255):
    return tuple(rng.randint(lo, hi) for _ in range(3))


def _contrast(c1, c2):
    l1 = 0.299 * c1[0] + 0.587 * c1[1] + 0.114 * c1[2]
    l2 = 0.299 * c2[0] + 0.587 * c2[1] + 0.114 * c2[2]
    return abs(l1 - l2)


def make_crop(number: int, rng: random.Random, fonts) -> Image.Image:
    W = H = 256
    bg = _rand_color(rng)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    # gradiente/franja simple tipo camiseta
    if rng.random() < 0.5:
        c2 = _rand_color(rng)
        for y in range(0, H, 4):
            t = y / H
            col = tuple(int(bg[i] * (1 - t) + c2[i] * t) for i in range(3))
            draw.rectangle([0, y, W, y + 4], fill=col)
    if rng.random() < 0.3:  # franjas verticales
        c3 = _rand_color(rng)
        for x in range(0, W, rng.randint(30, 60)):
            draw.rectangle([x, 0, x + rng.randint(10, 25), H], fill=c3)

    # numero con contraste garantizado
    for _ in range(10):
        fg = _rand_color(rng)
        if _contrast(fg, bg) > 60:
            break
    font_path = rng.choice(fonts)
    size = rng.randint(90, 170)
    font = ImageFont.truetype(font_path, size)
    text = str(number)
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    x = (W - tw) // 2 + rng.randint(-25, 25)
    y = (H - th) // 2 + rng.randint(-30, 10)
    if rng.random() < 0.4:  # borde (outline) tipico de dorsales
        draw.text((x, y), text, font=font, fill=_rand_color(rng),
                  stroke_width=rng.randint(2, 5), stroke_fill=fg)
    else:
        draw.text((x, y), text, font=font, fill=fg)

    # degradacion del dominio broadcast
    arr = img
    if rng.random() < 0.7:  # perspectiva/rotacion
        arr = arr.rotate(rng.uniform(-18, 18), resample=Image.BILINEAR,
                         fillcolor=bg, expand=False)
    # downscale a la resolucion real del digito (12-45 px de alto) y volver
    digit_h = rng.randint(12, 45)
    small_h = max(16, int(256 * digit_h / max(th, 1)))
    small_w = max(12, int(small_h * rng.uniform(0.7, 1.1)))
    arr = arr.resize((small_w, small_h), Image.BILINEAR)
    if rng.random() < 0.5:
        arr = arr.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.3:  # motion blur horizontal
        k = rng.choice([3, 5])
        kernel = [0.0] * (k * k)
        for i in range(k):
            kernel[(k // 2) * k + i] = 1.0 / k
        arr = arr.filter(ImageFilter.Kernel((k, k), kernel))
    arr = arr.resize((160, 200), Image.BILINEAR)
    a = np.array(arr).astype(np.float32)
    a += np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(2, 10), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="datasets/soccernet/jersey-2023/synth_digits")
    ap.add_argument("--soccernet_root", default="datasets/soccernet/jersey-2023")
    ap.add_argument("--n_per_number", type=int, default=200)
    ap.add_argument("--confusable_boost", type=float, default=2.0)
    ap.add_argument("--jpeg_quality", type=int, nargs=2, default=[45, 90])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default=None,
                    help="default: <soccernet_root>/perframe_index_synth.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    fonts = [f for f in FONT_CANDIDATES if Path(f).exists()]
    if not fonts:
        raise SystemExit("No hay fuentes TTF disponibles")
    out_root = Path(a.out_dir)
    sn_root = Path(a.soccernet_root)

    index = []
    total = 0
    for num in range(1, 100):
        n = a.n_per_number
        if any(d in CONFUSABLE for d in str(num)):
            n = int(n * a.confusable_boost)
        d = out_root / f"{num:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = make_crop(num, rng, fonts)
            p = d / f"s{i:04d}.jpg"
            img.save(p, quality=rng.randint(*a.jpeg_quality))
            index.append({"image_path": p.relative_to(sn_root).as_posix(),
                          "jersey": num, "legibility": 1.0,
                          "tracklet_id": f"synth_{num}_{i}"})
            total += 1
        if num % 20 == 0:
            print(f"  numero {num}: {total} imagenes")

    out_json = Path(a.output_json or (sn_root / "perframe_index_synth.json"))
    out_json.write_text(json.dumps(index))
    print(f"[synth] {total} crops sinteticos -> {out_root} | indice {out_json}")


if __name__ == "__main__":
    main()
