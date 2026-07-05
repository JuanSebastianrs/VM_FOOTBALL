# scripts/build_parseq_cache.py
"""
Segundo lector SOTA: PARSeq (scene-text recognition, Koshkina & Elder 2024
usa este enfoque para dorsales — ver docs/2404.08401v5.pdf).

Corre PARSeq sobre los crops de un split del dataset y cachea una distribucion
(T, 99) por tracklet PARALELA al cache del modelo propio (mismo orden de
crop_paths), para ensamblar en la fusion temporal:

    p_frame ∝ p_modelo^(1-w) * p_parseq^w

Lectura -> distribucion: si PARSeq lee 1-2 digitos validos (1..99) con
confianza c (producto de probs por caracter), p[num]=c y el resto uniforme;
si no lee nada valido, distribucion uniforme (neutral en la mezcla).

  python scripts/build_parseq_cache.py --split test \
      --output outputs/_verify/jersey_test_eval/parseq_probs_test.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.infer_jersey_on_dataset import build_splits  # noqa: E402


def parse_number(label: str):
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits or len(digits) > 2:
        return None
    n = int(digits)
    return n if 1 <= n <= 99 else None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/jersey_tracking_v2_224")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", default="parseq_tiny",
                    help="parseq_tiny | parseq (base, mas preciso y lento)")
    ap.add_argument("--checkpoint", default=None,
                    help="pesos fine-tuneados (runs/parseq_jersey_ft/best.pt) "
                         "a cargar sobre --model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--min_legibility", type=float, default=0.30,
                    help="crops por debajo -> distribucion uniforme sin inferir")
    ap.add_argument("--legibility_scores", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("baudm/parseq", args.model, pretrained=True,
                           trust_repo=True)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        print(f"[parseq] fine-tuned weights: {args.checkpoint} "
              f"(val {ckpt.get('val_acc', float('nan')):.3f})")
    model = model.eval().to(device)
    img_tf = None
    try:  # transform oficial del repo de parseq
        from strhub.data.module import SceneTextDataModule
        img_tf = SceneTextDataModule.get_transform(model.hparams.img_size)
    except Exception:
        import torchvision.transforms as T
        img_tf = T.Compose([T.Resize((32, 128)), T.ToTensor(),
                            T.Normalize(0.5, 0.5)])

    leg = {}
    if args.legibility_scores and Path(args.legibility_scores).exists():
        import json
        leg = json.loads(Path(args.legibility_scores).read_text())

    root = Path(args.dataset_dir)
    records = [r for r in build_splits(root / "tracklets.json", root / "splits.json")
               if r.get("split") == args.split]
    print(f"{args.split}: {len(records)} tracklets | modelo {args.model} en {device}")

    UNIFORM = np.full(99, 1.0 / 99.0)
    out = {}
    n_read = n_crops = 0
    for i, rec in enumerate(records):
        key = f"{rec['sequence']}|{rec['track_id']}"
        paths = rec.get("crop_paths", [])
        probs = np.tile(UNIFORM, (len(paths), 1))
        todo, idxs = [], []
        for j, cp in enumerate(paths):
            if leg and leg.get(Path(cp).as_posix(), 1.0) < args.min_legibility:
                continue
            p = root / cp
            if p.exists():
                todo.append(p)
                idxs.append(j)
        for b in range(0, len(todo), args.batch_size):
            batch_paths = todo[b:b + args.batch_size]
            x = torch.stack([img_tf(Image.open(p).convert("RGB"))
                             for p in batch_paths]).to(device)
            logits = model(x)
            pred = logits.softmax(-1)
            labels, confs = model.tokenizer.decode(pred)
            for k, (lab, cf) in enumerate(zip(labels, confs)):
                n_crops += 1
                num = parse_number(lab)
                if num is None:
                    continue
                c = float(cf.prod().clamp(0, 1)) if hasattr(cf, "prod") else float(cf)
                if c <= 0:
                    continue
                row = np.full(99, (1.0 - c) / 98.0)
                row[num - 1] = c
                probs[idxs[b + k]] = row
                n_read += 1
        out[key] = probs
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)} tracklets ({n_read}/{n_crops} lecturas validas)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"[parseq] {n_read}/{n_crops} crops con lectura valida -> {args.output}")


if __name__ == "__main__":
    main()
