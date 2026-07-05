# scripts/eval_sr_reader_ab.py
"""
A/B de super-resolucion para el lector de dorsales (PARSeq fine-tuneado).

Mide exact-match del lector sobre los crops legibles de un split, leyendo:
  A) el crop original (224px), y
  B) el crop pasado por Real-ESRGAN x4 (spandrel) antes del transform.

La palanca documentada en docs/jersey_perframe_v2.md SS21: el techo es la
resolucion de los digitos (8-30 px nativos). Si el SR no gana aqui (val),
se descarta barato sin tocar test.

  python scripts/eval_sr_reader_ab.py --split val \
      --checkpoint runs/parseq_jersey_ft/best.pt \
      --sr_model models/RealESRGAN_x4plus.pth
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

from training.identification.finetune_parseq_jersey import (  # noqa: E402
    build_tracking_samples, parse_number)


def load_reader(model_name, checkpoint, device):
    model = torch.hub.load("baudm/parseq", model_name, pretrained=True,
                           trust_repo=True)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
    model = model.eval().to(device)
    try:
        from strhub.data.module import SceneTextDataModule
        tf = SceneTextDataModule.get_transform(model.hparams.img_size)
    except Exception:
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((32, 128)), T.ToTensor(), T.Normalize(0.5, 0.5)])
    return model, tf


@torch.no_grad()
def read_batch(model, tf, pil_images, device):
    x = torch.stack([tf(im) for im in pil_images]).to(device)
    logits = model(x)
    labels, confs = model.tokenizer.decode(logits.softmax(-1))
    return [parse_number(lab) for lab in labels]


@torch.no_grad()
def sr_batch(sr, pil_images, device, chunk=4):
    """Real-ESRGAN x4 sobre PILs 224 -> PILs 896."""
    out = []
    for i in range(0, len(pil_images), chunk):
        x = torch.stack([
            torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0)
            for im in pil_images[i:i + chunk]]).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            y = sr(x)
        y = y.clamp(0, 1).mul(255).byte().cpu()
        out.extend(Image.fromarray(t.permute(1, 2, 0).numpy()) for t in y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/jersey_tracking_v2_224")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="parseq")
    ap.add_argument("--checkpoint", default="runs/parseq_jersey_ft/best.pt")
    ap.add_argument("--sr_model", default="models/RealESRGAN_x4plus.pth")
    ap.add_argument("--min_legibility", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="0 = todos")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tf = load_reader(args.model, args.checkpoint, device)

    from spandrel import ModelLoader
    sr = ModelLoader().load_from_file(args.sr_model).model.eval().to(device)

    samples = build_tracking_samples(Path(args.dataset_dir), args.split,
                                     args.min_legibility)
    if args.limit:
        samples = samples[:args.limit]
    print(f"{args.split}: {len(samples)} crops legibles | lector {args.model}"
          f"{' +FT' if args.checkpoint else ''} | SR {Path(args.sr_model).name}")

    hits_a = hits_b = agree = total = 0
    for i in range(0, len(samples), args.batch_size):
        chunk = samples[i:i + args.batch_size]
        pils = [Image.open(p).convert("RGB") for p, _ in chunk]
        gts = [int(lab) for _, lab in chunk]
        preds_a = read_batch(model, tf, pils, device)
        preds_b = read_batch(model, tf, sr_batch(sr, pils, device), device)
        for pa, pb, gt in zip(preds_a, preds_b, gts):
            total += 1
            hits_a += int(pa == gt)
            hits_b += int(pb == gt)
            agree += int(pa == pb)
        if (i // args.batch_size) % 10 == 9:
            print(f"  {total}/{len(samples)}: original {hits_a / total:.4f} "
                  f"| SR {hits_b / total:.4f}", flush=True)

    print("\n=== RESULTADO A/B ===")
    print(f"crops: {total}")
    print(f"A) original: exact-match {hits_a / total:.4f} ({hits_a})")
    print(f"B) SR x4:    exact-match {hits_b / total:.4f} ({hits_b})")
    print(f"acuerdo A==B: {agree / total:.4f}")


if __name__ == "__main__":
    main()
