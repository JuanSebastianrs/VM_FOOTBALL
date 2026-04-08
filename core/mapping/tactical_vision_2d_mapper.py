import cv2
import numpy as np
import json
import os
import glob
import argparse
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from vendor.pnlcalib.model import cls_hrnet, cls_hrnet_l
from vendor.pnlcalib.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    coords_to_dict,
    complete_keypoints
)
from vendor.pnlcalib.utils.utils_calib import FramebyFrameCalib


# ---------------------------------------------------------------------------
# Projection helpers (match original PnLCalib inference.py exactly)
# ---------------------------------------------------------------------------

def build_P_from_cam_params(cam_params):
    """
    Build the 3x4 projection matrix P from a PnLCalib cam_params dict.
    This is identical to `projection_from_cam_params` in the original repo.
    """
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)
    return P


def ground_homography_from_P(P):
    """
    Extract the 3x3 ground-plane (Z=0) homography from a 3x4 projection matrix.
    For Z=0, the third column of P is irrelevant, so H = [P[:,0] | P[:,1] | P[:,3]].
    Returns H_inv (image -> world).
    """
    H = P[:, [0, 1, 3]]
    H_inv = np.linalg.inv(H)
    return H_inv


# ---------------------------------------------------------------------------
# Homography smoother — smooth the final H_inv matrix directly with EMA.
# This avoids any Euler angle interpolation issues.
# ---------------------------------------------------------------------------

class HomographySmoother:
    """
    Exponential Moving Average smoother for 3x3 homography matrices.
    Smooths H_inv directly so projection is stable frame-to-frame.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.H_inv = None

    def update(self, H_inv_new):
        if self.H_inv is None:
            self.H_inv = H_inv_new.copy()
        else:
            self.H_inv = self.alpha * H_inv_new + (1.0 - self.alpha) * self.H_inv
        return self.H_inv


# ---------------------------------------------------------------------------
# Minimap drawing
# ---------------------------------------------------------------------------

def draw_pitch_cv2(scale=10, margin=50):
    """Draw a 2D pitch minimap using OpenCV."""
    w = int(105 * scale)
    h = int(68 * scale)
    img_w, img_h = w + 2 * margin, h + 2 * margin

    pitch = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    pitch[:] = (60, 120, 50)  # BGR green

    white = (255, 255, 255)
    thickness = 2

    pt = lambda x, y: (int(x * scale) + margin, int(y * scale) + margin)

    # Field outline
    cv2.rectangle(pitch, pt(0, 0), pt(105, 68), white, thickness)
    # Half-way line
    cv2.line(pitch, pt(52.5, 0), pt(52.5, 68), white, thickness)
    # Centre circle
    cv2.circle(pitch, pt(52.5, 34), int(9.15 * scale), white, thickness)
    cv2.circle(pitch, pt(52.5, 34), 2, white, -1)
    # Goal areas
    cv2.rectangle(pitch, pt(0, 24.85), pt(5.5, 43.15), white, thickness)
    cv2.rectangle(pitch, pt(105 - 5.5, 24.85), pt(105, 43.15), white, thickness)
    # Penalty areas
    cv2.rectangle(pitch, pt(0, 13.85), pt(16.5, 54.15), white, thickness)
    cv2.rectangle(pitch, pt(105 - 16.5, 13.85), pt(105, 54.15), white, thickness)
    # Penalty spots
    cv2.circle(pitch, pt(11, 34), 2, white, -1)
    cv2.circle(pitch, pt(105 - 11, 34), 2, white, -1)
    # Penalty arcs
    cv2.ellipse(pitch, pt(11, 34), (int(9.15 * scale), int(9.15 * scale)),
                0, -53.13, 53.13, white, thickness)
    cv2.ellipse(pitch, pt(105 - 11, 34), (int(9.15 * scale), int(9.15 * scale)),
                0, 126.87, 233.13, white, thickness)

    return pitch


# ---------------------------------------------------------------------------
# Projection: image pixel  ->  minimap pixel   via  H_inv
# ---------------------------------------------------------------------------

def project_point(x_img, y_img, H_inv, scale, margin, w_pitch, h_pitch):
    """
    Project an image-space point (x_img, y_img) onto the minimap using H_inv.
    H_inv maps image coords -> PnLCalib world coords (centred on pitch centre).
    Returns (mx, my) minimap pixel coords, or (None, None) if out of bounds.
    """
    pt_world = H_inv @ np.array([x_img, y_img, 1.0])
    pt_world /= pt_world[2]

    # PnLCalib world has origin at pitch centre -> shift to top-left origin
    x_world = pt_world[0] + 52.5
    y_world = pt_world[1] + 34.0

    # Loose sanity bounds
    if x_world < -5 or x_world > 110 or y_world < -5 or y_world > 73:
        return None, None

    mx = int(x_world * scale) + margin
    my = int(y_world * scale) + margin

    if 0 <= mx < w_pitch and 0 <= my < h_pitch:
        return mx, my
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--pnlcalib_kp_weights", type=str, default="models/SV_kp")
    parser.add_argument("--pnlcalib_line_weights", type=str, default="models/SV_lines")
    parser.add_argument("--pnlcalib_kp_cfg", type=str,
                        default="vendor/pnlcalib/config/hrnetv2_w48.yaml")
    parser.add_argument("--pnlcalib_l_cfg", type=str,
                        default="vendor/pnlcalib/config/hrnetv2_w48_l.yaml")
    parser.add_argument("--kp_threshold", type=float, default=0.3434)
    parser.add_argument("--line_threshold", type=float, default=0.7867)
    parser.add_argument("--disable_pnl_refine", action="store_true")
    parser.add_argument("--smooth_alpha", type=float, default=0.3,
                        help="EMA alpha for H_inv smoothing (0=frozen, 1=no smooth)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # --- Load HRNet configs ---
    with open(args.pnlcalib_kp_cfg, 'r') as fh:
        cfg = yaml.safe_load(fh)
    with open(args.pnlcalib_l_cfg, 'r') as fh:
        cfgl = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # --- Load HRNet models (exactly as original PnLCalib inference.py) ---
    print("Cargando modelo PnLCalib Keypoints HRNet...")
    model_kp = cls_hrnet.get_cls_net(cfg)
    model_kp.load_state_dict(
        torch.load(args.pnlcalib_kp_weights, map_location=device, weights_only=False))
    model_kp.to(device).eval()

    print("Cargando modelo PnLCalib Lines HRNet...")
    model_l = cls_hrnet_l.get_cls_net(cfgl)
    model_l.load_state_dict(
        torch.load(args.pnlcalib_line_weights, map_location=device, weights_only=False))
    model_l.to(device).eval()

    transform_resize = T.Resize((540, 960))

    # --- Load tracking data ---
    print("Cargando datos de tracking...")
    with open(args.detections, "r") as fh:
        detections = {d["frame_id"]: d for d in json.load(fh)}
    with open(args.trajectory, "r") as fh:
        trajectory = {t["frame_id"]: t for t in json.load(fh)}

    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        print("Error: No hay imágenes en la ruta de la secuencia.")
        return

    frame_0 = cv2.imread(images[0])
    h_ori, w_ori = frame_0.shape[:2]

    # --- Video layout ---
    scale = 8
    margin = 40
    base_pitch = draw_pitch_cv2(scale, margin)
    h_pitch, w_pitch = base_pitch.shape[:2]

    target_video_h = max(h_ori, h_pitch)
    scale_ori = target_video_h / h_ori
    new_w_ori = int(w_ori * scale_ori)

    out_w = new_w_ori + w_pitch
    out_h = target_video_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, 25.0, (out_w, out_h))

    # --- Calibration objects ---
    cam = FramebyFrameCalib(iwidth=w_ori, iheight=h_ori, denormalize=True)
    smoother = HomographySmoother(alpha=args.smooth_alpha)

    last_valid_H_inv = None
    pnl_refine = not args.disable_pnl_refine

    calib_ok = 0
    calib_fail = 0

    print(f"Procesando {len(images)} frames...")

    for img_path in tqdm(images):
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
        frame = cv2.imread(img_path)

        # ---- PnLCalib inference (mirrors original inference.py exactly) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = TF.to_tensor(pil).float().unsqueeze(0)
        if tensor.size()[-1] != 960:
            tensor = transform_resize(tensor)
        tensor = tensor.to(device)
        _, _, h_t, w_t = tensor.size()

        with torch.no_grad():
            hm_kp = model_kp(tensor)
            hm_l = model_l(tensor)

        # Exclude background channel (last channel)
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :])

        kp_dict = coords_to_dict(kp_coords, threshold=args.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=args.line_threshold)

        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0], lines_dict[0], w=w_t, h=h_t, normalize=True
        )

        cam.update(kp_dict, lines_dict)
        result = cam.heuristic_voting(refine_lines=pnl_refine)

        # ---- Build H_inv from raw PnLCalib cam_params (no angle reconstruction) ----
        if result is not None:
            cam_params = result['cam_params']
            P = build_P_from_cam_params(cam_params)
            H_inv_raw = ground_homography_from_P(P)
            H_inv = smoother.update(H_inv_raw)
            last_valid_H_inv = H_inv
            calib_ok += 1
        else:
            H_inv = last_valid_H_inv
            calib_fail += 1

        # ---- Render minimap ----
        pitch_frame = base_pitch.copy()

        if H_inv is not None:
            # -- Players --
            frame_data = detections.get(frame_id, {"players": []})
            for p in frame_data.get("players", []):
                x_min = p["x_min"]
                y_min = p["y_min"]
                x_max = p["x_max"]
                y_max = p["y_max"]
                track_id = p.get("track_id", -1)

                # Draw bbox on video frame
                cv2.rectangle(frame,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              (255, 0, 0), 2)

                # Project foot-point (bottom-centre of bbox)
                x_foot = (x_min + x_max) / 2.0
                y_foot = y_max

                mx, my = project_point(
                    x_foot, y_foot, H_inv, scale, margin, w_pitch, h_pitch
                )
                if mx is not None:
                    cv2.circle(pitch_frame, (mx, my), 6, (255, 0, 0), -1)
                    cv2.putText(pitch_frame, str(track_id), (mx + 8, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)

            # -- Ball (Viterbi trajectory) --
            ball = trajectory.get(frame_id)
            if ball is not None:
                bx, by = ball["x"], ball["y"]
                is_dummy = ball.get("is_dummy", False)
                color = (0, 0, 255) if is_dummy else (0, 255, 255)
                cv2.circle(frame, (int(bx), int(by)), 5, color, -1)

                mx, my = project_point(
                    bx, by, H_inv, scale, margin, w_pitch, h_pitch
                )
                if mx is not None:
                    cv2.circle(pitch_frame, (mx, my), 5, (0, 165, 255), -1)

        # ---- Compose side-by-side ----
        frame_resized = cv2.resize(frame, (new_w_ori, target_video_h))
        pad_top = (target_video_h - h_pitch) // 2
        pad_bot = target_video_h - h_pitch - pad_top
        pitch_padded = cv2.copyMakeBorder(
            pitch_frame, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=[0, 0, 0])

        out_video.write(np.hstack((frame_resized, pitch_padded)))

    out_video.release()
    print(f"\nCalibración exitosa: {calib_ok}/{len(images)} frames")
    print(f"Calibración fallida: {calib_fail}/{len(images)} frames")
    print(f"Video guardado en: {args.output}")


if __name__ == '__main__':
    main()
