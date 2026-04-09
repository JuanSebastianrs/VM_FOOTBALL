import os
import cv2
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Phase 2: Camera Motion Compensation (CMC)")
    parser.add_argument("--sequence_dir", type=str, required=True,
                        help="Path to sequence directory containing 'img1' folder")
    parser.add_argument("--output_json", type=str, default="secuencia_197_cmc.json",
                        help="Path to save the affine CMC matrices")
    parser.add_argument("--fast", action="store_true", help="Use ORB instead of ECC for speed")
    return parser.parse_args()

def extract_orb_matrix(img_t, img_t1):
    """Fallback orb-based aligner if ECC fails."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img_t, None)
    kp2, des2 = orb.detectAndCompute(img_t1, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.eye(2, 3, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # keep top matches
    matches = matches[:min(len(matches), 500)]
    
    if len(matches) < 4:
        return np.eye(2, 3, dtype=np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate Affine transform (rigid + scale/shear but minimal)
    # Using RANSAC to be robust
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    return M

def compute_cmc():
    args = get_args()
    
    img_dir = os.path.join(args.sequence_dir, "img1")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if not images:
        print(f"Error: No images found in {img_dir}")
        return

    print(f"Found {len(images)} frames. Computing CMC (Fast Mode: {args.fast})...")
    
    cmc_matrices = {}
    
    # Define ECC criteria
    number_of_iterations = 50
    termination_eps = 1e-3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    warp_mode = cv2.MOTION_AFFINE

    prev_img = None
    prev_frame_id = -1
    
    for i in tqdm(range(len(images))):
        img_path = images[i]
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0])
        curr_color = cv2.imread(img_path)
        curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
        
        if prev_img is None:
            prev_img = curr_gray
            prev_frame_id = frame_id
            continue
            
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        if args.fast:
            warp_matrix = extract_orb_matrix(prev_img, curr_gray)
        else:
            try:
                # ECC alignment
                _, warp_matrix = cv2.findTransformECC(prev_img, curr_gray, warp_matrix, warp_mode, criteria, None, 1)
            except Exception:
                # Fallback to feature matching ORB
                warp_matrix = extract_orb_matrix(prev_img, curr_gray)
            
        # Store transition: mapping from prev_frame_id to frame_id
        # We store it in a dict with string key "t->t+1"
        key = f"{prev_frame_id}->{frame_id}"
        cmc_matrices[key] = warp_matrix.tolist()
        
        prev_img = curr_gray
        prev_frame_id = frame_id
        
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_json, 'w') as f:
        json.dump(cmc_matrices, f, indent=4)
        
    print(f"CMC computation complete. Exported to {args.output_json}")

if __name__ == "__main__":
    compute_cmc()
