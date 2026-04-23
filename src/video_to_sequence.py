import cv2
import os
from pathlib import Path
import argparse

def extract_frames(video_path, output_dir, seq_id="SNMOT-MILLOS", max_seconds=120):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    img_dir = output_dir / "img1"
    os.makedirs(img_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height} @ {fps} FPS")
    print(f"Total frames: {total_frames}")
    
    max_frames = int(max_seconds * fps) if max_seconds > 0 else total_frames
    print(f"Extracting up to {max_frames} frames ({max_seconds} seconds)...")
    
    count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
            
        count += 1
        # Pipeline expects numeric filenames like 000001.jpg
        frame_name = f"{count:06d}.jpg"
        output_path = img_dir / frame_name
        
        # Save frame as JPEG
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        extracted_count += 1
        if extracted_count % 100 == 0:
            print(f"Processed {extracted_count}/{max_frames} frames...")
            
    cap.release()
    print(f"Done! Extracted {extracted_count} frames to {img_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to image sequence.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--id", type=str, default="SNMOT-MILLOS", help="Sequence ID")
    parser.add_argument("--seconds", type=int, default=120, help="Max seconds to extract")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.id, args.seconds)
