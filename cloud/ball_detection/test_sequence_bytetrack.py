import cv2
import os
import glob
from ultralytics import YOLO

model_path = "models/yolo_ball_data_centric.pt"
sequence_folder = "datasets/test_seq_116"
output_video_path = "cloud/ball_detection/results/inference_video/test_seq_116_bytetrack.mp4"

def make_bytetrack_sequence():
    print("==================================================")
    print(" BASELINE: YOLO Data-Centric + ByteTrack (Sequence)")
    print("==================================================")
    
    if not os.path.exists(sequence_folder):
        print(f"[ERROR] Sequence folder not found: {sequence_folder}")
        return

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    
    image_files = sorted(glob.glob(os.path.join(sequence_folder, "*.jpg")))
    if not image_files:
        print("[ERROR] No images found in sequence.")
        return

    print(f"[INFO] Found {len(image_files)} frames. Running ByteTrack...")
    
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape
    size = (width, height)
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    
    for i, img_path in enumerate(image_files):
        if i % 50 == 0:
            print(f"Tracking frame {i}/{len(image_files)}...")
            
        frame = cv2.imread(img_path)
        
        # persist=True allows YOLO tracking to remember memory across the loop!
        results = model.track(
            source=frame, 
            conf=0.15,
            iou=0.45,
            imgsz=1280, 
            tracker="bytetrack.yaml", 
            persist=True, 
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    out.release()
    print(f"\n[OK] ByteTrack Sequence Video saved to: {output_video_path}")

if __name__ == "__main__":
    make_bytetrack_sequence()
