import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

def test_video(input_path: str, model_path: str, output_path: str, conf: float = 0.25):
    """
    Test the best YOLO ball detection model on a video or an image sequence folder.
    Reads frames, runs inference, draws bounding boxes, and saves the output mp4.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Determine if input is a video file or a directory of frames
    is_dir = input_path.is_dir()
    
    if is_dir:
        print(f"Opening image sequence folder: {input_path}")
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        # Get all images and sort them alphabetically to maintain sequence
        frames = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
        
        if not frames:
            print(f"Error: No images found in {input_path}")
            return
            
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            print(f"Error reading first frame: {frames[0]}")
            return
            
        height, width = first_frame.shape[:2]
        fps = 30.0  # Default fps for image sequences
        total_frames = len(frames)
    else:
        print(f"Opening video file: {input_path}")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error opening video stream or file: {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing ({total_frames} frames)...")
    frame_count = 0

    if is_dir:
        for frame_file in frames:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
                
            results = model.predict(source=frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames", end='\r')
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames", end='\r')
                
        cap.release()

    print(f"\nFinished! Output saved to: {output_path}")

    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO ball detection on a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--model", type=str, default="cloud/ball_detection/results/exp2_nano_1280/weights/best.pt", help="Path to the trained PyTorch model (best.pt).")
    parser.add_argument("--output", type=str, default="ball_detection_output.mp4", help="Path to save the output processed video.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
    
    args = parser.parse_args()
    
    test_video(args.video, args.model, args.output, args.conf)
