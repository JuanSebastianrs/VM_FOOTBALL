import os
import argparse
import subprocess

def run_command(cmd, desc):
    print(f"\n{'='*50}")
    print(f"🚀 Running: {desc}")
    print(cmd)
    print(f"{'='*50}\n")
    subprocess.run(cmd, shell=True, check=True)

def get_args():
    parser = argparse.ArgumentParser(description="TacticalVision AI: End-to-End Offline Tracking Pipeline")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to sequence directory 'img1'")
    parser.add_argument("--yolo_weights", type=str, required=True, help="Path to YOLO ball detection weights")
    parser.add_argument("--rfdetr_weights", type=str, default="models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth", help="Path to RF-DETR weights for player detection")
    parser.add_argument("--sam2_weights", type=str, required=True, help="Path to SAM2 weights")
    parser.add_argument("--output_dir", type=str, default="tactical_results", help="Directory for output files")
    parser.add_argument("--output_plot", type=str, default="", help="Optional: Path to save Tracking Metrics Graphic (.png)")
    parser.add_argument("--pnlcalib_kp_weights", type=str, default="models/SV_kp", help="Path to PnLCalib keypoints weights")
    parser.add_argument("--pnlcalib_line_weights", type=str, default="models/SV_lines", help="Path to PnLCalib line weights")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    seq_name = os.path.basename(os.path.normpath(args.sequence_dir))
    
    # Files
    det_json = os.path.join(args.output_dir, f"{seq_name}_detections.json")
    cmc_json = os.path.join(args.output_dir, f"{seq_name}_cmc.json")
    traj_json = os.path.join(args.output_dir, f"{seq_name}_trajectory.json")
    
    # Decide plot path
    plot_path = args.output_plot if args.output_plot else os.path.join(args.output_dir, f"{seq_name}_tracking_metrics.png")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Phase 1: Feature Extraction
    run_command(
        f"python {os.path.join(base_dir, 'core', 'detection', 'tactical_vision_extractor.py')} --sequence_dir {args.sequence_dir} --yolo_weights {args.yolo_weights} --rfdetr_weights {args.rfdetr_weights} --output_json {det_json}",
        "Phase 1: Feature Extraction (YOLOv26 + RF-DETR)"
    )
    
    # Phase 2: CMC
    run_command(
        f"python {os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_cmc.py')} --sequence_dir {args.sequence_dir} --output_json {cmc_json} --fast",
        "Phase 2: Camera Motion Compensation"
    )
    
    # Phase 3-5: Dynamic Graph & Viterbi
    run_command(
        f"python {os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_hmm.py')} --detections_json {det_json} --cmc_json {cmc_json} --output_json {traj_json}",
        "Phases 3, 4, 5: HMM Graph & Viterbi Decoding"
    )
    
    # Phase 6: Zero-shot segmentation & Rendering
    run_command(
        f"python {os.path.join(base_dir, 'core', 'segmentation', 'tactical_vision_sam2.py')} --sequence_dir {args.sequence_dir} --detections_json {det_json} --trajectory_json {traj_json} --sam2_weights {args.sam2_weights} --output_dir {args.output_dir}",
        "Phase 6: SAM2 Segmentation and Rendering"
    )
    
    # Phase 7: Evaluation
    run_command(
        f"python {os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_eval.py')} --sequence_dir {args.sequence_dir} --trajectory_json {traj_json} --output_plot {plot_path}",
        "Phase 7: Metrics Evaluation & Plot Generation"
    )
    
    # Phase 8: Team Clustering (must run BEFORE 2D mapping so teams are available)
    team_json = os.path.join(args.output_dir, f"{seq_name}_team_assignments.json")
    team_video = os.path.join(args.output_dir, f"{seq_name}_team_clustering.mp4")
    run_command(
        f"python {os.path.join(base_dir, 'core', 'clustering', 'team_clustering_phase.py')} --sequence_dir {args.sequence_dir} --rfdetr_weights {args.rfdetr_weights} --detections_json {det_json} --output_json {team_json} --output_video {team_video}",
        "Phase 8: Team Clustering (JSON-driven, shared track_ids)"
    )
    
    # Phase 9: 2D Field Mapping (consumes team clustering for team-colored minimap)
    mapper_output = os.path.join(args.output_dir, f"{seq_name}_2d_map.mp4")
    run_command(
        f"python {os.path.join(base_dir, 'core', 'mapping', 'tactical_vision_2d_mapper.py')} --sequence_dir {args.sequence_dir} --detections {det_json} --trajectory {traj_json} --team_assignments {team_json} --pnlcalib_kp_weights {args.pnlcalib_kp_weights} --pnlcalib_line_weights {args.pnlcalib_line_weights} --output {mapper_output}",
        "Phase 9: 2D Field Mapping with PnLCalib + Team Colors"
    )
    
    print("\n TacticalVision AI Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
