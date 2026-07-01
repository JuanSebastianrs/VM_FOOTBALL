import os
import sys
import argparse
import subprocess

def run_command(cmd, desc):
    print(f"\n{'='*50}")
    print(f"Running: {desc}")
    print(" ".join(cmd) if isinstance(cmd, list) else cmd)
    print(f"{'='*50}\n")
    if isinstance(cmd, list):
        subprocess.run(cmd, check=True)
    else:
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
    parser.add_argument("--jersey_model", type=str, default=None, help="Path to jersey identification model (enables Phase 10)")
    parser.add_argument("--roster_json", type=str, default=None, help="Optional per-team roster JSON for jersey assignment")
    parser.add_argument("--team_mapping", type=str, default=None, help="team_audit.json or inline '0:right,1:left'")
    parser.add_argument("--p1_threshold", type=float, default=0.85, help="Jersey lock confidence threshold")
    parser.add_argument("--margin_threshold", type=float, default=0.20, help="Jersey margin over second-best threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for jersey model inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic jersey inference")
    parser.add_argument("--run_team_audit", action="store_true", help="Auto-run team clustering audit after Phase 8 (requires gameinfo.ini)")
    parser.add_argument("--inference_mode", type=str, default="temporal", choices=["mil", "temporal", "hybrid"], help="Jersey inference mode: mil=top-K bag, temporal=per-frame fusion, hybrid=both")
    parser.add_argument("--legibility_model", type=str, default=None, help="Path to legibility classifier model")
    parser.add_argument("--legibility_threshold", type=float, default=0.5, help="Legibility threshold for multi-crop variant selection")
    parser.add_argument("--multi_crop", action="store_true", help="Enable multi-crop candidates")
    parser.add_argument("--min_legible_frames", type=int, default=4, help="Minimum legible frames required to filter tracklet")
    parser.add_argument("--min_peak_quality", type=float, default=0.3, help="Minimum quality score for a peak")
    parser.add_argument("--fusion_mode", type=str, default="geometric", choices=["geometric", "arithmetic", "topk_geometric"], help="Jersey temporal fusion strategy")
    parser.add_argument("--temperature", type=float, default=1.0, help="Per-frame probability temperature before jersey fusion")
    parser.add_argument("--link_fragments", action="store_true", help="Jersey: link same-player track fragments and pool evidence")
    parser.add_argument("--split_on_switch", action="store_true", help="Jersey: detect ID switches inside tracklets, never lock across them")
    parser.add_argument("--reassign_conflicts", action="store_true", help="Jersey: conflict losers fall back to best non-conflicting alternative")
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
        [
            sys.executable, os.path.join(base_dir, 'core', 'detection', 'tactical_vision_extractor.py'),
            "--sequence_dir", args.sequence_dir,
            "--yolo_weights", args.yolo_weights,
            "--rfdetr_weights", args.rfdetr_weights,
            "--output_json", det_json
        ],
        "Phase 1: Feature Extraction (YOLOv26 + RF-DETR)"
    )
    
    # Phase 2: CMC
    run_command(
        [
            sys.executable, os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_cmc.py'),
            "--sequence_dir", args.sequence_dir,
            "--output_json", cmc_json,
            "--fast"
        ],
        "Phase 2: Camera Motion Compensation"
    )
    
    # Phase 3-5: Dynamic Graph & Viterbi
    run_command(
        [
            sys.executable, os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_hmm.py'),
            "--detections_json", det_json,
            "--cmc_json", cmc_json,
            "--output_json", traj_json
        ],
        "Phases 3, 4, 5: HMM Graph & Viterbi Decoding"
    )
    
    # Phase 6: Zero-shot segmentation & Rendering
    run_command(
        [
            sys.executable, os.path.join(base_dir, 'core', 'segmentation', 'tactical_vision_sam2.py'),
            "--sequence_dir", args.sequence_dir,
            "--detections_json", det_json,
            "--trajectory_json", traj_json,
            "--sam2_weights", args.sam2_weights,
            "--output_dir", args.output_dir
        ],
        "Phase 6: SAM2 Segmentation and Rendering"
    )
    
    # Phase 7: Evaluation
    run_command(
        [
            sys.executable, os.path.join(base_dir, 'core', 'tracking', 'tactical_vision_eval.py'),
            "--sequence_dir", args.sequence_dir,
            "--trajectory_json", traj_json,
            "--output_plot", plot_path
        ],
        "Phase 7: Metrics Evaluation & Plot Generation"
    )
    
    # Phase 8: Team Clustering
    team_json = os.path.join(args.output_dir, f"{seq_name}_team_assignments.json")
    team_video = os.path.join(args.output_dir, f"{seq_name}_team_clustering.mp4")
    run_command(
        [
            sys.executable, os.path.join(base_dir, 'core', 'clustering', 'team_clustering_phase.py'),
            "--sequence_dir", args.sequence_dir,
            "--rfdetr_weights", args.rfdetr_weights,
            "--detections_json", det_json,
            "--output_json", team_json,
            "--output_video", team_video
        ],
        "Phase 8: Team Clustering (JSON-driven, shared track_ids)"
    )

    # Phase 8.5: Team Clustering Audit
    team_audit_json = os.path.join(args.output_dir, f"{seq_name}_team_audit.json")
    if args.run_team_audit:
        gi_path = os.path.join(args.sequence_dir, "gameinfo.ini")
        if os.path.exists(gi_path):
            run_command(
                [
                    sys.executable, os.path.join(base_dir, 'scripts', 'audit_team_clustering.py'),
                    "--sequence_dir", args.sequence_dir,
                    "--team_json", team_json,
                    "--detections_json", det_json,
                    "--output_json", team_audit_json
                ],
                "Phase 8.5: Team Clustering Audit"
            )
        else:
            print(f"\n[WARNING] --run_team_audit enabled but gameinfo.ini not found at {gi_path}. Skipping audit.")

    # Phase 10: Jersey Number Identification
    if args.jersey_model:
        jersey_json = os.path.join(args.output_dir, f"{seq_name}_jersey_identity.json")
        cmd_jersey = [
            sys.executable, os.path.join(base_dir, 'core', 'identity', 'jersey_identity_phase.py'),
            "--sequence_dir", args.sequence_dir,
            "--detections_json", det_json,
            "--team_assignments_json", team_json,
            "--model_path", args.jersey_model,
            "--output_json", jersey_json,
            "--device", args.device,
            "--p1_threshold", str(args.p1_threshold),
            "--margin_threshold", str(args.margin_threshold),
            "--seed", str(args.seed),
            "--inference_mode", args.inference_mode,
            "--min_legible_frames", str(args.min_legible_frames),
            "--min_peak_quality", str(args.min_peak_quality),
            "--fusion_mode", args.fusion_mode,
            "--temperature", str(args.temperature)
        ]
        if args.roster_json:
            cmd_jersey.extend(["--roster_json", args.roster_json])
        if args.legibility_model:
            cmd_jersey.extend([
                "--legibility_model", args.legibility_model,
                "--legibility_threshold", str(args.legibility_threshold)
            ])
        if args.multi_crop:
            cmd_jersey.append("--multi_crop")
        if args.link_fragments:
            cmd_jersey.append("--link_fragments")
        if args.split_on_switch:
            cmd_jersey.append("--split_on_switch")
        if args.reassign_conflicts:
            cmd_jersey.append("--reassign_conflicts")
            
        # Resolve team mapping: explicit arg > auto-generated audit > nothing
        tm = args.team_mapping
        if not tm and args.run_team_audit and os.path.exists(team_audit_json):
            tm = team_audit_json
        if tm:
            cmd_jersey.extend(["--team_mapping", tm])
            
        run_command(cmd_jersey, "Phase 10: Jersey Number Identification")
    
    # Phase 9: 2D Field Mapping
    mapper_output = os.path.join(args.output_dir, f"{seq_name}_2d_map.mp4")
    cmd_mapper = [
        sys.executable, os.path.join(base_dir, 'core', 'mapping', 'tactical_vision_2d_mapper.py'),
        "--sequence_dir", args.sequence_dir,
        "--detections", det_json,
        "--trajectory", traj_json,
        "--team_assignments", team_json,
        "--pnlcalib_kp_weights", args.pnlcalib_kp_weights,
        "--pnlcalib_line_weights", args.pnlcalib_line_weights,
        "--output", mapper_output
    ]
    if args.jersey_model:
        # Phase 10 ran above; overlay locked/tentative jersey numbers in the final render
        cmd_mapper.extend(["--jersey_json", os.path.join(args.output_dir, f"{seq_name}_jersey_identity.json")])
    run_command(cmd_mapper, "Phase 9: 2D Field Mapping with PnLCalib + Team Colors + Jersey Numbers")
    
    print("\n TacticalVision AI Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()