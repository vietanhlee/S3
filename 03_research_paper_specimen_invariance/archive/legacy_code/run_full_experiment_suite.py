"""
run_full_experiment_suite.py
============================
Master CLI Orchestrator to execute all experiments for Paper 3:
"Learning Specimen-Invariant Diagnostic Representations for Timber Forensics"

Supported Execution Modes:
1. 'quick_test'     : Verifies the entire pipeline end-to-end (1 epoch dry run, checks all modules).
2. 'train_proposed' : Trains the Proposed Full Framework (GRL + vCLUB) across 5 folds.
3. 'train_baselines': Trains the key comparative baselines (ERM, Focal, DANN, CDAN, CLUB, Mixup).
4. 'ablation'       : Executes the complete ablation matrix (Masked Softmax, Sampler, Annealing, CLUB).
5. 'robustness'     : Evaluates physical stress resilience (Saw marks, illumination, abrasion, rotation).
6. 'sensitivity'    : Runs the hyperparameter sensitivity grid sweep over (beta, mu).
7. 'collect'        : Scans all outputs and regenerates all publication LaTeX tables.
8. 'all'            : Executes the complete experimental benchmark end-to-end.

Usage:
    # Quick sanity check:
    python run_full_experiment_suite.py --mode quick_test

    # Train proposed framework on fold 0:
    python run_full_experiment_suite.py --mode train_proposed --fold 0 --epochs 40

    # Collect all outputs into LaTeX tables for the paper:
    python run_full_experiment_suite.py --mode collect
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for p in [str(CURRENT_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    parser = argparse.ArgumentParser(description="Master Experiment Suite Runner for Paper 3")
    parser.add_argument(
        "--mode",
        type=str,
        default="collect",
        choices=[
            "quick_test",
            "train_proposed",
            "train_baselines",
            "ablation",
            "robustness",
            "sensitivity",
            "collect",
            "all",
        ],
        help="Experiment execution mode",
    )
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "resnet50", "tf_efficientnetv2_s", "swin_t"],
                        help="Visual backbone")
    parser.add_argument("--fold", type=int, default=0, help="Target fold index [0..4]")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--metadata_csv", type=str, default="01_data_paper_forensic_cites/out/metadata/metadata.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--image_root", type=str, default="01_data_paper_forensic_cites/out",
                        help="Path to image root")
    parser.add_argument("--output_base_dir", type=str, default="specimen_invariance_outputs",
                        help="Output directory")
    return parser.parse_args()


def run_command_stream(cmd_list, desc="Executing command"):
    print("\n" + "=" * 80)
    print(f"[*] {desc}")
    print(f"[*] Command: {' '.join(str(c) for c in cmd_list)}")
    print("=" * 80 + "\n")
    start_t = time.time()
    res = subprocess.run(cmd_list, cwd=str(CURRENT_DIR))
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"[!] Warning: Command failed with exit code {res.returncode} (Elapsed: {elapsed:.1f}s)")
    else:
        print(f"[+] Command completed successfully in {elapsed:.1f}s")
    return res.returncode


def main():
    args = parse_args()
    py_exec = sys.executable

    print("\n" + "#" * 85)
    print("  SPECIMEN-INVARIANT TIMBER FORENSICS: MASTER EXPERIMENTAL SUITE")
    print(f"  Execution Mode: {args.mode.upper()}")
    print(f"  Target Backbone: {args.backbone} | Fold: {args.fold} | Epochs: {args.epochs}")
    print("#" * 85 + "\n")

    # 1. Quick Test Mode
    if args.mode == "quick_test":
        print("[*] Running 1-epoch quick sanity test for Proposed Model...")
        cmd = [
            py_exec, "train.py",
            "--method", "conditional_grl_club",
            "--backbone", args.backbone,
            "--fold", str(args.fold),
            "--epochs", "1",
            "--batch_size", "16",
            "--metadata_csv", args.metadata_csv,
            "--image_root", args.image_root,
            "--output_base_dir", args.output_base_dir,
        ]
        run_command_stream(cmd, "Quick Sanity Test: Proposed Model (1 Epoch)")
        
        # Test collection script
        cmd_collect = [py_exec, "collect_all_benchmark_results.py", "--output_base_dir", args.output_base_dir]
        run_command_stream(cmd_collect, "Testing Results Collector")

    # 2. Train Proposed Framework (GRL + vCLUB)
    elif args.mode == "train_proposed":
        cmd = [
            py_exec, "train.py",
            "--method", "conditional_grl_club",
            "--backbone", args.backbone,
            "--fold", str(args.fold),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--metadata_csv", args.metadata_csv,
            "--image_root", args.image_root,
            "--output_base_dir", args.output_base_dir,
        ]
        run_command_stream(cmd, f"Training Proposed Model (Fold {args.fold}, Epochs {args.epochs})")

    # 3. Train Baselines
    elif args.mode == "train_baselines":
        baselines = ["focal", "ce", "mixup", "dann_unconditional", "cdan", "club"]
        for b in baselines:
            cmd = [
                py_exec, "train.py",
                "--method", b,
                "--backbone", args.backbone,
                "--fold", str(args.fold),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--metadata_csv", args.metadata_csv,
                "--image_root", args.image_root,
                "--output_base_dir", args.output_base_dir,
            ]
            run_command_stream(cmd, f"Training Baseline: {b} (Fold {args.fold})")

    # 4. Ablation Study
    elif args.mode == "ablation":
        cmd = [
            py_exec, "run_ablation_study.py",
            "--backbone", args.backbone,
            "--fold", str(args.fold),
            "--epochs", str(args.epochs),
            "--metadata_csv", args.metadata_csv,
            "--image_root", args.image_root,
            "--output_base_dir", args.output_base_dir,
        ]
        run_command_stream(cmd, "Executing Complete Ablation Matrix")

    # 5. Robustness Stress Testing
    elif args.mode == "robustness":
        cmd = [
            py_exec, "run_robustness_study.py",
            "--output_base_dir", args.output_base_dir,
        ]
        run_command_stream(cmd, "Executing Physical Robustness Stress Testing")

    # 6. Sensitivity Sweep
    elif args.mode == "sensitivity":
        cmd = [
            py_exec, "run_pareto_sweep.py",
            "--backbone", args.backbone,
            "--fold", str(args.fold),
            "--epochs", str(min(15, args.epochs)),
            "--metadata_csv", args.metadata_csv,
            "--image_root", args.image_root,
            "--output_base_dir", args.output_base_dir,
        ]
        run_command_stream(cmd, "Executing Hyperparameter Sensitivity Sweep")

    # 7. Collect All Results & Generate LaTeX Tables
    elif args.mode == "collect":
        cmd = [
            py_exec, "collect_all_benchmark_results.py",
            "--output_base_dir", args.output_base_dir,
        ]
        run_command_stream(cmd, "Generating All Publication LaTeX Tables")

    # 8. All Modes Sequentially
    elif args.mode == "all":
        print("[*] Launching Full End-to-End Experimental Suite...")
        # 1. Proposed
        run_command_stream([py_exec, "train.py", "--method", "conditional_grl_club", "--fold", str(args.fold), "--epochs", str(args.epochs)], "1. Proposed Training")
        # 2. Baselines
        run_command_stream([py_exec, "run_parallel_dispatcher.py", "--mode", "baselines", "--fold", str(args.fold), "--epochs", str(args.epochs)], "2. Baselines Dispatcher")
        # 3. Ablations
        run_command_stream([py_exec, "run_ablation_study.py", "--fold", str(args.fold), "--epochs", str(args.epochs)], "3. Ablation Study")
        # 4. Robustness
        run_command_stream([py_exec, "run_robustness_study.py"], "4. Robustness Study")
        # 5. Collect
        run_command_stream([py_exec, "collect_all_benchmark_results.py"], "5. Final LaTeX Table Compilation")

    print("\n" + "=" * 85)
    print("[SUCCESS] Experiment Suite Execution Finished!")
    print(f"Check outputs in: {Path(args.output_base_dir).resolve()}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
