"""
specimen_invariance_framework/run_parallel_dispatcher.py
========================================================
Automatic Multi-GPU Training Runner & Dispatcher.

Supported Strategies:
1. 'data_parallel' (DEFAULT & RECOMMENDED):
   - Trains models sequentially.
   - For EACH model, harnesses ALL detected GPUs simultaneously using torch.nn.DataParallel.
   - Ideal for maximizing throughput on Kaggle dual GPUs (e.g. 2x T4) while streaming live training progress.
   
2. 'task_parallel':
   - Trains different models concurrently in parallel.
   - Pin 1 model to GPU 0, 1 model to GPU 1, etc.
   - Best when memory allows and running multiple quick experiments at once.

Usage:
    # Run core 5 baselines using ALL GPUs concurrently per model (DataParallel):
    python run_parallel_dispatcher.py --mode baselines --fold 0 --epochs 13

    # Run backbone ablations using ALL GPUs concurrently per model:
    python run_parallel_dispatcher.py --mode backbones --fold 0 --epochs 13

    # Run in task-parallel mode (1 method per GPU concurrently):
    python run_parallel_dispatcher.py --mode baselines --strategy task_parallel
"""

import argparse
import os
import sys

# Suppress multiple OpenMP runtime initialization errors and Hugging Face Hub unauthenticated warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

import time
import subprocess
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from typing import List, Dict, Any, Optional

import torch


# Core Minimal Benchmark (Option A: 5 representative paradigms)
BASELINES = [
    "focal",                # Baseline: Standard empirical risk minimization (Focal loss)
    "mixup",                # Baseline: General data augmentation / regularization
    "dann_unconditional",   # Baseline: Unconditional domain adaptation (collapses on singleton taxa)
    "club",                 # Baseline: Variational mutual information bottleneck
    "conditional_grl",      # Proposed: Species-Conditioned Masked Softmax GRL
]

BACKBONES = [
    "convnext_tiny",
    "resnet50",
    "tf_efficientnetv2_s",
    "swin_t",
]


print_lock = Lock()


def safe_print(msg: str):
    with print_lock:
        print(msg, flush=True)


def run_data_parallel_sequential(
    tasks: List[Dict[str, Any]],
    script_args: argparse.Namespace,
    log_dir: Path,
    gpu_ids: List[Optional[int]],
) -> List[Dict[str, Any]]:
    """
    Executes tasks sequentially. For each task, PyTorch uses ALL visible GPUs
    concurrently via torch.nn.DataParallel. Real-time stdout is streamed to console
    and saved to the log file.
    """
    current_script_dir = Path(__file__).resolve().parent
    train_script = current_script_dir / "train.py"
    results_list: List[Dict[str, Any]] = []
    total_tasks = len(tasks)

    num_gpus = len([g for g in gpu_ids if g is not None])
    device_desc = f"{num_gpus} GPUs (DataParallel)" if num_gpus > 1 else (f"GPU {gpu_ids[0]}" if num_gpus == 1 else "CPU")

    safe_print("=" * 90)
    safe_print(f"[*] DATA-PARALLEL ENGINE ACTIVATED: Concurrently training each model on {device_desc}")
    safe_print(f"[*] Total queued tasks: {total_tasks} | Epochs per run: {script_args.epochs}")
    safe_print("=" * 90 + "\n")

    for idx, task in enumerate(tasks, start=1):
        method = task["method"]
        backbone = task.get("backbone", script_args.backbone)
        fold = task.get("fold", script_args.fold)
        seed = task.get("seed", script_args.seed)

        task_name = f"{method}_{backbone}_fold{fold}_seed{seed}"
        log_file = log_dir / f"{task_name}.log"
        run_output_dir = Path(script_args.output_base_dir) / task_name
        summary_file = run_output_dir / "train_summary.json"
        best_ckpt = run_output_dir / "best_model.pth"

        # Check for completed runs to avoid wasting hours of GPU time
        if summary_file.exists() and best_ckpt.exists() and not getattr(script_args, "force", False):
            safe_print("-" * 90)
            safe_print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> SKIPPING TASK [{idx}/{total_tasks}]: {method} ({backbone})")
            safe_print(f"         Status: Already completed and verified at {summary_file}")
            safe_print(f"         (To force re-training, use --force or remove {run_output_dir})")
            safe_print("-" * 90)
            results_list.append({
                "task_name": task_name,
                "method": method,
                "backbone": backbone,
                "gpu": device_desc,
                "duration_min": 0.0,
                "status": "SUCCESS (Cached)",
                "log_file": str(log_file),
            })
            continue

        start_time = time.time()
        start_dt = datetime.now().strftime("%H:%M:%S")

        safe_print("-" * 90)
        safe_print(f"[{start_dt}] >>> STARTING TASK [{idx}/{total_tasks}]: {method} ({backbone})")
        safe_print(f"         Strategy: Multi-GPU DataParallel across {device_desc}")
        safe_print(f"         Log File: {log_file}")
        safe_print("-" * 90)

        cmd = [
            sys.executable,
            "-u",
            str(train_script),
            "--method", method,
            "--backbone", backbone,
            "--fold", str(fold),
            "--epochs", str(script_args.epochs),
            "--batch_size", str(script_args.batch_size),
            "--seed", str(seed),
            "--metadata_csv", script_args.metadata_csv,
            "--image_root", script_args.image_root,
            "--output_base_dir", script_args.output_base_dir,
        ]

        env = os.environ.copy()
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["HF_HUB_VERBOSITY"] = "error"
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        if script_args.gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in script_args.gpus)

        status = "SUCCESS"
        error_msg = ""

        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(f"=== Execution Command: {' '.join(cmd)} ===\n")
                lf.write(f"=== Started: {datetime.now().isoformat()} on {device_desc} ===\n\n")
                lf.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(current_script_dir),
                    text=True,
                    bufsize=1,
                )

                # Stream stdout live to console and log file
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()

                process.wait()

                if process.returncode != 0:
                    status = f"FAILED (code {process.returncode})"
                    error_msg = f"Check log: {log_file}"

        except Exception as e:
            status = "ERROR"
            error_msg = str(e)

        elapsed_sec = time.time() - start_time
        elapsed_min = elapsed_sec / 60.0
        end_dt = datetime.now().strftime("%H:%M:%S")

        status_tag = f"[✓ {status}]" if "SUCCESS" in status else f"[✗ {status}]"
        safe_print(f"\n[{end_dt}] {status_tag} COMPLETED: {method} in {elapsed_min:.1f} mins. {error_msg}\n")

        results_list.append({
            "task_name": task_name,
            "method": method,
            "backbone": backbone,
            "gpu": device_desc,
            "duration_min": round(elapsed_min, 2),
            "status": status,
            "log_file": str(log_file),
        })

    return results_list


def worker_loop(
    gpu_id: Optional[int],
    task_queue: Queue,
    results_list: List[Dict[str, Any]],
    script_args: argparse.Namespace,
    log_dir: Path,
):
    """Worker thread that executes queued tasks on a pinned single GPU (Task-Parallel)."""
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        method = task.get("method")
        backbone = task.get("backbone", script_args.backbone)
        fold = task.get("fold", script_args.fold)
        seed = task.get("seed", script_args.seed)

        gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        task_name = f"{method}_{backbone}_fold{fold}_seed{seed}"
        log_file = log_dir / f"{task_name}.log"
        run_output_dir = Path(script_args.output_base_dir) / task_name
        summary_file = run_output_dir / "train_summary.json"
        best_ckpt = run_output_dir / "best_model.pth"

        if summary_file.exists() and best_ckpt.exists() and not getattr(script_args, "force", False):
            safe_print(f"[{datetime.now().strftime('%H:%M:%S')}] [{gpu_str}] >>> SKIPPING (Already completed): {method} ({backbone})")
            results_list.append({
                "task_name": task_name,
                "method": method,
                "backbone": backbone,
                "gpu": gpu_str,
                "duration_min": 0.0,
                "status": "SUCCESS (Cached)",
                "log_file": str(log_file),
            })
            task_queue.task_done()
            continue

        start_time = time.time()
        start_dt = datetime.now().strftime("%H:%M:%S")
        safe_print(f"[{start_dt}] [{gpu_str}] >>> LAUNCHING: {method} ({backbone}) -> Log: {log_file.name}")

        current_script_dir = Path(__file__).resolve().parent
        train_script = current_script_dir / "train.py"

        cmd = [
            sys.executable,
            "-u",
            str(train_script),
            "--method", method,
            "--backbone", backbone,
            "--fold", str(fold),
            "--epochs", str(script_args.epochs),
            "--batch_size", str(script_args.batch_size),
            "--seed", str(seed),
            "--metadata_csv", script_args.metadata_csv,
            "--image_root", script_args.image_root,
            "--output_base_dir", script_args.output_base_dir,
        ]
        if gpu_id is not None:
            cmd.extend(["--gpu", "0"])

        env = os.environ.copy()
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["HF_HUB_VERBOSITY"] = "error"
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        status = "SUCCESS"
        error_msg = ""

        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(f"=== Execution Command: {' '.join(cmd)} ===\n")
                lf.write(f"=== Started: {datetime.now().isoformat()} on {gpu_str} ===\n\n")
                lf.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(current_script_dir),
                    text=True,
                    bufsize=1,
                )

                for line in iter(process.stdout.readline, ''):
                    safe_print(f"[{gpu_str}][{method}] " + line.rstrip())
                    lf.write(line)
                    lf.flush()

                process.wait()

                if process.returncode != 0:
                    status = f"FAILED (code {process.returncode})"
                    tail_str = ""
                    try:
                        if log_file.exists():
                            with open(log_file, "r", encoding="utf-8", errors="ignore") as rf:
                                all_lines = [l.rstrip() for l in rf.readlines() if l.strip()]
                                tail_lines = all_lines[-12:]
                                if tail_lines:
                                    tail_str = "\n" + "\n".join(f"         [Trace] {line}" for line in tail_lines)
                    except Exception:
                        pass
                    error_msg = f"Check log: {log_file}{tail_str}"

        except Exception as e:
            status = "ERROR"
            error_msg = str(e)

        elapsed_sec = time.time() - start_time
        elapsed_min = elapsed_sec / 60.0
        end_dt = datetime.now().strftime("%H:%M:%S")

        status_tag = f"[✓ {status}]" if "SUCCESS" in status else f"[✗ {status}]"
        safe_print(
            f"[{end_dt}] [{gpu_str}] {status_tag} COMPLETED: {method} in {elapsed_min:.1f} mins. {error_msg}"
        )

        results_list.append({
            "task_name": task_name,
            "method": method,
            "backbone": backbone,
            "gpu": gpu_str,
            "duration_min": round(elapsed_min, 2),
            "status": status,
            "log_file": str(log_file),
        })

        task_queue.task_done()


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Training Dispatcher & Runner")
    parser.add_argument("--strategy", type=str, default="data_parallel", choices=["data_parallel", "task_parallel"],
                        help="Execution strategy: 'data_parallel' (default) uses all visible GPUs simultaneously for each model; 'task_parallel' trains different models concurrently (1 model per GPU).")
    parser.add_argument("--mode", type=str, default="baselines", choices=["baselines", "backbones", "custom"],
                        help="Mode: 'baselines' runs 5 core methods; 'backbones' runs 4 architectures; 'custom' uses --methods")
    parser.add_argument("--methods", nargs="+", type=str, default=None,
                        help="Custom list of methods to run (when --mode custom)")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Default backbone")
    parser.add_argument("--fold", type=int, default=0, help="Round-robin fold index")
    parser.add_argument("--epochs", type=int, default=13, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="Explicit list of GPU IDs to use (e.g. 0 1). If None, auto-detects all available GPUs.")
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--image_root", type=str, default="out", help="Image directory root")
    parser.add_argument("--output_base_dir", type=str, default="specimen_invariance_outputs", help="Output base directory")
    parser.add_argument("--force", action="store_true", help="Force re-training even if train_summary.json exists")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.output_base_dir) / "dispatcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Automatic GPU Detection
    if args.gpus is not None:
        gpu_ids = args.gpus
        print(f"[+] User-specified GPU pool: {gpu_ids}")
    else:
        detected_count = torch.cuda.device_count()
        if detected_count > 0:
            gpu_ids = list(range(detected_count))
            print(f"[+] Automatically detected {detected_count} GPU(s):")
            for i in gpu_ids:
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            gpu_ids = [None]  # Fallback to CPU execution
            print("[!] No CUDA GPUs detected. Falling back to CPU execution.")

    # 2. Build Task List
    tasks = []
    if args.mode == "baselines":
        for m in BASELINES:
            tasks.append({"method": m, "backbone": args.backbone, "fold": args.fold, "seed": args.seed})
    elif args.mode == "backbones":
        for b in BACKBONES:
            tasks.append({"method": "conditional_grl", "backbone": b, "fold": args.fold, "seed": args.seed})
    elif args.mode == "custom":
        if not args.methods:
            raise ValueError("When using --mode custom, please specify --methods <m1> <m2> ...")
        for m in args.methods:
            tasks.append({"method": m, "backbone": args.backbone, "fold": args.fold, "seed": args.seed})

    t_start = time.time()

    # 3. Execute according to Strategy
    if args.strategy == "data_parallel":
        results_list = run_data_parallel_sequential(tasks, args, log_dir, gpu_ids)
    else:
        # Task-parallel concurrent mode
        num_workers = len(gpu_ids)
        print(f"\n[+] Task-Parallel Mode: {len(tasks)} tasks | Worker concurrency: {num_workers} parallel workers\n")
        task_queue = Queue()
        for t in tasks:
            task_queue.put(t)

        results_list: List[Dict[str, Any]] = []
        threads = []

        for worker_idx in range(num_workers):
            assigned_gpu = gpu_ids[worker_idx]
            t = Thread(
                target=worker_loop,
                args=(assigned_gpu, task_queue, results_list, args, log_dir),
                daemon=True,
            )
            t.start()
            threads.append(t)

        task_queue.join()
        for _ in range(num_workers):
            task_queue.put(None)
        for t in threads:
            t.join()

    total_duration_min = (time.time() - t_start) / 60.0

    # 4. Executive Summary Report
    print("\n" + "=" * 90)
    print("                    MULTI-GPU TRAINING DISPATCHER SUMMARY REPORT                    ")
    print("=" * 90)
    print(f"{'Method':<20} | {'Backbone':<18} | {'Device':<24} | {'Duration (m)':<12} | {'Status':<15}")
    print("-" * 90)
    for r in results_list:
        print(f"{r['method']:<20} | {r['backbone']:<18} | {r['gpu']:<24} | {r['duration_min']:<12.2f} | {r['status']:<15}")
    print("=" * 90)
    print(f"[+] Total execution time: {total_duration_min:.2f} minutes.")
    print(f"[+] Model checkpoints and summaries saved in: '{args.output_base_dir}'")


if __name__ == "__main__":
    main()
