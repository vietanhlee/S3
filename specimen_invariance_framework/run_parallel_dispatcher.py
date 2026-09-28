"""
specimen_invariance_framework/run_parallel_dispatcher.py
========================================================
Automatic Multi-GPU Task Dispatcher & Parallel Runner.

Functionality:
1. Automatically detects available CUDA GPUs via PyTorch (torch.cuda.device_count()).
2. Spawns a parallel worker pool matching the exact number of detected GPUs.
3. Automatically queues all experimental methods (or backbones) and dispatches tasks
   dynamically to the next free GPU as soon as a previous run completes.
4. Isolates each process on its dedicated GPU (--gpu <id> and CUDA_VISIBLE_DEVICES).
5. Directs detailed outputs to per-method log files while displaying a clean real-time
   progress dashboard in the console.
6. Prints an executive completion summary table with durations and exit statuses.

Usage:
    # Run all 13 baselines in parallel across all detected GPUs:
    python run_parallel_dispatcher.py --mode baselines --fold 0 --epochs 40

    # Run backbone ablations in parallel across all detected GPUs:
    python run_parallel_dispatcher.py --mode backbones --fold 0 --epochs 40

    # Run custom methods:
    python run_parallel_dispatcher.py --methods conditional_grl ce focal arcface
"""

import argparse
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from typing import List, Dict, Any, Optional

import torch


BASELINES = [
    "conditional_grl",
    "focal",
    "arcface",
    "strong_reg",
    "mixup",
    "dann_unconditional",
    "club",
    "supcon",
    "semihard_triplet",
    "frozen_linear",
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


def worker_loop(
    gpu_id: Optional[int],
    task_queue: Queue,
    results_list: List[Dict[str, Any]],
    script_args: argparse.Namespace,
    log_dir: Path,
):
    """Worker thread that executes queued tasks on a pinned GPU."""
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

        start_time = time.time()
        start_dt = datetime.now().strftime("%H:%M:%S")
        safe_print(f"[{start_dt}] [{gpu_str}] >>> LAUNCHING: {method} ({backbone}) -> Log: {log_file.name}")

        # Build execution command
        cmd = [
            sys.executable,
            "train.py",
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
            cmd.extend(["--gpu", str(gpu_id)])

        # Set environment with pinned GPU
        env = os.environ.copy()
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
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
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
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Training Dispatcher")
    parser.add_argument("--mode", type=str, default="baselines", choices=["baselines", "backbones", "custom"],
                        help="Mode: 'baselines' runs 13 methods; 'backbones' runs 4 architectures; 'custom' uses --methods")
    parser.add_argument("--methods", nargs="+", type=str, default=None,
                        help="Custom list of methods to run (when --mode custom)")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Default backbone")
    parser.add_argument("--fold", type=int, default=0, help="Round-robin fold index")
    parser.add_argument("--epochs", type=int, default=17, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="Explicit list of GPU IDs to use (e.g. 0 1). If None, auto-detects all available GPUs.")
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--image_root", type=str, default="out", help="Image directory root")
    parser.add_argument("--output_base_dir", type=str, default="specimen_invariance_outputs", help="Output base directory")
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
            gpu_ids = [None]  # Fallback to single CPU worker
            print("[!] No CUDA GPUs detected. Falling back to single CPU process execution.")

    num_workers = len(gpu_ids)

    # 2. Build Task List
    task_queue = Queue()
    if args.mode == "baselines":
        task_methods = BASELINES
        for m in task_methods:
            task_queue.put({"method": m, "backbone": args.backbone, "fold": args.fold, "seed": args.seed})
    elif args.mode == "backbones":
        for b in BACKBONES:
            task_queue.put({"method": "conditional_grl", "backbone": b, "fold": args.fold, "seed": args.seed})
    elif args.mode == "custom":
        if not args.methods:
            raise ValueError("When using --mode custom, please specify --methods <m1> <m2> ...")
        for m in args.methods:
            task_queue.put({"method": m, "backbone": args.backbone, "fold": args.fold, "seed": args.seed})

    total_tasks = task_queue.qsize()
    print(f"\n[+] Total queued tasks: {total_tasks} | Worker concurrency: {num_workers} parallel workers")
    print(f"[+] Detailed output logs will be stored in: {log_dir.resolve()}\n")

    # 3. Spawn Parallel Workers
    results_list: List[Dict[str, Any]] = []
    threads = []
    t_start = time.time()

    for worker_idx in range(num_workers):
        assigned_gpu = gpu_ids[worker_idx]
        t = Thread(
            target=worker_loop,
            args=(assigned_gpu, task_queue, results_list, args, log_dir),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for queue to be empty
    task_queue.join()

    # Stop workers
    for _ in range(num_workers):
        task_queue.put(None)
    for t in threads:
        t.join()

    total_duration_min = (time.time() - t_start) / 60.0

    # 4. Executive Summary Report
    print("\n" + "=" * 90)
    print("                    MULTI-GPU PARALLEL DISPATCHER SUMMARY REPORT                    ")
    print("=" * 90)
    print(f"{'Method':<20} | {'Backbone':<18} | {'Device':<8} | {'Duration (m)':<12} | {'Status':<15}")
    print("-" * 90)
    for r in results_list:
        print(f"{r['method']:<20} | {r['backbone']:<18} | {r['gpu']:<8} | {r['duration_min']:<12.2f} | {r['status']:<15}")
    print("=" * 90)
    print(f"[+] Total execution time: {total_duration_min:.2f} minutes across {num_workers} parallel GPU worker(s).")
    print(f"[+] Check individual model checkpoints in '{args.output_base_dir}' for evaluations.")


if __name__ == "__main__":
    main()
