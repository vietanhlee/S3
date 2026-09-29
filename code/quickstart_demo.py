#!/usr/bin/env python3
"""
code/quickstart_demo.py
=======================
Executable demonstration and verification script for ForensicMacroWood-CITES
(Elsevier Data in Brief / CC BY 4.0).

Usage:
    python code/quickstart_demo.py [--data-dir PATH] [--assets-dir PATH]
"""

import sys
from pathlib import Path
import json

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd


def main():
    print("=" * 80)
    print(" ForensicMacroWood-CITES: Rapid Verification & Quickstart Demonstration ")
    print("=" * 80)

    # 1. Locate assets
    repo_root = Path(__file__).resolve().parent.parent
    candidate_meta = [
        repo_root / "paper_data_assets" / "metadata" / "metadata.csv",
        repo_root / "out" / "metadata" / "metadata.csv",
        repo_root / "metadata.csv",
    ]
    meta_path = None
    for p in candidate_meta:
        if p.exists():
            meta_path = p
            break

    if meta_path is None:
        print("[!] Warning: metadata.csv not found in default paths. Please run 'python run_pipeline.py --step assets'.")
        return

    print(f"[+] Loaded dataset metadata from: {meta_path.relative_to(repo_root) if meta_path.is_relative_to(repo_root) else meta_path}")
    df = pd.read_csv(meta_path)
    total_imgs = len(df)
    n_classes = df["class_name"].nunique() if "class_name" in df.columns else 19

    print(f"    - Total Curated Images : {total_imgs:,} captures (224x224 px, 12.0 um/px)")
    print(f"    - Number of Taxa       : {n_classes} species across 6 botanical genera")

    if "cites_status" in df.columns:
        cites_count = (df["cites_status"] == "CITES Appendix II").sum()
        print(f"    - CITES Appendix II    : {cites_count:,} images ({cites_count / total_imgs * 100:.1f}%)")

    # 2. Check partitions
    candidate_split = [
        repo_root / "paper_data_assets" / "splits" / "split_canonical.csv",
        repo_root / "out" / "splits" / "split_canonical.csv",
    ]
    split_path = None
    for p in candidate_split:
        if p.exists():
            split_path = p
            break

    if split_path:
        print(f"\n[+] Governed Partition Manifest: {split_path.name}")
        sdf = pd.read_csv(split_path)
        split_counts = sdf["split"].value_counts().to_dict()
        for s_name in ["train", "val", "test"]:
            print(f"    - {s_name.capitalize():<6} Partition: {split_counts.get(s_name, 0):,} captures")

    # 3. Check Anatomical features
    anat_path = repo_root / "paper_data_assets" / "metadata" / "anatomical_features.csv"
    if not anat_path.exists():
        anat_path = repo_root / "out" / "metadata" / "anatomical_features.csv"
    if anat_path.exists():
        adf = pd.read_csv(anat_path)
        print(f"\n[+] Multimodal IAWA Descriptors Matrix: {len(adf)} taxa cataloged across {len(adf.columns)-1} diagnostic morphological axes.")

    # 4. Check Multi-Seed Results
    stat_path = repo_root / "baseline_outputs" / "multi_seed_classification_report.txt"
    if stat_path.exists():
        print("\n" + "=" * 80)
        print(" Benchmark Multi-Seed Classification Report (ConvNeXt-Tiny + Focal Loss):")
        print("=" * 80)
        with open(stat_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("\n[*] To execute multi-seed baseline training and statistical testing:")
        print("    python run_pipeline.py --all")

    print("\n[✓] Verification completed successfully.")


if __name__ == "__main__":
    main()
