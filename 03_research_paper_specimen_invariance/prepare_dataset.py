"""
prepare_dataset.py
==================
Filters the timber metadata to exactly 11 target species (110 physical vouchers, 3,702 images),
and creates deterministic dual-split partitions (Seen In-Specimen vs. Unseen Physical Specimen holdouts)
using a fixed random seed (Seed 42).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# 8 species explicitly excluded due to limited physical vouchers or data instability
EXCLUDED_SPECIES = [
    ("Sindora", "cochinchinensis"),
    ("Pterocarpus", "soyauxii"),
    ("Afzelia", "pachyloba"),
    ("Afzelia", "quanzensis"),
    ("Afzelia", "africana"),
    ("Guibourtia", "coleosperma"),
    ("Pterocarpus", "macrocarpus"),
    ("Dalbergia", "cochinchinensis"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 11-species timber dataset and splits")
    parser.add_argument(
        "--raw_metadata",
        type=str,
        default="01_data_paper_forensic_cites/out/metadata/metadata.csv",
        help="Path to raw source metadata CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="03_research_paper_specimen_invariance/data",
        help="Directory to save filtered metadata and splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for deterministic splitting"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_metadata)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        # Fallback to look relative to repo root if run from subfolder
        raw_path = Path("..") / raw_path
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot find raw metadata at {args.raw_metadata}")

    print(f"[*] Loading raw metadata from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"    - Raw total records: {len(df)}")

    # 1. Filter excluded species
    mask_excluded = df.apply(lambda r: (r["genus"], r["species"]) in EXCLUDED_SPECIES, axis=1)
    df_filtered = df[~mask_excluded].copy().reset_index(drop=True)

    print(f"[*] Excluded {mask_excluded.sum()} images across 8 noisy/insufficient taxa.")
    print(f"[+] Retained {len(df_filtered)} images across target clean taxa.")

    # 2. Validate taxonomic distribution
    taxa_groups = df_filtered.groupby(["genus", "species"])
    num_taxa = len(taxa_groups)
    print(f"[+] Total remaining taxa: {num_taxa}")

    species_list = []
    specimen_counts = {}
    for (genus, species), group in taxa_groups:
        full_name = f"{genus} {species}"
        species_list.append(full_name)
        n_vouchers = group["specimen_id"].nunique()
        specimen_counts[full_name] = n_vouchers
        print(f"    - {full_name:30s}: {n_vouchers} vouchers, {len(group)} images")
        if n_vouchers != 10:
            print(f"      [!] Warning: {full_name} has {n_vouchers} vouchers (expected 10)")

    species_list = sorted(species_list)
    species_to_idx = {name: i for i, name in enumerate(species_list)}
    df_filtered["taxon_name"] = df_filtered["genus"] + " " + df_filtered["species"]
    df_filtered["species_idx"] = df_filtered["taxon_name"].map(species_to_idx)

    # Global specimen ID mapping
    unique_vouchers = sorted(df_filtered["specimen_id"].unique())
    voucher_to_global_idx = {v: i for i, v in enumerate(unique_vouchers)}
    df_filtered["global_specimen_idx"] = df_filtered["specimen_id"].map(voucher_to_global_idx)

    # 3. Create Dual-Split Partitions with Seed 42
    rng = np.random.RandomState(args.seed)

    # --- Split A: Unseen Physical Specimens Split (8 vouchers train / 2 vouchers test per species) ---
    train_vouchers_all = []
    test_unseen_vouchers_all = []

    for taxon in species_list:
        taxon_df = df_filtered[df_filtered["taxon_name"] == taxon]
        vouchers = sorted(taxon_df["specimen_id"].unique())
        # Deterministic shuffle
        permuted = rng.permutation(vouchers).tolist()
        # 8 for training, 2 for strictly unseen out-of-specimen testing
        train_v = permuted[:8]
        test_v = permuted[8:]
        train_vouchers_all.extend(train_v)
        test_unseen_vouchers_all.extend(test_v)

    train_vouchers_set = set(train_vouchers_all)
    test_unseen_vouchers_set = set(test_unseen_vouchers_all)

    is_train_specimen = df_filtered["specimen_id"].isin(train_vouchers_set)
    df_train_pool = df_filtered[is_train_specimen].copy().reset_index(drop=True)
    df_test_unseen = df_filtered[~is_train_specimen].copy().reset_index(drop=True)

    # --- Split B: In-Specimen Seen Test Holdout (20% holdout from train pool, per species) ---
    train_indices_final = []
    seen_test_indices = []

    for taxon in species_list:
        sub_df = df_train_pool[df_train_pool["taxon_name"] == taxon]
        sub_indices = sub_df.index.tolist()
        shuffled = rng.permutation(sub_indices).tolist()
        n_test = int(round(len(shuffled) * 0.20))
        seen_test_indices.extend(shuffled[:n_test])
        train_indices_final.extend(shuffled[n_test:])

    df_train_final = df_train_pool.loc[train_indices_final].copy().reset_index(drop=True)
    df_test_seen = df_train_pool.loc[seen_test_indices].copy().reset_index(drop=True)

    # Mark partition column
    # 0 = train, 1 = test_seen (in-specimen), 2 = test_unseen (out-of-specimen)
    df_filtered["eval_split"] = "unassigned"
    df_filtered.loc[df_filtered["image_id"].isin(df_train_final["image_id"]), "eval_split"] = "train"
    df_filtered.loc[df_filtered["image_id"].isin(df_test_seen["image_id"]), "eval_split"] = "test_seen"
    df_filtered.loc[df_filtered["image_id"].isin(df_test_unseen["image_id"]), "eval_split"] = "test_unseen"

    # 4. Save metadata CSV and Split JSON
    csv_out = out_dir / "metadata_11species.csv"
    df_filtered.to_csv(csv_out, index=False)
    print(f"\n[+] Saved clean metadata to: {csv_out}")

    split_manifest = {
        "seed": args.seed,
        "num_species": len(species_list),
        "species_list": species_list,
        "species_to_idx": species_to_idx,
        "num_train_vouchers": len(train_vouchers_all),
        "num_unseen_vouchers": len(test_unseen_vouchers_all),
        "train_vouchers": [str(v) for v in sorted(train_vouchers_all)],
        "test_unseen_vouchers": [str(v) for v in sorted(test_unseen_vouchers_all)],
        "counts": {
            "total_images": len(df_filtered),
            "train_images": len(df_train_final),
            "test_seen_images": len(df_test_seen),
            "test_unseen_images": len(df_test_unseen),
        },
        "image_ids": {
            "train": df_train_final["image_id"].tolist(),
            "test_seen": df_test_seen["image_id"].tolist(),
            "test_unseen": df_test_unseen["image_id"].tolist(),
        }
    }

    json_out = out_dir / "splits_11species.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"[+] Saved split manifest to: {json_out}")

    print("\n" + "=" * 60)
    print("  DUAL-SPLIT PARTITION SUMMARY (Seed 42)")
    print("=" * 60)
    print(f"  Total Images          : {len(df_filtered)}")
    print(f"  Training Set          : {len(df_train_final)} images (across 88 physical vouchers)")
    print(f"  Test Seen (In-Spec)   : {len(df_test_seen)} images (across same 88 vouchers)")
    print(f"  Test Unseen (Out-Spec): {len(df_test_unseen)} images (across 22 strictly UNSEEN vouchers)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
