"""
plot_eda_split_end_version.py
=============================
Script vẽ lại Figure 1 (Partition Class Distribution) chuẩn Elsevier Data in Brief:
  - Loại bỏ tiền tố 'IC4SDMacroWood' trong tiêu đề: 'Partition Class Distribution (Total: 6,414)'
  - Tăng tỷ lệ chiều cao (taller figure, figsize=(12, 7.5)) giúp các cột cao ráo, rõ ràng
  - Viết tắt chữ đầu của chi thực vật (A. africana, D. cochinchinensis, P. pubescens...)
  - Số liệu chuẩn xác 100%: Train: 3,959 (61.7%) / Val: 1,265 (19.7%) / Test: 1,190 (18.6%)
  - Xuất ra cả file PDF (vector) và PNG (300 DPI) vào thư mục paper_data/fig/

Sử dụng:
    python plot_eda_split_end_version.py
"""

import os
import sys

# Đảm bảo in tiếng Việt có dấu an toàn trên Windows terminal (cmd/powershell)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_eda_distribution(split_csv_path: Optional[Path] = None, output_dir: Optional[Path] = None):
    # Xác định thư mục lưu ảnh đích (mặc định paper_data/fig)
    if output_dir:
        fig_dir = Path(output_dir)
    else:
        fig_dir = Path(__file__).resolve().parent.parent / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thử nạp trực tiếp từ file CSV phân vùng chính thức (out/splits/split_canonical.csv)
    candidate_csvs = [
        split_csv_path,
        Path("out/splits/split_canonical.csv"),
        Path(__file__).resolve().parent.parent.parent / "out" / "splits" / "split_canonical.csv",
        Path("paper_data_assets/splits/split_canonical.csv")
    ]
    
    csv_file = None
    for cand in candidate_csvs:
        if cand and Path(cand).exists():
            csv_file = Path(cand)
            break

    data = []
    if csv_file:
        print(f"[*] Đang đọc dữ liệu phân vùng trực tiếp từ: {csv_file}")
        df = pd.read_csv(csv_file)
        c_counts = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
        
        for full_name in sorted(df["class_name"].unique()):
            genus, spec = full_name.split(" ", 1)
            short_name = f"{genus[0]}. {spec}"
            t_cnt = int(c_counts.loc[full_name, "train"]) if "train" in c_counts.columns else 0
            v_cnt = int(c_counts.loc[full_name, "val"]) if "val" in c_counts.columns else 0
            te_cnt = int(c_counts.loc[full_name, "test"]) if "test" in c_counts.columns else 0
            data.append((full_name, short_name, t_cnt, v_cnt, te_cnt))
    else:
        print("[!] Không tìm thấy file split CSV, sử dụng dữ liệu tĩnh chuẩn 6,414 ảnh...")
        # Dữ liệu phân vùng 19 loài chuẩn xác 100% sau khi lọc ảnh mờ (Tổng 6,414 ảnh: Train 3,959 / Val 1,265 / Test 1,190)
        data = [
            ("Afzelia africana", "A. africana", 129, 54, 58),
            ("Afzelia bella", "A. bella", 240, 80, 80),
            ("Afzelia pachyloba", "A. pachyloba", 44, 32, 40),
            ("Afzelia quanzensis", "A. quanzensis", 232, 81, 56),
            ("Dalbergia cochinchinensis", "D. cochinchinensis", 212, 71, 71),
            ("Dalbergia melanoxylon", "D. melanoxylon", 172, 59, 60),
            ("Dalbergia oliveri", "D. oliveri", 190, 63, 63),
            ("Dalbergia rimosa", "D. rimosa", 180, 60, 60),
            ("Dalbergia tonkinensis", "D. tonkinensis", 195, 63, 67),
            ("Guibourtia arnoldiana", "G. arnoldiana", 188, 64, 71),
            ("Guibourtia coleosperma", "G. coleosperma", 216, 72, 72),
            ("Guibourtia ehie", "G. ehie", 320, 40, 40),
            ("Peltogyne pubescens", "P. pubescens", 220, 75, 76),
            ("Pterocarpus erinaceus", "P. erinaceus", 203, 64, 69),
            ("Pterocarpus indicus", "P. indicus", 163, 92, 57),
            ("Pterocarpus macrocarpus", "P. macrocarpus", 331, 54, 47),
            ("Pterocarpus soyauxii", "P. soyauxii", 350, 68, 68),
            ("Sindora cochinchinensis", "S. cochinchinensis", 182, 105, 67),
            ("Sindora tonkinensis", "S. tonkinensis", 192, 68, 68)
        ]

    short_names = [d[1] for d in data]
    train_counts = [d[2] for d in data]
    val_counts = [d[3] for d in data]
    test_counts = [d[4] for d in data]

    total_train = sum(train_counts)
    total_val = sum(val_counts)
    total_test = sum(test_counts)
    total_all = total_train + total_val + total_test

    p_train = total_train / total_all * 100
    p_val = total_val / total_all * 100
    p_test = total_test / total_all * 100

    print("=" * 68)
    print("VẼ BIỂU ĐỒ PHÂN BỐ PHÂN VÙNG (FIGURE 1) - TĂNG CHIỀU CAO & RÚT GỌN TÊN")
    print("=" * 68)
    print(f"Tổng số ảnh : {total_all:,}")
    print(f"  - Train    : {total_train:,} ({p_train:.1f}%)")
    print(f"  - Val      : {total_val:,} ({p_val:.1f}%)")
    print(f"  - Test     : {total_test:,} ({p_test:.1f}%)")

    # Khởi tạo figure với chiều cao dài ra nổi bật (figsize=(10.5, 8.5)), tỷ lệ thanh thoát chuẩn journal
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=300)

    indices = np.arange(len(short_names))
    width = 0.54

    # Vẽ stacked bar chart
    p1 = ax.bar(indices, train_counts, width, label="train", color="#1f77b4")
    p2 = ax.bar(indices, val_counts, width, bottom=train_counts, label="val", color="#ff7f0e")
    bottom_test = [t + v for t, v in zip(train_counts, val_counts)]
    p3 = ax.bar(indices, test_counts, width, bottom=bottom_test, label="test", color="#2ca02c")

    # Điền phần trăm bên trong các đoạn cột
    for i in range(len(short_names)):
        t_val = train_counts[i]
        v_val = val_counts[i]
        te_val = test_counts[i]
        c_total = t_val + v_val + te_val

        # Train label
        if t_val > 0:
            ax.text(i, t_val / 2.0, f"{t_val / c_total * 100:.1f}%",
                    ha="center", va="center", fontsize=6.5, color="white", fontweight="semibold")

        # Val label
        if v_val > 0:
            ax.text(i, t_val + v_val / 2.0, f"{v_val / c_total * 100:.1f}%",
                    ha="center", va="center", fontsize=6.5, color="white", fontweight="semibold")

        # Test label
        if te_val > 0:
            ax.text(i, t_val + v_val + te_val / 2.0, f"{te_val / c_total * 100:.1f}%",
                    ha="center", va="center", fontsize=6.5, color="white", fontweight="semibold")

    # Hộp thông tin tỷ lệ tổng thể (Overall split ratio) ở góc trên bên trái
    overall_text = (
        f"Overall split ratio:\n"
        f"  Train: {total_train:,} ({p_train:.1f}%)\n"
        f"  Val:   {total_val:,} ({p_val:.1f}%)\n"
        f"  Test:  {total_test:,} ({p_test:.1f}%)"
    )
    props = dict(boxstyle='round,pad=0.55', facecolor='#faedd0', edgecolor='#8c7b64', alpha=0.92, linewidth=1.1)
    ax.text(0.018, 0.965, overall_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=props, fontfamily="monospace")

    # Tiêu đề: Đã bỏ chữ "IC4SDMacroWood"
    ax.set_title(f"Partition Class Distribution (Total: {total_all:,})", fontsize=13.5, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=11.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Image count", fontsize=11.5, fontweight="bold", labelpad=10)

    # Đặt nhãn trục hoành là tên viết tắt của chi, in nghiêng chuẩn danh pháp sinh học
    ax.set_xticks(indices)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9.5, fontstyle="italic", fontweight="normal")
    ax.set_ylim(0, 520)

    # Legend ở góc trên bên phải
    ax.legend(loc="upper right", fontsize=10.5, frameon=True, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")

    plt.tight_layout()

    # Xuất file chuẩn publication
    save_pdf = fig_dir / "eda_split_end_version.pdf"
    save_png = fig_dir / "eda_split_end_version.png"
    plt.savefig(str(save_pdf), bbox_inches="tight")
    plt.savefig(str(save_png), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Đã lưu thành công biểu đồ phân bố phân vùng:")
    print(f"  - PDF : {save_pdf.resolve()}")
    print(f"  - PNG : {save_png.resolve()}")
    print("=" * 68)

if __name__ == "__main__":
    plot_eda_distribution()
