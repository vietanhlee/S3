"""
plot_confusion_matrix_focal.py
==============================
Script vẽ lại ma trận nhầm lẫn (Confusion Matrix) chuẩn xác 100% khớp với kết quả thực nghiệm 
Multiclass Focal Loss trên tập Test (N=1,190 ảnh) cho bài báo Data in Brief.

Đảm bảo:
  - Khớp 100% từng giá trị ma trận và tỷ lệ phần trăm trong ảnh benchmark
  - Tiêu đề hiển thị chuẩn: N=1,190
  - Xuất ra cả file PDF (vector) và PNG (300 DPI) vào thư mục fig/

Sử dụng:
    python plot_confusion_matrix_focal.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_cm():
    fig_dir = Path(__file__).parent / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "Afzelia africana",
        "Afzelia bella",
        "Afzelia pachyloba",
        "Afzelia quanzensis",
        "Dalbergia cochinchinensis",
        "Dalbergia melanoxylon",
        "Dalbergia oliveri",
        "Dalbergia rimosa",
        "Dalbergia tonkinensis",
        "Guibourtia arnoldiana",
        "Guibourtia coleosperma",
        "Guibourtia ehie",
        "Peltogyne pubescens",
        "Pterocarpus erinaceus",
        "Pterocarpus indicus",
        "Pterocarpus macrocarpus",
        "Pterocarpus soyauxii",
        "Sindora cochinchinensis",
        "Sindora tonkinensis"
    ]

    short_names = [
        "A. africana",
        "A. bella",
        "A. pachyloba",
        "A. quanzensis",
        "D. cochinchinensis",
        "D. melanoxylon",
        "D. oliveri",
        "D. rimosa",
        "D. tonkinensis",
        "G. arnoldiana",
        "G. coleosperma",
        "G. ehie",
        "Peltogyne pubescens",
        "P. erinaceus",
        "P. indicus",
        "P. macrocarpus",
        "P. soyauxii",
        "S. cochinchinensis",
        "S. tonkinensis"
    ]

    # Ma trận nhầm lẫn thực nghiệm chuẩn xác 100% theo ảnh benchmark (N_test = 1,190)
    cm = np.zeros((19, 19), dtype=int)

    # 0: Afzelia africana (supp 58)
    cm[0, 0] = 34   # 59%
    cm[0, 1] = 6    # 10%
    cm[0, 3] = 9    # 16%
    cm[0, 16] = 9   # 16%

    # 1: Afzelia bella (supp 80)
    cm[1, 1] = 80   # 100%

    # 2: Afzelia pachyloba (supp 40)
    cm[2, 1] = 21   # 52%
    cm[2, 2] = 1    # 2%
    cm[2, 7] = 2    # 5% (D. rimosa)
    cm[2, 11] = 13  # 32% (G. ehie)
    cm[2, 16] = 3   # 8% (P. soyauxii)

    # 3: Afzelia quanzensis (supp 56)
    cm[3, 3] = 32   # 57%
    cm[3, 10] = 24  # 43% (G. coleosperma)

    # 4: Dalbergia cochinchinensis (supp 71)
    cm[4, 4] = 70   # 99%
    cm[4, 7] = 1    # 1% (D. rimosa)

    # 5: Dalbergia melanoxylon (supp 60)
    cm[5, 5] = 60   # 100%

    # 6: Dalbergia oliveri (supp 63)
    cm[6, 6] = 60   # 95%
    cm[6, 15] = 3   # 5% (P. macrocarpus)

    # 7: Dalbergia rimosa (supp 60)
    cm[7, 7] = 55   # 92%
    cm[7, 8] = 5    # 8% (D. tonkinensis)

    # 8: Dalbergia tonkinensis (supp 67)
    cm[8, 8] = 67   # 100%

    # 9: Guibourtia arnoldiana (supp 71)
    cm[9, 9] = 69   # 97%
    cm[9, 18] = 2   # 3% (S. tonkinensis)

    # 10: Guibourtia coleosperma (supp 72)
    cm[10, 10] = 72 # 100%

    # 11: Guibourtia ehie (supp 40)
    cm[11, 11] = 40 # 100%

    # 12: Peltogyne pubescens (supp 76)
    cm[12, 12] = 76 # 100%

    # 13: Pterocarpus erinaceus (supp 69)
    cm[13, 0] = 1   # 1% (A. africana)
    cm[13, 13] = 66 # 96%
    cm[13, 14] = 2  # 3% (P. indicus)

    # 14: Pterocarpus indicus (supp 57)
    cm[14, 13] = 7  # 12% (P. erinaceus)
    cm[14, 14] = 50 # 88%

    # 15: Pterocarpus macrocarpus (supp 47)
    cm[15, 15] = 47 # 100%

    # 16: Pterocarpus soyauxii (supp 68)
    cm[16, 15] = 2  # 3% (P. macrocarpus)
    cm[16, 16] = 66 # 97%

    # 17: Sindora cochinchinensis (supp 67)
    cm[17, 17] = 67 # 100%

    # 18: Sindora tonkinensis (supp 68)
    cm[18, 8] = 1   # 1% (D. tonkinensis)
    cm[18, 9] = 1   # 1% (G. arnoldiana)
    cm[18, 13] = 2  # 3% (P. erinaceus)
    cm[18, 18] = 64 # 94%

    total_samples = int(cm.sum())
    total_correct = int(np.trace(cm))
    accuracy = total_correct / total_samples

    print("=" * 68)
    print("XUẤT MA TRẬN NHẦM LẪN CHUẨN XÁC 100% CHO FOCAL LOSS")
    print("=" * 68)
    print(f"[+] Tổng số mẫu kiểm thử: {total_samples:,} (N_test = 1,190)")
    print(f"[+] Số mẫu phân loại đúng: {total_correct:,} / {total_samples:,} ({accuracy*100:.2f}%)")

    # Chuẩn hóa theo hàng để tính tỷ lệ % (Recall của từng lớp)
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    # Khởi tạo khung hình với tỷ lệ chuẩn xác
    fig, ax = plt.subplots(figsize=(11.5, 9.5), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Tiêu đề với N=1,190
    ax.set_title(f"ConvNeXt-Tiny Baseline — Confusion Matrix (Test Split, N={total_samples:,})", 
                 fontsize=12, fontweight="bold", pad=14)
    
    # Thanh màu chuẩn
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Đặt nhãn các trục
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8.5, fontweight="medium")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=8.5, fontweight="medium")

    # Điền giá trị vào từng ô (chỉ điền ô có giá trị > 0)
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                pct = int(round(cm_norm[i, j] * 100))
                ax.text(j, i, f"{val}\n({pct}%)",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=6.5,
                        fontweight="bold",
                        color="white" if cm_norm[i, j] > thresh else "black")

    ax.set_ylabel("Ground-Truth Species Nomenclature", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Taxonomic Epithet", fontsize=10.5, fontweight="bold", labelpad=8)
    
    plt.tight_layout()

    # Lưu cả file PDF (vector) và PNG (high-res)
    save_pdf = fig_dir / "confusion_matrix_focal_test.pdf"
    save_png = fig_dir / "confusion_matrix_focal_test.png"
    plt.savefig(str(save_pdf), bbox_inches="tight")
    plt.savefig(str(save_png), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Đã lưu thành công ma trận nhầm lẫn:")
    print(f"  - PDF : {save_pdf.resolve()}")
    print(f"  - PNG : {save_png.resolve()}")
    print("=" * 68)

if __name__ == "__main__":
    plot_cm()
