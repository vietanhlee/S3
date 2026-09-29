# 🎯 Lộ Trình Đăng Q1 Journal — Fine-Grained Wood Species Identification

> **Dự án hiện tại:** S3 Wood Dataset — 19 loài gỗ (6 chi: Afzelia, Dalbergia, Guibourtia, Peltogyne, Pterocarpus, Sindora)
> **Backbone:** ConvNeXt-Tiny (pretrained, freeze 90%)
> **Đã có:** Classification baseline (Focal Loss), 17+ Metric Learning methods, Barlow Twins SSL, End Version Split, Grad-CAM, t-SNE, Clustering Metrics, WandB tracking

---

## Mục Lục

1. [Novelty — Đóng góp mới](#1-novelty--đóng-góp-mới)
2. [Domain-Specific Data Augmentation](#2-domain-specific-data-augmentation)
3. [Ablation Study](#3-ablation-study)
4. [Statistical Significance](#4-statistical-significance)
5. [Cross-Dataset Validation](#5-cross-dataset-validation)
6. [Practical Deployment Analysis](#6-practical-deployment-analysis)
7. [So sánh với SOTA đã công bố](#7-so-sánh-với-sota-đã-công-bố)
8. [Cấu trúc Paper đề xuất](#8-cấu-trúc-paper-đề-xuất)
9. [Journal mục tiêu](#9-journal-mục-tiêu)

---

## 1. Novelty — Đóng góp mới

> [!IMPORTANT]
> Reviewer Q1 sẽ hỏi đầu tiên: *"What is new? Why should I care?"*. Chỉ benchmark thì tối đa đạt Q2. Cần ít nhất **1 novelty rõ ràng** (ideally 2-3).

### 1.1 Taxonomy-Aware Hierarchical Loss (Novelty chính — Đề xuất phương pháp mới)

**Ý tưởng cốt lõi:** Cấu trúc phân loại học của gỗ có dạng cây phả hệ:

```
Family
  └── Genus (Chi)
        └── Species (Loài)
```

Các loss function hiện tại (Triplet, Contrastive, ArcFace,...) coi mọi cặp negative pair là **như nhau** — tức là phạt việc nhầm Afzelia africana với Afzelia bella (cùng chi) **bằng đúng mức phạt** nhầm với Dalbergia oliveri (khác chi hoàn toàn). Điều này phi lý về mặt sinh học.

**Thiết kế cụ thể:**

```
Hierarchical Margin:
  - Cùng loài (same species):     margin = 0 (anchor-positive → kéo lại gần)
  - Cùng chi, khác loài:          margin = m₁ (nhỏ hơn, cho phép gần hơn)
  - Khác chi hoàn toàn:           margin = m₂ (lớn hơn, đẩy ra xa)
  
  Với m₁ < m₂ (ví dụ: m₁ = 0.3, m₂ = 0.8)
```

**Cách code loss function:**

```python
class TaxonomyAwareTripletLoss(nn.Module):
    """Triplet Loss với margin phân cấp theo cấu trúc phân loại học.
    
    - Cặp negative cùng chi (intra-genus): margin nhỏ (m₁)
      → cho phép các loài cùng chi nằm tương đối gần nhau
    - Cặp negative khác chi (inter-genus): margin lớn (m₂)
      → buộc phải phân tách rõ ràng giữa các chi
    """
    
    def __init__(self, m_intra_genus=0.3, m_inter_genus=0.8):
        super().__init__()
        self.m_intra = m_intra_genus  # margin cho cặp cùng chi
        self.m_inter = m_inter_genus  # margin cho cặp khác chi
    
    def forward(self, embeddings, labels, genus_labels):
        # genus_labels: label ở cấp Chi (0-5 cho 6 chi)
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        loss = 0.0
        count = 0
        for i in range(len(labels)):
            # Positive: cùng loài
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels)) != i)
            # Negative cùng chi: khác loài nhưng cùng chi
            neg_intra_mask = (labels != labels[i]) & (genus_labels == genus_labels[i])
            # Negative khác chi: khác chi hoàn toàn  
            neg_inter_mask = (genus_labels != genus_labels[i])
            
            if pos_mask.any() and neg_intra_mask.any():
                d_pos = dist_mat[i][pos_mask].max()  # hardest positive
                d_neg = dist_mat[i][neg_intra_mask].min()  # hardest negative cùng chi
                loss += F.relu(d_pos - d_neg + self.m_intra)
                count += 1
            
            if pos_mask.any() and neg_inter_mask.any():
                d_pos = dist_mat[i][pos_mask].max()
                d_neg = dist_mat[i][neg_inter_mask].min()  # hardest negative khác chi
                loss += F.relu(d_pos - d_neg + self.m_inter)
                count += 1
        
        return loss / max(count, 1)
```

**Biến thể nâng cao — Taxonomy-Aware ArcFace:**

```python
class TaxonomyAwareArcFace(nn.Module):
    """ArcFace với angular margin thay đổi theo mức phân loại học.
    
    Margin cho lớp đúng: m (tiêu chuẩn, ví dụ 0.5)
    Penalty cho lớp cùng chi: giảm margin xuống m * α (α < 1)
    Penalty cho lớp khác chi: giữ nguyên m
    
    → Hiệu quả: mô hình "khoan dung" hơn với sai lầm trong cùng chi,
      nhưng "khắt khe" với sai lầm xuyên chi.
    """
    
    def __init__(self, embedding_dim, num_classes, genus_of_class,
                 scale=30.0, margin=0.5, intra_genus_factor=0.6):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.intra_genus_factor = intra_genus_factor
        self.genus_of_class = genus_of_class  # dict: class_idx → genus_idx
        
        # Xây dựng ma trận margin: (num_classes, num_classes)
        self.margin_matrix = self._build_margin_matrix(num_classes)
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def _build_margin_matrix(self, num_classes):
        """Tạo ma trận margin: M[i][j] = margin khi true_class=i, predicted=j."""
        M = torch.full((num_classes, num_classes), self.margin)
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    M[i][j] = self.margin  # margin chuẩn cho lớp đúng
                elif self.genus_of_class[i] == self.genus_of_class[j]:
                    M[i][j] = self.margin * self.intra_genus_factor  # giảm cho cùng chi
        return M
```

**Tại sao đây là novelty mạnh:**
- Chưa có paper nào áp dụng hierarchical taxonomy margin cho bài toán wood species identification
- Biện luận sinh học rõ ràng: các loài cùng chi có cấu trúc vi mô tương đồng (xylem, vessel, parenchyma)
- Dễ tổng quát hóa sang các bài toán fine-grained classification khác có cấu trúc taxonomy (chim, hoa, côn trùng)

---

### 1.2 Embedding-Aware Data Splitting (Novelty phụ — End Version Split)

**Ý tưởng:** Phương pháp chia dữ liệu End Version Split hiện tại của bạn đã là một đóng góp có giá trị:

- Sử dụng **embedding similarity** (EfficientNetV2-M + Swin-Large) để phát hiện và tách các ảnh tương đồng về ngữ nghĩa (semantic similarity) ra các tập khác nhau
- **9 phương pháp chia** (PP1-PP9), mỗi phương pháp tối ưu cho một nhóm loài khác nhau
- **Kết hợp tối ưu cho từng loài** dựa trên F1-score thực nghiệm

**Cách biện luận thành novelty:**

| Vấn đề | Random Split truyền thống | End Version Split (của bạn) |
|--------|---------------------------|------------------------------|
| Data leakage | ❌ Các ảnh từ cùng mẫu gỗ (cùng subfolder) có thể rơi vào cả train/test | ✅ Dùng embedding similarity để phát hiện và tách |
| Đánh giá thiên vị | ❌ Test accuracy bị inflated do memorization | ✅ Test set chứa ảnh thực sự "khác biệt" |
| Class-specific | ❌ Áp dụng 1 chiến lược cho tất cả | ✅ Tối ưu riêng cho từng loài |

**Thí nghiệm cần chạy để chứng minh:**
1. So sánh accuracy trên **Random Split** vs **End Version Split** → chứng minh random split overestimate performance
2. Tính **cosine similarity** trung bình giữa train/test cho mỗi phương pháp → chứng minh End Version tách tốt hơn
3. Vẽ histogram phân phối khoảng cách embedding train↔test cho cả hai phương pháp

---

### 1.3 Comprehensive Benchmark Framework (Novelty phụ — Quy mô đánh giá)

Benchmark 17+ metric learning methods trên cùng một pipeline đồng nhất cho bài toán wood identification:

- **Pairwise losses:** Contrastive, Hard Contrastive
- **Triplet-based:** Triplet, Semi-Hard Triplet, Soft-Margin Triplet
- **Proxy-based:** Proxy Anchor, SoftTriple
- **Angular margin:** ArcFace, Sub-center ArcFace, Circle Loss, Angular Loss
- **Mining-based:** Multi-Similarity, Lifted Structured
- **Self-supervised:** SimCLR, BYOL, SimSiam, Barlow Twins, SupCon

> [!NOTE]
> Benchmark quy mô này chưa từng xuất hiện trong lĩnh vực wood identification. Các paper trước chỉ so sánh tối đa 3-4 phương pháp.

---

## 2. Domain-Specific Data Augmentation

### 2.1 Các kỹ thuật augmentation đặc thù cho ảnh cấu trúc gỗ

Ảnh mặt cắt gỗ có đặc điểm riêng biệt so với ảnh tự nhiên thông thường:

| Đặc điểm ảnh gỗ | Augmentation phù hợp | Lý do |
|------------------|----------------------|-------|
| **Hướng thớ gỗ (grain direction)** | Rotation 0°/90°/180°/270° + Random rotation ±30° | Mẫu gỗ có thể được cắt theo bất kỳ hướng nào |
| **Biến thiên màu sắc tự nhiên** | ColorJitter (brightness=0.3, contrast=0.3) | Cùng loài gỗ nhưng màu sắc thay đổi theo tuổi, độ ẩm, vị trí cây |
| **Biến thiên tỷ lệ phóng đại** | RandomResizedCrop (scale=0.6-1.0) | Ảnh macro chụp ở các mức zoom khác nhau |
| **Cấu trúc vân lặp lại (texture periodicity)** | RandomGrayscale (p=0.1) | Buộc mô hình học cấu trúc vân thay vì chỉ dựa vào màu |
| **Nhiễu ánh sáng kính hiển vi** | GaussianBlur (kernel=3-7) | Mô phỏng lỗi lấy nét khi chụp qua kính hiển vi |
| **Phản xạ bề mặt** | RandomAdjustSharpness | Bề mặt gỗ đánh bóng tạo hiệu ứng phản xạ khác nhau |

**Augmentation pipeline đề xuất (nâng cấp so với hiện tại):**

```python
# Hiện tại (trong utils/data.py build_transforms):
transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
transforms.RandomRotation(degrees=30)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)
transforms.RandomGrayscale(p=0.05)

# Đề xuất bổ sung cho paper:
# 1. Mở rộng vùng crop để buộc mô hình phải nhận diện từ ít thông tin hơn
transforms.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8, 1.2))

# 2. Bổ sung GaussianBlur mô phỏng mất nét kính hiển vi
transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

# 3. Tăng xác suất grayscale để ép học cấu trúc vân (texture)
transforms.RandomGrayscale(p=0.15)

# 4. Random Erasing mô phỏng vùng gỗ bị hư/nứt
transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))

# 5. Elastic Transform (cho ảnh microscopy)
# Cần dùng albumentations:
import albumentations as A
A.ElasticTransform(alpha=30, sigma=5, p=0.1)
# → Mô phỏng biến dạng nhẹ của mẫu gỗ khi cắt lát mỏng
```

### 2.2 Ablation augmentation (cần chạy)

Chạy **cùng một phương pháp tốt nhất** (ví dụ ArcFace) với 4 cấu hình augmentation khác nhau:

| Cấu hình | Mô tả | Mục đích |
|-----------|-------|----------|
| **A0** | Baseline (chỉ Resize + Normalize) | Đường cơ sở |
| **A1** | Augmentation hiện tại (code hiện có) | Hiện trạng |
| **A2** | A1 + GaussianBlur + Elastic + RandomErasing | Nâng cấp đề xuất |
| **A3** | A2 + CutMix/MixUp | Kỹ thuật nâng cao |

---

## 3. Ablation Study

> [!IMPORTANT]
> Ablation study **bắt buộc** cho Q1. Reviewer cần biết **mỗi thành phần đóng góp bao nhiêu % vào kết quả cuối cùng**.

### 3.1 Các thí nghiệm ablation cần chạy

#### A. Backbone Ablation

| Thí nghiệm | Backbone | Params | Mục đích |
|-------------|----------|--------|----------|
| B1 | ResNet-50 | 25M | Baseline cổ điển |
| B2 | EfficientNetV2-S | 22M | Kiến trúc hiệu quả |
| **B3** | **ConvNeXt-Tiny** | **29M** | **Hiện tại (default)** |
| B4 | Swin-Tiny | 28M | Transformer-based |
| B5 | DINOv2-ViT-S/14 | 22M | Foundation model |

Chạy tất cả backbone trên **cùng 1 loss function tốt nhất** (ví dụ ArcFace hoặc SupCon).

#### B. Embedding Dimension Ablation

| Thí nghiệm | Dim | Mục đích |
|-------------|-----|----------|
| D1 | 64 | Compact |
| D2 | 128 | Trung bình |
| **D3** | **256** | **Hiện tại** |
| D4 | 512 | Lớn |

#### C. Freeze Ratio Ablation

| Thí nghiệm | Freeze | Mục đích |
|-------------|--------|----------|
| F1 | 0% (full fine-tune) | Tất cả layers đều học |
| F2 | 50% | Nửa trên học |
| F3 | 75% | Chỉ 1/4 trên học |
| **F4** | **90%** | **Hiện tại** |
| F5 | 100% (linear probe) | Chỉ projection head học |

#### D. Data Split Ablation

| Thí nghiệm | Split Method | Mục đích |
|-------------|-------------|----------|
| S1 | Random Split (stratified) | Baseline |
| S2 | Subfolder-aware Split | Tách theo subfolder |
| **S3** | **End Version Split** | **Đề xuất** |

#### E. Loss Function Ablation (Top-5)

Chọn top-5 loss functions từ benchmark và phân tích sâu:

| Loss | Mục đích viết paper |
|------|---------------------|
| Triplet (Hard) | Baseline metric learning |
| ArcFace | SOTA angular margin |
| Multi-Similarity | SOTA mining-based |
| SupCon | SOTA contrastive |
| **Taxonomy-Aware** | **Đề xuất mới** |

### 3.2 Cách trình bày kết quả ablation

Mỗi bảng ablation cần có:
- **Bold** cho kết quả tốt nhất
- **Underline** cho kết quả tốt thứ 2
- Δ (delta) so với baseline
- ↑ / ↓ chỉ hướng tốt/xấu

Ví dụ:

```
Table 3: Ablation study on embedding dimension.
┌──────────┬─────────┬──────────┬────────────┬──────────────┐
│ Dim      │ mAP(%)  │ R@1(%)   │ Silhouette │ NMI          │
├──────────┼─────────┼──────────┼────────────┼──────────────┤
│ 64       │ 78.2    │ 82.1     │ 0.312      │ 0.756        │
│ 128      │ 83.5    │ 86.3     │ 0.378      │ 0.812        │
│ 256      │ **87.1**│ **89.7** │ **0.421**  │ **0.856**    │
│ 512      │ 86.8    │ 89.2     │ 0.415      │ 0.849        │
└──────────┴─────────┴──────────┴────────────┴──────────────┘
```

---

## 4. Statistical Significance

> [!WARNING]
> Paper Q1 **không chấp nhận** kết quả từ 1 lần chạy duy nhất. Cần ít nhất 3-5 runs với random seed khác nhau.

### 4.1 Multi-Run Protocol

```python
SEEDS = [42, 123, 456, 789, 1024]  # 5 random seeds

for seed in SEEDS:
    set_seed(seed)
    # Chạy toàn bộ pipeline: split → train → evaluate
    # Lưu kết quả: outputs_{method}_{seed}/

# Báo cáo: mean ± std cho mỗi metric
```

### 4.2 Statistical Tests cần chạy

**a) Paired t-test (so sánh 2 phương pháp):**

```python
from scipy import stats

# Ví dụ: so sánh Taxonomy-Aware vs ArcFace
taxonomy_maps = [87.1, 86.8, 87.5, 86.3, 87.9]  # mAP từ 5 seeds
arcface_maps  = [84.2, 85.1, 83.8, 84.7, 84.5]

t_stat, p_value = stats.ttest_rel(taxonomy_maps, arcface_maps)
# Nếu p < 0.05 → "significantly better" (có ý nghĩa thống kê)
```

**b) Friedman test + Nemenyi post-hoc (so sánh nhiều phương pháp):**

```python
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Mỗi hàng = 1 seed, mỗi cột = 1 method
results_matrix = np.array([
    [87.1, 84.2, 82.5, 85.3, 81.2],  # seed 42
    [86.8, 85.1, 83.1, 84.8, 80.9],  # seed 123
    # ... thêm các seed khác
])

# Friedman test: có khác biệt có ý nghĩa giữa các method không?
stat, p = friedmanchisquare(*results_matrix.T)

# Nemenyi post-hoc: cặp nào khác biệt cụ thể?
nemenyi = sp.posthoc_nemenyi_friedman(results_matrix)
```

**c) Critical Difference Diagram:**

Vẽ biểu đồ CD (Critical Difference) — chuẩn vàng cho so sánh nhiều phương pháp:

```python
# Sử dụng thư viện autorank
from autorank import autorank, plot_stats

result = autorank(df_results, alpha=0.05, verbose=False)
plot_stats(result)
# → Sinh biểu đồ Nemenyi CD diagram tự động
```

### 4.3 Cách trình bày kết quả

```
Table 2: Comparison of metric learning methods (mean ± std over 5 runs).
┌───────────────────┬───────────────┬───────────────┬──────────────────┐
│ Method            │ mAP (%)       │ R@1 (%)       │ NMI              │
├───────────────────┼───────────────┼───────────────┼──────────────────┤
│ Triplet           │ 78.2 ± 1.3    │ 82.1 ± 0.9    │ 0.756 ± 0.012    │
│ ArcFace           │ 84.5 ± 0.8    │ 87.3 ± 0.6    │ 0.821 ± 0.008    │
│ SupCon            │ 85.1 ± 0.7    │ 88.0 ± 0.5    │ 0.835 ± 0.007    │
│ Taxonomy (ours)   │ **87.3 ± 0.5**│ **89.8 ± 0.4**│ **0.856 ± 0.005**│
└───────────────────┴───────────────┴───────────────┴──────────────────┘
† indicates statistically significant improvement (p < 0.05, paired t-test)
```

---

## 5. Cross-Dataset Validation

### 5.1 Các dataset gỗ công khai khác

| Dataset | Classes | Images | Nguồn |
|---------|---------|--------|-------|
| **WOOD-AUTH** | 12 loài | ~3,000 | Aristotle University of Thessaloniki |
| **ForestSpecies** | 41 loài | ~2,942 | Brazilian Forest Service (SFB) |
| **UFPR Wood** | 112 loài (phổ biến dùng) | ~12,000 | Federal University of Paraná |
| **FIDS30** | 30 loài | ~3,000 | Kaggle public dataset |

### 5.2 Cách thực hiện cross-dataset validation

**Kịch bản 1 — Transfer Learning (ưu tiên):**
1. Train model trên S3 dataset (19 loài)
2. Đóng băng backbone, thay projection head
3. Fine-tune trên dataset mục tiêu (ví dụ UFPR Wood)
4. So sánh với training from scratch trên dataset mục tiêu

**Kịch bản 2 — Zero-shot Retrieval:**
1. Train model trên S3 dataset
2. Extract embeddings trên dataset mục tiêu (không fine-tune)
3. Đánh giá bằng KNN classifier (k=1, k=5)
4. → Chứng minh embedding space có tính tổng quát

**Kịch bản 3 — Domain Adaptation (nâng cao):**
1. Train trên S3 (source domain)
2. Dùng few-shot (5-shot, 10-shot) trên dataset mục tiêu (target domain)
3. So sánh với fully-supervised trên target domain

### 5.3 Nếu không có dataset khác

Nếu không tiếp cận được dataset gỗ khác, có thể test trên **dataset fine-grained classification tương tự**:

| Dataset | Classes | Tính tương đồng |
|---------|---------|------------------|
| **CUB-200-2011** | 200 loài chim | Fine-grained, texture-based |
| **Oxford Flowers-102** | 102 loài hoa | Fine-grained, visual similarity |
| **DTD (Describable Textures)** | 47 texture categories | Texture recognition = wood grain |

> [!TIP]
> **DTD** (Describable Textures Dataset) là lựa chọn tốt nhất cho cross-validation vì bài toán texture recognition rất gần với nhận dạng thớ gỗ.

---

## 6. Practical Deployment Analysis

### 6.1 Inference Benchmarking

```python
import time

model.eval()
dummy = torch.randn(1, 3, 224, 224).to(device)

# Warm-up
for _ in range(50):
    _ = model(dummy)

# Benchmark
torch.cuda.synchronize()
times = []
for _ in range(200):
    start = time.perf_counter()
    _ = model(dummy)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

avg_ms = np.mean(times) * 1000
fps = 1000.0 / avg_ms
print(f"Latency: {avg_ms:.2f} ms | FPS: {fps:.1f}")
```

### 6.2 Bảng so sánh cần có trong paper

```
Table 6: Efficiency comparison of backbone architectures.
┌──────────────────┬────────┬────────┬──────────┬─────────┬─────────┐
│ Backbone         │ Params │ FLOPs  │ Latency  │ mAP(%)  │ FPS     │
│                  │ (M)    │ (G)    │ (ms)     │         │ (GPU)   │
├──────────────────┼────────┼────────┼──────────┼─────────┼─────────┤
│ ResNet-50        │ 25.6   │ 4.1    │ 3.2      │ 81.3    │ 312.5   │
│ EfficientNetV2-S │ 21.5   │ 2.9    │ 4.1      │ 83.7    │ 243.9   │
│ ConvNeXt-Tiny    │ 28.6   │ 4.5    │ 3.8      │ 87.1    │ 263.2   │
│ Swin-Tiny        │ 28.3   │ 4.5    │ 5.2      │ 86.5    │ 192.3   │
│ DINOv2-ViT-S/14 │ 22.0   │ 5.5    │ 4.6      │ 88.2    │ 217.4   │
└──────────────────┴────────┴────────┴──────────┴─────────┴─────────┘
```

### 6.3 Few-Shot / Open-Set Evaluation

```python
# Kịch bản: có 1 loài gỗ mới chưa từng train
# → Chỉ cần 5 ảnh mẫu, không cần retrain

def few_shot_eval(model, support_set, query_set, k_shot=5):
    """
    support_set: k_shot ảnh của loài mới
    query_set: ảnh cần phân loại
    
    1. Extract embedding cho support_set → tính centroid
    2. Extract embedding cho query_set
    3. Phân loại bằng nearest centroid
    """
    support_embs = model(support_set)  # (k_shot, 256)
    centroid = support_embs.mean(dim=0)  # (256,)
    
    query_embs = model(query_set)  # (N, 256)
    distances = torch.cdist(query_embs.unsqueeze(0), centroid.unsqueeze(0).unsqueeze(0))
    
    return distances  # → so sánh với các centroid lớp cũ
```

**Thí nghiệm few-shot cần chạy:**

| k-shot | Accuracy mong đợi | Mục đích |
|--------|-------------------|----------|
| 1-shot | ~60-70% | Extreme low-resource |
| 5-shot | ~75-85% | Realistic deployment |
| 10-shot | ~80-90% | Comfortable deployment |
| 20-shot | ~85-92% | Near full-supervised |

---

## 7. So sánh với SOTA đã công bố

### 7.1 Các paper cần cite và so sánh

**Wood species identification:**

| Paper | Năm | Method | Dataset | Best Accuracy |
|-------|------|--------|---------|---------------|
| Hafemann et al. | 2014 | CNN features + SVM | ForestSpecies (41 sp.) | 95.7% |
| de Geus et al. | 2020 | Inception-ResNet | UFPR (112 sp.) | 93.2% |
| Lens et al. | 2020 | VGG-16 transfer | CITES-listed species | 88.4% |
| Hwang et al. | 2021 | EfficientNet-B4 | Korean wood (38 sp.) | 96.1% |
| Ravindran et al. | 2021 | CNN + computer vision | ForestSpecies | 92.6% |
| da Silva et al. | 2022 | ViT + attention | Brazilian woods | 94.8% |
| Figueroa-Mata et al. | 2022 | Xception, DenseNet | Costa Rican woods | 92.4% |

**Fine-grained classification (methodology):**

| Paper | Năm | Method | Relevance |
|-------|------|--------|-----------|
| Khosla et al. | 2020 | SupCon (NeurIPS) | Supervised contrastive baseline |
| Deng et al. | 2019 | ArcFace (CVPR) | Angular margin loss |
| Wang et al. | 2019 | Multi-Similarity (CVPR) | Smart mining |
| Sun et al. | 2020 | Circle Loss (CVPR) | Unified pair weighting |
| Zbontar & Bardes | 2021 | Barlow Twins (ICML) | Redundancy reduction SSL |
| Oquab et al. | 2023 | DINOv2 | Foundation model for fine-grained |

### 7.2 Cách so sánh công bằng

- Nếu dataset khác → báo cáo **relative improvement** trên dataset của bạn
- Nếu cùng dataset → so sánh trực tiếp (nhưng S3 có thể chưa ai dùng → bạn đặt benchmark đầu tiên)
- Luôn ghi rõ: backbone, input size, embedding dim, augmentation, epochs

---

## 8. Cấu trúc Paper đề xuất

```
Title: Taxonomy-Aware Metric Learning for Fine-Grained Wood 
       Species Identification: A Comprehensive Benchmark

Abstract (250 từ)

1. Introduction
   - Tầm quan trọng nhận dạng gỗ (CITES, chống buôn lậu)
   - Thách thức fine-grained: cùng chi rất giống nhau
   - Gaps trong nghiên cứu hiện tại
   - Đóng góp của paper (3 bullet points)

2. Related Work
   2.1 Wood Species Identification
   2.2 Fine-Grained Visual Recognition  
   2.3 Metric Learning

3. Methodology
   3.1 Dataset (S3) và Data Preprocessing
   3.2 Embedding-Aware Data Splitting (End Version Split)
   3.3 Taxonomy-Aware Hierarchical Loss
   3.4 Domain-Specific Augmentation
   3.5 Evaluation Protocol

4. Experiments
   4.1 Implementation Details
   4.2 Main Results: Benchmark 17+ Methods
   4.3 Ablation Studies
       - Backbone ablation
       - Embedding dimension
       - Freeze ratio
       - Data splitting
       - Augmentation strategies
   4.4 Taxonomy-Aware vs. Flat Loss Analysis
   4.5 Genus-Level Error Analysis (Grad-CAM + t-SNE)
   4.6 Few-Shot / Open-Set Evaluation
   4.7 Cross-Dataset Transfer

5. Discussion
   5.1 Why Taxonomy-Aware Margins Help
   5.2 Practical Deployment Considerations
   5.3 Limitations

6. Conclusion

Appendix:
   A. Full Benchmark Results (tất cả 17 methods)
   B. Per-Class Retrieval Reports
   C. Hyperparameter Sensitivity
   D. Grad-CAM Gallery
```

---

## 9. Journal mục tiêu

### Tier 1 — Q1 Computer Vision / Pattern Recognition

| Journal | IF | Review Time | Phù hợp |
|---------|-----|-------------|---------|
| **Pattern Recognition** | 8.0 | 3-6 tháng | ⭐⭐⭐⭐⭐ (fine-grained classification) |
| **IEEE TPAMI** | 23.6 | 6-12 tháng | ⭐⭐⭐ (rất khó, cần novelty cực mạnh) |
| **Computer Vision and Image Understanding** | 4.3 | 3-6 tháng | ⭐⭐⭐⭐ |
| **Neural Networks** | 7.8 | 2-4 tháng | ⭐⭐⭐⭐ |
| **Expert Systems with Applications** | 8.5 | 2-5 tháng | ⭐⭐⭐⭐⭐ (applied AI) |

### Tier 2 — Q1 Domain-Specific (Forestry / Wood Science)

| Journal | IF | Review Time | Phù hợp |
|---------|-----|-------------|---------|
| **Wood Science and Technology** | 3.4 | 3-6 tháng | ⭐⭐⭐⭐⭐ (domain exact) |
| **IAWA Journal** | 1.8 | 3-6 tháng | ⭐⭐⭐⭐⭐ (wood anatomy) |
| **Forest Ecology and Management** | 3.7 | 3-6 tháng | ⭐⭐⭐⭐ |
| **Trees - Structure and Function** | 2.7 | 2-4 tháng | ⭐⭐⭐⭐ |

### Tier 3 — Q1 Interdisciplinary

| Journal | IF | Review Time | Phù hợp |
|---------|-----|-------------|---------|
| **Scientific Reports** | 4.6 | 2-4 tháng | ⭐⭐⭐⭐ (dễ đăng Q1) |
| **Engineering Applications of AI** | 8.0 | 2-5 tháng | ⭐⭐⭐⭐⭐ |
| **Applied Soft Computing** | 8.7 | 2-5 tháng | ⭐⭐⭐⭐ |
| **Ecological Informatics** | 5.8 | 2-4 tháng | ⭐⭐⭐⭐ |

> [!TIP]
> **Lời khuyên:** Nếu target tốc độ + khả năng chấp nhận cao:
> - **Expert Systems with Applications** hoặc **Engineering Applications of AI** — IF cao, chấp nhận benchmark + applied method, review nhanh.
> - **Pattern Recognition** — Nếu novelty đủ mạnh (Taxonomy-Aware Loss).
> - **Wood Science and Technology** hoặc **IAWA Journal** — Nếu muốn đi theo hướng domain-specific, reviewer am hiểu wood anatomy sẽ đánh giá cao dataset contribution.

---

## Checklist Tổng Hợp

| # | Hạng mục | Chi tiết | Ước lượng thời gian |
|---|----------|----------|---------------------|
| 1 | Taxonomy-Aware Loss | Code loss + train + so sánh | 1-2 tuần |
| 2 | Multi-seed runs (5 seeds) | Chạy top-5 methods × 5 seeds = 25 runs | 2-3 tuần (GPU time) |
| 3 | Backbone ablation | 5 backbones × best loss × 3 seeds = 15 runs | 1-2 tuần |
| 4 | Embedding dim ablation | 4 dims × 3 seeds = 12 runs | 1 tuần |
| 5 | Augmentation ablation | 4 configs × 3 seeds = 12 runs | 1 tuần |
| 6 | Statistical tests | Code Friedman + Nemenyi + CD diagram | 2-3 ngày |
| 7 | Inference benchmarking | Measure FLOPs, latency, FPS | 1-2 ngày |
| 8 | Few-shot evaluation | 1-shot, 5-shot, 10-shot, 20-shot | 1 tuần |
| 9 | Cross-dataset (nếu có) | Transfer to UFPR Wood hoặc DTD | 1-2 tuần |
| 10 | Viết paper | Draft → revision → submission | 3-4 tuần |
| | **Tổng ước lượng** | | **~10-14 tuần** |
