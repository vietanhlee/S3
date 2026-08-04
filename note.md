Tôi cần bạn làm một file training mới lại và sử dụng  PP end_version mới (viết lại vào file training mới). Đại khái là vẫn như context ngữ nghĩa form như file train_split_comparision.py cũ những chỉ dùng cho một PP chuẩn cuối. Phương pháp chuẩn cuối này là sự kết hợp của nhiều PP cũ áp dụng cho từng loài gỗ (class). Cụ thể, tối ưu cho việc chia data, chủ yếu là việc chọn tập test cho từng class, ví dụ viết PP1 của val thì ta chia theo PP1 (tập train chia theo PP1, val lấy test còn test lấy của val PP1) cụ thể thì theo đánh giá tôi chọn như sau:

Afzelia africana: PP8 của val
Afzelia bella: PP4 của val
Afzelia pachyloba: PP2 của val
Afzelia quanzensis: PP7 của val 
Dalbergia cochinchinensis: PP4 của test
Dalbergia melanoxylon: PP4 của test
Dalbergia oliveri: PP5 của test
Dalbergia rimosa: PP4 của test
Dalbergia tonkinensis: PP4 của val
Guibourtia arnoldiana: PP4 của val 
Guibourtia coleosperma: PP4 của test
Guibourtia ehie: PP2 của val
Peltogyne pubescens: PP4 của val 
Pterocarpus erinaceus: PP2 của test 
Pterocarpus indicus: PP4 của val
Pterocarpus macrocarpus: PP9 của val
Pterocarpus soyauxii: PP4 của test
Pterocarpus sp: bỏ class này, không training class này nữa
Sindora cochinchinensis: PP9 của val 
Sindora tonkinensis: PP7 của test

1. Thêm các chỉ số đánh giá sau cho 2 file tripletloss vs constractive loss (thêm metric đó cho báo cáo cuối): Tỷ lệ Intra-class vs Inter-class Distance (in tỉ lệ này lên cái ảnh histogram ấy luôn nhé), Silhouette Score và các chỉ số sau (tự tìm hiểu và code cho cẩn thận):

1. Chỉ số Davies-Bouldin (Davies-Bouldin Index - DBI)

Ý nghĩa: DBI tính toán tỷ lệ giữa sự phân tán (độ phình to) của các cụm và khoảng cách giữa các tâm cụm (centroids) đó.

Tiêu chí: Giá trị càng thấp càng tốt (tối thiểu là 0).

Ứng dụng cho WoodID: DBI rất giỏi trong việc phát hiện "sự chồng lấn" (overlap). Nếu DBI của mô hình giảm mạnh sau khi áp dụng Contrastive Loss, điều đó chứng minh toán học rằng cụm chứa ảnh Gỗ Trắc đã co cụm lại và tách rời hoàn toàn khỏi cụm chứa ảnh Gỗ Hương.

2. Chỉ số Calinski-Harabasz (Calinski-Harabasz Index - CHI)

Ý nghĩa: Còn được gọi là Variance Ratio Criterion, CHI đánh giá độ tốt của không gian nhúng bằng cách tính tỷ lệ giữa tổng phương sai liên cụm (inter-cluster dispersion) và tổng phương sai nội cụm (intra-cluster dispersion).

Tiêu chí: Giá trị càng cao càng tốt. Khoảng cách giữa các cụm càng lớn và các điểm trong cụm càng đặc thì điểm CHI càng "bay".

Ứng dụng cho WoodID: Điểm này rất nhạy với các cụm hình cầu. Nếu Metric Learning của bạn thành công trong việc ép các feature vector của cùng một loài hội tụ về một điểm trung tâm, CHI sẽ phản ánh điều đó rất rõ rệt.

3. Chỉ số Dunn (Dunn Index)

Ý nghĩa: Đây là chỉ số đo lường tỷ lệ giữa khoảng cách nhỏ nhất giữa hai điểm thuộc hai cụm khác nhau (inter-cluster) và khoảng cách lớn nhất giữa hai điểm trong cùng một cụm (intra-cluster).

Tiêu chí: Giá trị càng cao càng tốt.

Ứng dụng cho WoodID: Dunn Index đặc biệt khắt khe đối với các điểm kỳ dị (outliers). Trong dữ liệu gỗ, sẽ có những bức ảnh macro bị nhiễu hoặc có cấu trúc bất thường. Nếu mô hình của bạn đẩy được các "ca khó" này về đúng cụm mà không làm giãn nở cụm đó ra quá to, Dunn Index sẽ cao.

4. Normalized Mutual Information (NMI)

Ý nghĩa: Mặc dù bài toán của bạn đã có nhãn (20 loài), NMI thường được dùng để đánh giá chất lượng của biểu diễn đặc trưng (feature representation) dưới góc độ phân cụm không giám sát (unsupervised clustering). Bạn lấy tập embedding chạy qua thuật toán K-Means (với $K=20$), sau đó so sánh kết quả phân cụm của K-Means với nhãn gốc.

Tiêu chí: Chạy từ 0 đến 1. Giá trị càng gần 1 càng tốt.

Ứng dụng cho WoodID: Rất nhiều paper top-tier về Metric Learning dùng NMI để chứng minh rằng: "Không gian embedding của tôi xịn đến mức, kể cả bỏ lớp phân loại đi và chỉ dùng K-Means thuần túy, nó vẫn gom nhóm chính xác các loài gỗ."



2. Báo cáo cuối của 2 file constractive loss với triplet loss cần đưa ra các metrics đánh giá đó cho từng loài (class) nữa chứ không tổng hợp chung như hiện tại được (giống như cái classification report ấy). Làm nó cho tập val và test luôn nhé (làm 1 bảng cho các class luôn, không cần riêng từng chi đâu)



Pairwise losses: 
Triplet-based: 
Proxy-based: 
Angular margin: 
Mining-based: Multi-
`Self-supervised: SimCLR, BYOL, SimSiam, Barlow Twins, SupCon`

{
  "Afzelia africana": [
    "PP9",
    "test",
    "swin"
  ],
  "Afzelia bella": [
    "PP5",
    "test",
    "swin"
  ],
  "Afzelia pachyloba": [
    "PP4",
    "val",
    "swin"
  ],
  "Afzelia quanzensis": [
    "PP9",
    "test",
    "swin"
  ],
  "Dalbergia cochinchinensis": [
    "PP4",
    "val",
    "swin"
  ],
  "Dalbergia melanoxylon": [
    "PP2",
    "val",
    "swin"
  ],
  "Dalbergia oliveri": [
    "PP7",
    "val",
    "swin"
  ],
  "Dalbergia rimosa": [
    "PP7",
    "val",
    "swin"
  ],
  "Dalbergia tonkinensis": [
    "PP4",
    "test",
    "swin"
  ],
  "Guibourtia arnoldiana": [
    "PP4",
    "test",
    "swin"
  ],
  "Guibourtia coleosperma": [
    "PP1",
    "test",
    "swin"
  ],
  "Guibourtia ehie": [
    "PP5",
    "test",
    "swin"
  ],
  "Peltogyne pubescens": [
    "PP1",
    "val",
    "swin"
  ],
  "Pterocarpus erinaceus": [
    "PP9",
    "test",
    "swin"
  ],
  "Pterocarpus indicus": [
    "PP9",
    "test",
    "swin"
  ],
  "Pterocarpus macrocarpus": [
    "PP7",
    "val",
    "swin"
  ],
  "Pterocarpus soyauxii": [
    "PP4",
    "test",
    "swin"
  ],
  "Sindora cochinchinensis": [
    "PP9",
    "test",
    "swin"
  ],
  "Sindora tonkinensis": [
    "PP7",
    "test",
    "swin"
  ]
}
}