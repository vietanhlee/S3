# BÁO CÁO HỌC THUẬT (Q1 JOURNAL REPORT SKELETON)

Tài liệu này bao gồm hai phần chính:
1. **Source Code LaTeX hoàn chỉnh (`main.tex`)** ở dạng khung sườn chuẩn học thuật hai cột (định dạng Overleaf/IEEEtran twocolumn) với các công thức toán học và cấu trúc bảng số liệu cực kỳ chi tiết, được tự động scale để vừa vặn với chiều rộng của cột.
2. **Nội dung bản thảo chi tiết bằng tiếng Việt** tương ứng với từng phần để bạn dễ dàng làm việc và thảo luận với giảng viên hướng dẫn.

---

## 1. Source Code LaTeX (`main.tex`)

Bạn có thể copy toàn bộ nội dung khối code dưới đây và dán thẳng vào trình soạn thảo Overleaf hoặc LaTeX compiler:

```latex
\documentclass[journal,twocolumn]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T5]{fontenc}
\usepackage[vietnamese,english]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{multirow}
\usepackage{array}

\begin{document}

\title{Class-wise Embedding-Guided Specimen Splitting for Leakage-Free Fine-Grained Wood Species Identification}

\author{Viet-Anh Le, and Collaborators}

\maketitle

\begin{abstract}
Nhận diện tự động loài gỗ là một nhiệm vụ quan trọng trong giám sát thương mại lâm sản, truy xuất nguồn gốc gỗ quý và hỗ trợ thực thi Công ước CITES trong bối cảnh khai thác gỗ bất hợp pháp vẫn là mối đe dọa lớn đối với đa dạng sinh học. Tuy nhiên, bài toán này có hai thách thức cốt lõi. Thứ nhất, ảnh mặt cắt gỗ thuộc nhóm fine-grained visual classification, trong đó các loài cùng chi có cấu trúc mạch, tia gỗ và nhu mô rất tương đồng, trong khi biến thiên nội bộ loài lại lớn do tuổi cây, vùng sinh trưởng, vị trí cắt và điều kiện chụp. Thứ hai, các quy trình đánh giá dùng random image-level split thường gây rò rỉ dữ liệu cấp mẫu vật, vì nhiều ảnh được cắt hoặc chụp từ cùng một mẫu vật lý nhưng bị phân tán đồng thời vào tập huấn luyện và tập kiểm thử. Điều này làm mô hình ghi nhớ vết xước, màu sắc, hướng cắt hoặc nhiễu cục bộ của mẫu thay vì học đặc trưng giải phẫu có khả năng tổng quát hóa.

Để giải quyết các hạn chế trên, nghiên cứu này đề xuất Class-wise Embedding-Guided Specimen Split (CEGS-Split), một giao thức chia dữ liệu theo từng loài nhằm cô lập mẫu vật lý và giảm rò rỉ ngữ nghĩa giữa train-test. CEGS-Split biểu diễn mỗi subfolder như một specimen group độc lập, trích xuất embedding bằng các backbone mạnh như EfficientNetV2 và Swin Transformer, sau đó chọn chiến lược phân hoạch phù hợp cho từng loài từ 4 chiến lược lõi: Mahalanobis-based split, hierarchical clustering split, stratified group split và agglomerative stratified split. Chúng tôi xây dựng benchmark trên bộ dữ liệu S3 gồm 19 loài thuộc 6 chi, so sánh 14 hàm loss metric learning và 4 thuật toán self-supervised learning trong cùng một pipeline đánh giá. Khung đề xuất tạo ra giao thức kiểm thử không rò rỉ, phản ánh thực tế hơn, đồng thời đánh giá không gian nhúng qua các chỉ số truy vấn và phân cụm như Recall@k, mAP, Silhouette, Davies-Bouldin, Dunn và NMI.
\end{abstract}

\begin{IEEEkeywords}
Fine-grained visual classification, Wood species identification, Metric learning, Data leakage, Deep learning, Self-supervised learning, CITES.
\end{IEEEkeywords}

\section{Introduction}

\subsection{Background}
Khai thác và buôn bán gỗ trái phép gây thiệt hại nghiêm trọng cho hệ sinh thái rừng, làm suy giảm đa dạng sinh học và tạo áp lực lên các loài gỗ thương mại có giá trị cao. Trong quản lý thương mại quốc tế, Công ước CITES yêu cầu nhận diện chính xác các loài được bảo vệ như \textit{Dalbergia tonkinensis}, \textit{Dalbergia cochinchinensis} hoặc các loài thuộc chi \textit{Afzelia}. Các phương pháp giám định truyền thống dựa vào chuyên gia giải phẫu gỗ quan sát cấu trúc vi mô dưới kính hiển vi, có độ tin cậy cao nhưng tốn thời gian, phụ thuộc mạnh vào kinh nghiệm chuyên gia và khó triển khai ở quy mô lớn tại cửa khẩu hoặc hiện trường.

Sự phát triển của computer vision và deep learning mở ra khả năng tự động hóa nhận diện gỗ từ ảnh mặt cắt macro hoặc macro-microscopic. Thay vì chỉ dựa vào mô tả thủ công của mạch gỗ, tia gỗ và nhu mô, mô hình học sâu có thể học biểu diễn phân biệt trực tiếp từ dữ liệu ảnh. Tuy nhiên, đối với ảnh gỗ, một pipeline học sâu chỉ thực sự có ý nghĩa nếu nó đánh giá được khả năng nhận diện trên mẫu vật lý chưa từng xuất hiện, chứ không chỉ trên những patch khác của cùng một khối gỗ.

\subsection{Problem Statement}
Nhận diện loài gỗ là bài toán fine-grained visual classification. Khoảng cách hình thái giữa các loài cùng chi thường rất nhỏ: các đặc trưng như phân bố lỗ mạch, dạng nhu mô quanh mạch, mật độ tia gỗ hoặc cấu trúc thớ có thể tương đồng đến mức khó phân biệt bằng mắt thường. Ngược lại, cùng một loài có thể có biến thiên rất lớn do gỗ lõi và gỗ giác, độ tuổi cây, điều kiện đất, độ ẩm, góc cắt và điều kiện chiếu sáng. Do đó, mô hình cần vừa học được đặc trưng ổn định của loài, vừa chịu được dịch chuyển phân phối ở cấp mẫu vật.

Về mặt dữ liệu, giả sử $\mathcal{X}=\{x_1,x_2,\dots,x_M\}$ là tập ảnh và $\mathcal{S}=\{S_1,S_2,\dots,S_N\}$ là tập mẫu vật lý, trong đó mỗi mẫu vật thường được lưu dưới dạng một subfolder. Một ảnh $x_i$ có nhãn loài $y_i$ và thuộc về đúng một mẫu vật $S(x_i)\in\mathcal{S}$. Nếu chia ngẫu nhiên ở cấp ảnh, các ảnh
\begin{equation}
\{x_{j,1},x_{j,2},\dots,x_{j,m}\}\in S_j
\end{equation}
có thể xuất hiện đồng thời trong train, validation và test. Khi đó, test set không còn độc lập về mặt mẫu vật, làm sai lệch kết quả đánh giá.

\subsection{Limitations of Prior Works}
Nhiều nghiên cứu nhận diện gỗ báo cáo độ chính xác rất cao khi dùng random split ở cấp ảnh. Tuy nhiên, trong các cơ sở dữ liệu gỗ, nhiều ảnh thường được chụp từ cùng một tiêu bản hoặc cắt ra từ cùng một block vật lý. CNN hoặc Vision Transformer có năng lực ghi nhớ mạnh, nên khi một mẫu vật xuất hiện đồng thời trong train và test, mô hình có thể khai thác các shortcut như màu nền, vết xước, vết dao cắt, độ bóng bề mặt hoặc nhiễu camera. Các dấu hiệu này không phải marker phân loại học, vì vậy kết quả kiểm thử trong phòng thí nghiệm có thể bị thổi phồng và giảm mạnh khi triển khai trên block gỗ mới.

Một hạn chế khác là phần lớn pipeline dùng cross-entropy classifier đơn thuần. Cách tiếp cận này tối ưu biên quyết định khép kín cho các lớp đã biết, nhưng chưa trực tiếp tổ chức không gian nhúng đặc trưng một cách tối ưu. Với dữ liệu gỗ, việc biểu diễn không gian nhúng thông qua deep metric learning hoặc học tự giám sát đóng vai trò quan trọng để đảm bảo khả năng truy xuất mở (open-set retrieval).

\subsection{Motivation and Proposed Solution}
Nghiên cứu này đặt mục tiêu xây dựng một giao thức nhận diện gỗ có khả năng tổng quát hóa trên mẫu vật mới. Trước hết, chúng tôi áp dụng nguyên tắc specimen-level isolation: toàn bộ ảnh thuộc cùng một subfolder chỉ được phép xuất hiện trong một tập duy nhất. Điều kiện rời rạc được phát biểu như sau:
\begin{equation}
\mathcal{S}_{\text{train}}\cap\mathcal{S}_{\text{val}}=\emptyset,\quad
\mathcal{S}_{\text{train}}\cap\mathcal{S}_{\text{test}}=\emptyset,\quad
\mathcal{S}_{\text{val}}\cap\mathcal{S}_{\text{test}}=\emptyset.
\end{equation}
Từ đó,
\begin{equation}
\mathcal{D}_{\text{train}}=\{x_i\in\mathcal{X}\mid S(x_i)\in\mathcal{S}_{\text{train}}\},
\end{equation}
\begin{equation}
\mathcal{D}_{\text{val}}=\{x_i\in\mathcal{X}\mid S(x_i)\in\mathcal{S}_{\text{val}}\},\quad
\mathcal{D}_{\text{test}}=\{x_i\in\mathcal{X}\mid S(x_i)\in\mathcal{S}_{\text{test}}\}.
\end{equation}

Trên nền tảng đó, chúng tôi đề xuất Class-wise Embedding-Guided Specimen Split (CEGS-Split), một phương pháp lai chọn chiến lược chia phù hợp theo từng loài dựa trên phân bố embedding. Cách tiếp cận này chuyển trọng tâm từ đánh giá ngẫu nhiên ở cấp ảnh sang đánh giá cô lập ở cấp mẫu vật, phù hợp hơn với retrieval, clustering và kiểm thử open-set trên các block gỗ chưa từng thấy.

\subsection{Key Contributions}
Các đóng góp chính của nghiên cứu gồm:
\begin{itemize}
    \item Đề xuất giao thức Class-wise Embedding-Guided Specimen Split (CEGS-Split) nhằm loại bỏ rò rỉ dữ liệu cấp mẫu vật bằng cách xem subfolder là đơn vị phân chia cơ sở và dùng embedding similarity để phát hiện các nhóm mẫu tương đồng.
    \item Xây dựng benchmark thống nhất cho 14 hàm loss metric learning và 4 phương pháp self-supervised learning trên bộ dữ liệu S3 gồm 19 loài thuộc 6 chi.
    \item Đánh giá mô hình bằng cả chỉ số retrieval và chỉ số hình học của không gian nhúng, bao gồm Recall@k, Precision@k, mAP, AUC, Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn và NMI.
\end{itemize}

\section{Related Work}

\subsection{Wood Species Identification}
Các phương pháp nhận diện gỗ truyền thống sử dụng đặc trưng thủ công như LBP, GLCM, histogram màu hoặc mô tả hình thái mạch gỗ. Giai đoạn gần đây, CNN như VGG, ResNet, EfficientNet, MobileNet và ConvNeXt được dùng rộng rãi để học đặc trưng trực tiếp từ ảnh. Các kiến trúc Transformer cũng cho thấy tiềm năng nhờ khả năng mô hình hóa quan hệ dài hạn trong texture. Dù vậy, nhiều nghiên cứu vẫn dùng random split ở cấp ảnh, chưa kiểm soát đầy đủ rò rỉ mẫu vật. Vì vậy, độ chính xác cao chưa chắc phản ánh khả năng nhận diện trên block gỗ mới.

\subsection{Deep Metric Learning}
Deep metric learning học một ánh xạ $f_\theta(x)$ sao cho các mẫu cùng lớp gần nhau và các mẫu khác lớp xa nhau trong không gian nhúng. Các loss cổ điển như Contrastive Loss và Triplet Loss tối ưu trực tiếp khoảng cách Euclid, nhưng thường phụ thuộc vào mining và có thể hội tụ chậm khi batch chứa nhiều cặp dễ. Các phương pháp hiện đại như Multi-Similarity, Circle Loss, ArcFace, SubCenter ArcFace, Proxy Anchor và SoftTriple đưa thêm trọng số động, margin góc hoặc proxy học được để làm biên phân tách sắc nét hơn. Đối với ảnh gỗ, DML đặc biệt phù hợp vì mục tiêu cuối cùng không chỉ là gán nhãn mà còn là truy vấn mẫu tương tự và kiểm tra cấu trúc cụm của các loài.

\subsection{Self-Supervised Representation Learning}
Self-supervised learning học biểu diễn từ các biến đổi của ảnh mà không phụ thuộc hoàn toàn vào nhãn thủ công. SimCLR tối ưu InfoNCE bằng mẫu âm, trong khi BYOL và SimSiam dùng predictor và stop-gradient để tránh collapse mà không cần negative pairs. Barlow Twins tối ưu ma trận tương quan chéo để vừa kéo hai view của cùng ảnh lại gần, vừa giảm dư thừa giữa các chiều đặc trưng. Với ảnh gỗ, SSL có khả năng học texture, hướng thớ và cấu trúc mạch bền vững trước biến đổi ảnh; tuy nhiên cần được đánh giá trong bối cảnh không rò rỉ cấp mẫu vật.

\section{Proposed Methodology}

\subsection{Overall Framework}
Pipeline đề xuất gồm sáu bước. Thứ nhất, dữ liệu S3 được chuẩn hóa và lọc nhãn không đủ định danh cấp loài. Thứ hai, các ảnh được nhóm theo subfolder đại diện cho mẫu vật lý. Thứ ba, embedding ban đầu được trích xuất bằng backbone mạnh như EfficientNetV2-M hoặc Swin-Large để mô tả phân bố ngữ nghĩa của từng mẫu vật. Thứ tư, CEGS-Split chọn chiến lược phân chia theo từng loài để tạo train, validation và test không rò rỉ. Thứ năm, ConvNeXt-Tiny được fine-tune với projection head phi tuyến để học embedding 256 chiều. Thứ sáu, mô hình được đánh giá bằng retrieval, clustering, t-SNE và Grad-CAM/Finer-CAM.

\subsection{Class-wise Embedding-Guided Specimen Split}
Với mỗi subfolder $S_j$, embedding đại diện được tính bằng centroid của các ảnh thuộc subfolder:
\begin{equation}
E_j=\frac{1}{|S_j|}\sum_{x_i\in S_j} e_i,
\end{equation}
trong đó $e_i\in\mathbb{R}^{D}$ là vector đặc trưng của ảnh $x_i$. Nếu số chiều lớn so với số subfolder, PCA được dùng để giảm chiều về $d'=\min(G-2,128)$ nhằm ổn định ước lượng hiệp phương sai và giảm curse of dimensionality.

Giao thức CEGS-Split hoạt động bằng cách quét qua một không gian các chiến lược phân chia tiềm năng (gồm các nhóm phương pháp chính: khoảng cách Mahalanobis, phân cụm phân cấp, phân nhóm có phân tầng và phân cụm phân tầng) để xác định thuật toán phù hợp nhất cho từng loài gỗ cụ thể. Trong đó, 4 chiến lược phân chia lõi được phát triển nhằm duy trì tính cô lập subfolder nghiêm ngặt ở các mức độ hình học và phân phối khác nhau. Phép phân chia ngẫu nhiên có phân tầng đóng vai trò baseline so sánh hiệu năng, còn các phương pháp còn lại đều cố gắng giữ tính cô lập mẫu vật lý ở các mức độ khác nhau nhằm triệt tiêu rò rỉ thông tin. CEGS-Split chọn phương pháp theo từng class thay vì ép toàn bộ dataset dùng một quy tắc duy nhất.

\subsubsection{Mahalanobis Distance-based Split}
Chiến lược này xem các mẫu vật xa centroid lớp là outlier ngữ nghĩa cần đưa vào tập kiểm thử hoặc kiểm định để stress-test khả năng tổng quát hóa. Với tập mẫu chưa phân bổ $U$, centroid và hiệp phương sai được tính bởi:
\begin{equation}
\mu_U=\frac{1}{|U|}\sum_{j\in U} z_j,
\end{equation}
\begin{equation}
\begin{aligned}
\Sigma_U = &\frac{1}{|U|-1}\sum_{j\in U}(z_j-\mu_U)(z_j-\mu_U)^T \\
&+\epsilon I.
\end{aligned}
\end{equation}
Khoảng cách Mahalanobis của mẫu $j$ đến centroid hiện tại là
\begin{equation}
\begin{aligned}
d_{\text{Mahal}}(z_j,\mu_U) = \big[ &(z_j-\mu_U)^T\Sigma_U^{-1} \\
&(z_j-\mu_U) \big]^{1/2}.
\end{aligned}
\end{equation}
Tại mỗi vòng lặp, mẫu xa nhất $j^*=\arg\max_{j\in U}d_{\text{Mahal}}(z_j,\mu_U)$ được đưa vào test hoặc validation theo mục tiêu kích thước, còn phần còn lại được dùng cho train.

\subsubsection{Hierarchical Clustering-based Split}
Chiến lược này phát hiện các cụm mẫu vật tự nhiên trong cùng một loài bằng Agglomerative Clustering với Ward's linkage. Số cụm $K^*$ được chọn bằng elbow method trên WCSS:
\begin{equation}
\text{WCSS}(K)=\sum_{k=1}^{K}\sum_{z_j\in\mathcal{C}_k}\|z_j-\mu_{\mathcal{C}_k}\|_2^2.
\end{equation}
Sau khi chuẩn hóa $K$ và $\text{WCSS}(K)$ về $[0,1]$, điểm khuỷu tay được chọn bởi
\begin{equation}
K^*=\arg\max_K \frac{|x_K+y_K-1|}{\sqrt{2}}.
\end{equation}
Mỗi cụm $\mathcal{C}_k$ được giữ nguyên khi phân bổ vào train, validation hoặc test. Điều này ngăn các nhóm texture gần nhau bị chia nhỏ qua nhiều tập, giảm nguy cơ cluster-level leakage.

\subsubsection{Stratified Group-based Split}
Stratified Group Split tối ưu đồng thời hai ràng buộc: cô lập group và cân bằng phân phối lớp. Với $w_{g,c}$ là số ảnh của lớp $c$ trong group $g$, mục tiêu phân phối ở fold $f$ là
\begin{equation}
T_{f,c}=\frac{1}{K}\sum_{g=1}^{G}w_{g,c}.
\end{equation}
Thuật toán gán tham lam chọn fold
\begin{equation}
f^*=\arg\min_f \sum_{c=1}^{C}\left(\frac{C_{f,c}+w_{g,c}}{T_{f,c}}-1\right)^2,
\end{equation}
trong đó $C_{f,c}$ là số mẫu hiện có của lớp $c$ trong fold $f$. Cách chia này phù hợp với các loài cần ưu tiên cân bằng lớp và ổn định gradient.

\subsubsection{Agglomerative Stratified Split}
Phương pháp này kết hợp phân cụm và phân tầng theo khoảng cách. Sau khi gom cụm, khoảng cách từ centroid cụm $\mu_{\mathcal{C}_k}$ đến centroid toàn lớp $\mu_{\text{global}}$ được tính bằng:
\begin{equation}
\begin{aligned}
d(\mathcal{C}_k) = \big[ &(\mu_{\mathcal{C}_k}-\mu_{\text{global}})^T\Sigma_{\text{global}}^{-1} \\
&(\mu_{\mathcal{C}_k}-\mu_{\text{global}}) \big]^{1/2}.
\end{aligned}
\end{equation}
Các cụm được chia thành ba dải Near, Mid và Far. Khi phân bổ, thuật toán chọn tập có độ hụt lớn nhất so với mục tiêu:
\begin{equation}
\Delta_s=\max(0,N_s^{\text{target}}-N_s^{\text{current}}),\quad s\in\{\text{train},\text{val},\text{test}\}.
\end{equation}
Nhờ đó, cả ba tập đều chứa mẫu điển hình, mẫu trung gian và mẫu dị biệt, thay vì dồn toàn bộ outlier vào test.

\subsection{CEGS-Split Configuration}
CEGS-Split là một cấu hình lai theo từng loài. Thay vì áp dụng một chiến lược duy nhất cho toàn bộ dataset, mỗi loài được gán phương pháp chia phù hợp với đặc điểm sinh học, số lượng subfolder và kết quả phân tích phân bố không gian biểu diễn. Cấu hình này được suy ra từ quy trình tìm kiếm tự động quét các chiến lược chia dữ liệu trên từng lớp, dùng EfficientNetV2-M để tạo embedding phục vụ thuật toán phân hoạch, dùng Swin-Large để đo nearest-neighbor similarity giữa train và test, rồi chọn phương pháp có mean nearest-neighbor similarity nhỏ nhất nhằm giảm thiểu tối đa nguy cơ rò rỉ ngữ nghĩa. Bảng \ref{tab:cegs_split} trình bày cấu hình chi tiết đang được triển khai thực tế.

\begin{table}[htbp]
\centering
\caption{Class-specific configuration of the CEGS-Split protocol}
\label{tab:cegs_split}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ll}
\toprule
\textbf{Species} & \textbf{Split Strategy} \\
\midrule
\textit{Afzelia africana} & Stratified Group \\
\textit{Afzelia bella} & Hierarchical Clustering \\
\textit{Afzelia pachyloba} & Agglomerative Stratified \\
\textit{Afzelia quanzensis} & Mahalanobis Iterative \\
\textit{Dalbergia cochinchinensis} & Agglomerative Stratified \\
\textit{Dalbergia melanoxylon} & Mahalanobis Iterative \\
\textit{Dalbergia oliveri} & Stratified Group \\
\textit{Dalbergia rimosa} & Hierarchical Clustering \\
\textit{Dalbergia tonkinensis} & Hierarchical Clustering \\
\textit{Guibourtia arnoldiana} & Hierarchical Clustering \\
\textit{Guibourtia coleosperma} & Agglomerative Stratified \\
\textit{Guibourtia ehie} & Hierarchical Clustering \\
\textit{Peltogyne pubescens} & Mahalanobis Iterative \\
\textit{Pterocarpus erinaceus} & Agglomerative Stratified \\
\textit{Pterocarpus indicus} & Agglomerative Stratified \\
\textit{Pterocarpus macrocarpus} & Hierarchical Clustering \\
\textit{Pterocarpus soyauxii} & Hierarchical Clustering \\
\textit{Sindora cochinchinensis} & Mahalanobis Iterative \\
\textit{Sindora tonkinensis} & Agglomerative Stratified \\
\bottomrule
\end{tabular}%
}
\end{table}

Với các loài có cấu hình hoán đổi (Swapped (Val)), kết quả chia tạm thời được điều chỉnh như sau:
\begin{equation}
\mathcal{D}_{\text{train}}=\mathcal{D}_{\text{train\_temp}},\quad
\mathcal{D}_{\text{val}}=\mathcal{D}_{\text{test\_temp}},\quad
\mathcal{D}_{\text{test}}=\mathcal{D}_{\text{val\_temp}}.
\end{equation}
Với các loài có cấu hình tiêu chuẩn (Standard (Test)), kết quả ban đầu được giữ nguyên. Cơ chế này giúp cân bằng độ khó giữa validation và test theo đặc thù từng loài, đồng thời vẫn duy trì cô lập subfolder nghiêm ngặt.

\subsection{Feature Backbone and Nonlinear Projection Head}
Mô hình nhận diện chính sử dụng ConvNeXt-Tiny làm backbone. Phần lớn các lớp đầu, khoảng $90\%$, được đóng băng để giữ lại đặc trưng tổng quát học từ ImageNet và giảm overfitting trên các lớp có số mẫu hạn chế. Đặc trưng thô $\mathbf{f}\in\mathbb{R}^{768}$ được đưa qua projection head phi tuyến để tạo embedding $\mathbf{z}\in\mathbb{R}^{256}$. Với một projection head hai tầng, biểu diễn chuẩn hóa $L_2$ là:
\begin{equation}
\tilde{\mathbf{z}}=W_2\text{ReLU}(W_1\mathbf{f}+b_1)+b_2,
\end{equation}
\begin{equation}
\mathbf{z}=\frac{\tilde{\mathbf{z}}}{\|\tilde{\mathbf{z}}\|_2}.
\end{equation}
Do $\|\mathbf{z}\|_2=1$, khoảng cách Euclid giữa hai embedding liên hệ trực tiếp với cosine similarity:
\begin{equation}
d(\mathbf{z}_a,\mathbf{z}_b)=\|\mathbf{z}_a-\mathbf{z}_b\|_2=\sqrt{2(1-\mathbf{z}_a^T\mathbf{z}_b)}.
\end{equation}

\subsection{Metric Learning Baselines and Loss Formulations}
Để đánh giá khách quan các hướng học biểu diễn, chúng tôi benchmark 14 hàm loss metric learning trong cùng một pipeline backbone, projection head, augmentation và CEGS-Split. Ký hiệu chung: $\mathbf{z}_i\in\mathbb{R}^{D}$ là embedding đã chuẩn hóa $L_2$, $y_i$ là nhãn loài, $d_{ij}=\|\mathbf{z}_i-\mathbf{z}_j\|_2$, $S_{ij}=\mathbf{z}_i^T\mathbf{z}_j$ là cosine similarity và $[u]_+=\max(0,u)$. Với ảnh mặt cắt gỗ, các loss này không chỉ được so sánh theo accuracy mà còn theo khả năng tạo không gian nhúng có cấu trúc phân cụm ổn định cho các loài cùng chi.

\subsubsection{Vanilla Triplet Loss}
Triplet Loss tối ưu quan hệ ba mẫu gồm anchor $a$, positive $p$ cùng loài và negative $n$ khác loài. Mục tiêu là làm khoảng cách anchor-positive nhỏ hơn anchor-negative ít nhất một biên $\alpha$:
\begin{equation}
L_{\text{triplet}}=\frac{1}{|\mathcal{T}|}\sum_{(a,p,n)\in\mathcal{T}}\left[d(a,p)^2-d(a,n)^2+\alpha\right]_+.
\end{equation}
Đây là baseline nền tảng để kiểm tra liệu mô hình có học được cấu trúc kéo-gần/đẩy-xa cơ bản hay không. Hạn chế của nó là số bộ ba hợp lệ tăng theo $O(B^3)$ và phần lớn triplet trong batch nhanh chóng trở thành triplet dễ, cho loss bằng 0. Trên dữ liệu S3, điều này đặc biệt bất lợi vì các loài cùng chi tạo ra ranh giới khó nhất nhưng số mẫu trong batch có thể không đủ đa dạng.

\subsubsection{Semi-hard Triplet Loss}
Semi-hard Triplet Loss chọn negative không quá dễ cũng không quá nhiễu. Với một cặp $(a,p)$, negative $n$ được chọn nếu:
\begin{equation}
d(a,p)^2<d(a,n)^2<d(a,p)^2+\alpha.
\end{equation}
Loss vẫn giữ dạng triplet chuẩn:
\begin{equation}
L_{\text{semi-hard}}=\frac{1}{|\mathcal{T}_{sh}|}\sum_{(a,p,n)\in\mathcal{T}_{sh}}\left[d(a,p)^2-d(a,n)^2+\alpha\right]_+.
\end{equation}
Cơ chế này tạo gradient ổn định hơn hardest mining vì negative vẫn nằm xa positive một chút, tránh việc mô hình bị kéo theo outlier hoặc nhãn nhiễu. Đối với ảnh gỗ, semi-hard mining hữu ích khi cần học biên giữa các loài gần nhau như các loài thuộc \textit{Dalbergia} nhưng không muốn tối ưu quá mạnh trên các mẫu bất thường.

\subsubsection{Soft-Margin Triplet Loss}
Soft-Margin Triplet thay hinge cứng bằng hàm log-sum-exp mịn:
\begin{equation}
L_{\text{soft-triplet}}=\frac{1}{N}\sum_{i=1}^{N}\log\left[1+\exp(d(a_i,p_i)^2-d(a_i,n_i)^2)\right].
\end{equation}
Khác với triplet loss có margin cố định, hàm này vẫn duy trì gradient nhỏ ngay cả khi negative đã tương đối xa. Điều này giảm độ nhạy với việc chọn $\alpha$, đồng thời làm quá trình hội tụ mượt hơn. Trong bối cảnh S3, soft-margin đóng vai trò cầu nối giữa triplet baseline và các loss hiện đại vì nó giúp mô hình tiếp tục tinh chỉnh cấu trúc embedding thay vì dừng học sớm ở các vùng đã vượt biên.

\subsubsection{Angular Loss}
Angular Loss tối ưu quan hệ góc của tam giác triplet thay vì chỉ tối ưu khoảng cách Euclid. Với triplet $(a,p,n)$, loss có dạng:
\begin{equation}
\begin{aligned}
L_{\text{angular}} = &\frac{1}{|\mathcal{T}|}\sum_{(a,p,n)\in\mathcal{T}}\log \Big[ 1 + \exp \big( 4\tan^2\alpha \\
&(\mathbf{z}_a+\mathbf{z}_p)^T\mathbf{z}_n - 2(1+\tan^2\alpha)\mathbf{z}_a^T\mathbf{z}_p \big) \Big].
\end{aligned}
\end{equation}
Vì embedding được chuẩn hóa, biên góc ít bị ảnh hưởng bởi scale của đặc trưng. Điều này phù hợp với ảnh macro gỗ khi cùng một cấu trúc thớ có thể thay đổi cường độ do ánh sáng, độ phóng đại hoặc độ ẩm bề mặt, nhưng quan hệ góc trong không gian biểu diễn vẫn giữ được thông tin phân biệt.

\subsubsection{Multi-Similarity Loss}
Multi-Similarity Loss khai thác đồng thời nhiều cặp positive và negative trong batch, đồng thời gán trọng số lớn hơn cho các cặp khó:
\begin{equation}
\begin{aligned}
L_{\text{MS}} = &\frac{1}{N}\sum_{i=1}^{N} \bigg\{ \frac{1}{\alpha}\log \Big[ 1+\sum_{p\in P_i}\exp \big( -\alpha(S_{ip}-\lambda) \big) \Big] \\
&+ \frac{1}{\beta}\log \Big[ 1+\sum_{n\in N_i}\exp \big( \beta(S_{in}-\lambda) \big) \Big] \bigg\}.
\end{aligned}
\end{equation}
Thành phần positive phạt các mẫu cùng loài còn chưa đủ gần; thành phần negative phạt các mẫu khác loài nhưng có cosine similarity cao. Đây là một trong các baseline quan trọng nhất cho S3 vì các loài khác nhau nhưng cùng chi thường tạo hard negative tự nhiên, chẳng hạn \textit{Dalbergia oliveri} và \textit{Dalbergia tonkinensis}.

\subsubsection{Vanilla Contrastive Loss}
Contrastive Loss tối ưu từng cặp mẫu. Gọi $q_{ij}=0$ nếu $y_i=y_j$ and $q_{ij}=1$ nếu $y_i\ne y_j$, loss được viết:
\begin{equation}
L_{\text{contrastive}}=(1-q_{ij})\frac{1}{2}d_{ij}^{2}+q_{ij}\frac{1}{2}[m-d_{ij}]_+^2.
\end{equation}
Loss kéo các cặp cùng loài về gần nhau và đẩy các cặp khác loài ra ngoài margin $m$. Ưu điểm là trực quan và dễ triển khai; hạn chế là cùng một margin được áp dụng cho mọi cặp âm, nên nó chưa phản ánh đầy đủ độ khó khác nhau giữa các loài trong cùng chi và khác chi.

\subsubsection{Online Hard Contrastive Loss}
Online Hard Contrastive tập trung vào positive xa nhất và negative gần nhất trong batch:
\begin{equation}
d_{\text{pos\_max}}^i=\max_{p:y_p=y_i,p\ne i}d_{ip},\quad d_{\text{neg\_min}}^i=\min_{n:y_n\ne y_i}d_{in}.
\end{equation}
\begin{equation}
L_{\text{hard-cont}}=\frac{1}{B}\sum_{i=1}^{B}\left[(d_{\text{pos\_max}}^i)^2+[m-d_{\text{neg\_min}}^i]_+^2\right].
\end{equation}
Cách này dùng tài nguyên gradient cho các ranh giới khó nhất thay vì các cặp dễ. Với ảnh gỗ, nó giúp tập trung vào các mẫu có vân gần giống nhau giữa hai loài, nhưng cũng nhạy cảm hơn với outlier, ảnh nhiễu hoặc lỗi nhãn.

\subsubsection{Lifted Structured Loss}
Lifted Structured Loss mở rộng contrastive learning bằng cách xét toàn bộ negative liên quan đến một cặp positive $(i,j)$:
\begin{equation}
\begin{aligned}
L_{\text{lifted}} = &\frac{1}{2|P|}\sum_{(i,j)\in P} \Big[ d_{ij} + \log \Big( \sum_{k:y_k\ne y_i}\exp(\alpha-d_{ik}) \\
&+ \sum_{k:y_k\ne y_j}\exp(\alpha-d_{jk}) \Big) \Big]_+^2.
\end{aligned}
\end{equation}
Hàm $\log\sum\exp$ là xấp xỉ trơn của phép lấy cực đại, do đó các negative gần nhất tự động nhận lực đẩy lớn hơn. Loss này hữu ích để kiểm tra lợi ích của tối ưu toàn batch đối với texture gỗ, nơi nhiều loài có một vài mẫu đặc biệt gần nhau trong embedding.

\subsubsection{Circle Loss}
Circle Loss đưa ra một dạng tối ưu similarity thống nhất, trong đó positive và negative có trọng số tự thích ứng:
\begin{equation}
\begin{aligned}
L_{\text{circle}} = \log \bigg[ &1 + \sum_{n\in N_i}\exp\big(\gamma\alpha_n(S_{in}-\Delta_n)\big) \\
&\sum_{p\in P_i}\exp\big(-\gamma\alpha_p(S_{ip}-\Delta_p)\big) \bigg].
\end{aligned}
\end{equation}
với
\begin{equation}
\alpha_p=[O_p-S_{ip}]_+,\quad \alpha_n=[S_{in}-O_n]_+.
\end{equation}
Positive đã đủ gần sẽ có trọng số nhỏ, còn negative quá gần sẽ bị đẩy mạnh. Cơ chế này phù hợp với fine-grained recognition vì nó tránh over-optimizing các loài dễ, đồng thời dồn gradient vào các cặp cùng chi có độ tương đồng cao.

\subsubsection{Proxy Anchor Loss}
Proxy Anchor thay so sánh ảnh-ảnh bằng so sánh ảnh với proxy lớp. Với proxy $p$ của một lớp, loss là:
\begin{equation}
\begin{aligned}
L_{\text{PA}} = &\frac{1}{|P^+|}\sum_{p\in P^+}\log \Big[ 1+\sum_{\mathbf{z}\in X_p^+}\exp \big( -\alpha(S(\mathbf{z},p)-\delta) \big) \Big] \\
&+\frac{1}{|P|}\sum_{p\in P}\log \Big[ 1+\sum_{\mathbf{z}\in X_p^-}\exp \big( \alpha(S(\mathbf{z},p)+\delta) \big) \Big].
\end{aligned}
\end{equation}
Do số lớp của S3 tương đối nhỏ, proxy-based learning có chi phí thấp và gradient ổn định. Proxy Anchor cũng giúp tránh phụ thuộc quá nhiều vào batch composition, một điểm quan trọng khi các loài gỗ không cân bằng số ảnh và số subfolder.

\subsubsection{ArcFace}
ArcFace áp dụng additive angular margin lên logit của lớp đúng:
\begin{equation}
\begin{aligned}
L_{\text{ArcFace}} = -\frac{1}{N}\sum_{i=1}^{N}\log \frac{\exp\big(s\cos(\theta_{y_i}+m)\big)}{\Lambda_i},
\end{aligned}
\end{equation}
where:
\begin{equation}
\begin{aligned}
\Lambda_i = \exp\big(s\cos(\theta_{y_i}+m)\big) + \sum_{j\ne y_i}\exp(s\cos\theta_j).
\end{aligned}
\end{equation}
Vì cả embedding và trọng số lớp đều được chuẩn hóa, ArcFace tạo biên geodesic trên mặt cầu đơn vị. Loss này thường mạnh trong fine-grained recognition vì nó ép cụm cùng lớp co lại trong một vùng góc hẹp và đẩy các lớp khác ra xa theo hướng góc, phù hợp với bài toán loài gỗ có khác biệt hình thái rất nhỏ.

\subsubsection{SubCenter ArcFace}
SubCenter ArcFace mở rộng ArcFace bằng nhiều tâm cho mỗi lớp. Với $K$ sub-centers, logit lớp đúng dùng tâm gần nhất:
\begin{equation}
\cos\theta_{y_i}=\max_{k=1}^{K}\frac{\mathbf{z}_i^T W_{y_i}^{(k)}}{\|\mathbf{z}_i\|_2\|W_{y_i}^{(k)}\|_2}.
\end{equation}
Sau đó áp dụng cùng angular margin như ArcFace. Cơ chế nhiều tâm rất phù hợp với S3 vì một loài có thể có phân bố đa đỉnh do gỗ lõi/gỗ giác, vùng chuyển tiếp, tuổi cây hoặc vị trí lát cắt. Thay vì ép tất cả biến thể vào một center duy nhất, SubCenter ArcFace cho phép mô hình học các mode texture riêng trong cùng loài.

\subsubsection{SoftTriple Loss}
SoftTriple cũng dùng nhiều center cho mỗi lớp nhưng dùng gán mềm thay vì lấy cực đại:
\begin{equation}
S_{i,c}=\sum_{k=1}^{K}\frac{\exp(\mathbf{z}_i^T w_c^{(k)}/\gamma)}{\sum_{t=1}^{K}\exp(\mathbf{z}_i^T w_c^{(t)}/\gamma)}\mathbf{z}_i^T w_c^{(k)}.
\end{equation}
Loss phân loại có margin:
\begin{equation}
L_{\text{SoftTriple}}=-\log\frac{\exp(\lambda(S_{i,y_i}-\delta))}{\exp(\lambda(S_{i,y_i}-\delta))+\sum_{c\ne y_i}\exp(\lambda S_{i,c})}.
\end{equation}
Soft assignment giúp gradient cập nhật nhiều center cùng lúc, giảm hiện tượng dead center. Với ảnh gỗ, SoftTriple có ý nghĩa vì sub-cluster texture có thể không tách rời cứng mà chuyển tiếp liên tục giữa các vùng gỗ.

\subsubsection{Supervised Contrastive Loss}
SupCon mở rộng SimCLR sang môi trường có nhãn bằng cách coi tất cả mẫu cùng lớp trong batch là positive:
\begin{equation}
L_{\text{SupCon}}=\sum_{i\in I}\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(\mathbf{z}_i^T\mathbf{z}_p/\tau)}{\sum_{a\in A(i)}\exp(\mathbf{z}_i^T\mathbf{z}_a/\tau)}.
\end{equation}
SupCon tối ưu căn chỉnh toàn batch và thường tạo embedding chuyển giao mạnh. Trong S3, nó giúp kéo nhiều ảnh cùng loài về gần nhau bất kể biến đổi chụp, nhưng vẫn có nguy cơ false-negative repulsion khi hai loài cùng chi có texture gần nhau bị đẩy quá mạnh.

\subsection{Self-Supervised Loss Formulations}
Trong học biểu diễn tự giám sát (Self-Supervised Learning - SSL), mô hình tối ưu hóa đặc trưng thông qua các biến đổi ngẫu nhiên của ảnh mà không sử dụng nhãn lớp giám sát. Phần này trình bày các hàm mất mát của 4 thuật toán SSL được khảo sát trong nghiên cứu.

\subsubsection{SimCLR Loss}
SimCLR (Simple Framework for Contrastive Learning of Visual Representations) tối ưu hóa hàm mất mát tương phản InfoNCE trên các cặp view được tăng cường. Với batch kích thước $N$, mỗi ảnh được tăng cường thành 2 view, tạo thành $2N$ mẫu. Cặp positive là $(i,j)$ gồm hai view của cùng một ảnh. Hàm loss cho cặp positive $(i,j)$ được định nghĩa là:
\begin{equation}
\begin{aligned}
\ell(i, j) = -\log \frac{\exp(\mathbf{z}_i^T\mathbf{z}_j/\tau)}{\Omega_i},
\end{aligned}
\end{equation}
where:
\begin{equation}
\begin{aligned}
\Omega_i = \sum_{k=1}^{2N} \mathbb{I}_{[k \ne i]} \exp(\mathbf{z}_i^T\mathbf{z}_k/\tau).
\end{aligned}
\end{equation}
trong đó $\tau$ là nhiệt độ (temperature hyperparameter). Hàm loss tổng thể là trung bình cộng của toàn bộ các cặp positive trong batch:
\begin{equation}
L_{\text{SimCLR}} = \frac{1}{2N} \sum_{k=1}^{N} \left[ \ell(2k-1, 2k) + \ell(2k, 2k-1) \right].
\end{equation}

\subsubsection{BYOL Loss}
BYOL (Bootstrap Your Own Latent) sử dụng hai mạng neural song song: mạng online (tham số $\theta$) và mạng target (tham số $\xi$). Mạng online dự đoán representation của mạng target từ một view khác. BYOL tối ưu hóa sai số bình phương trung bình (MSE) giữa vector dự đoán đã chuẩn hóa của mạng online $q_\theta(z_\theta)$ và vector biểu diễn của mạng target $z'_\xi$:
\begin{equation}
\begin{aligned}
L_{\text{MSE}}(q_\theta(z_\theta), z'_\xi) &= \left\| \frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|_2} - \frac{z'_\xi}{\|z'_\xi\|_2} \right\|_2^2 \\
&= 2 - 2 \frac{q_\theta(z_\theta)^T z'_\xi}{\|q_\theta(z_\theta)\|_2 \|z'_\xi\|_2}.
\end{aligned}
\end{equation}
Tránh hiện tượng sụp đổ biểu diễn (representation collapse) được đảm bảo bằng việc đặt cơ chế dừng gradient (\textit{stop-gradient}) trên mạng target:
\begin{equation}
L_{\text{BYOL}} = L_{\text{MSE}}(q_\theta(z_\theta), \text{stop\_gradient}(z'_\xi)).
\end{equation}
Tham số mạng target $\xi$ được cập nhật chậm theo cơ chế trung bình trượt lũy thừa (EMA) từ $\theta$:
\begin{equation}
\xi \leftarrow m \xi + (1-m) \theta,
\end{equation}
trong đó $m \in [0, 1]$ là hệ số momentum.

\subsubsection{SimSiam Loss}
SimSiam (Simple Siamese) đơn giản hóa BYOL bằng cách loại bỏ hoàn toàn mạng target momentum (cho $\xi = \theta$). SimSiam chứng minh rằng chỉ cần cơ chế stop-gradient đơn giản kết hợp với kiến trúc không đối xứng (predictor) là đủ để triệt tiêu sụp đổ biểu diễn. Hàm loss đối xứng được định nghĩa:
\begin{equation}
\begin{aligned}
L_{\text{SimSiam}} = &\frac{1}{2} \mathcal{D}(p_1, \text{stop\_gradient}(z_2)) \\
&+ \frac{1}{2} \mathcal{D}(p_2, \text{stop\_gradient}(z_1)),
\end{aligned}
\end{equation}
trong đó $p_1 = q_\theta(z_1)$ và $p_2 = q_\theta(z_2)$ là đầu ra của predictor cho hai view, còn $z_1, z_2$ là đầu ra của projection head. $\mathcal{D}$ là độ tương đồng cosine âm:
\begin{equation}
\mathcal{D}(p, z) = -\frac{p^T z}{\|p\|_2 \|z\|_2}.
\end{equation}

\subsubsection{Barlow Twins Loss}
Barlow Twins áp dụng nguyên lý giảm dư thừa (redundancy reduction) lên ma trận tương quan chéo của hai view. Gọi $C$ là ma trận tương quan chéo kích thước $D \times D$ tính dọc theo chiều batch $B$ giữa hai tập đặc trưng chiếu $\mathbf{z}^A$ và $\mathbf{z}^B$:
\begin{equation}
C_{ij} = \frac{\sum_{b=1}^{B} z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_{b=1}^{B} (z_{b,i)^2} \sqrt{\sum_{b=1}^{B} (z_{b,j}^B)^2}},
\end{equation}
trong đó $i, j$ đại diện cho các chiều của không gian đặc trưng nhúng. Hàm mất mát Barlow Twins được định nghĩa là:
\begin{equation}
L_{\text{BarlowTwins}} = \sum_{i=1}^{D} (1 - C_{ii})^2 + \lambda \sum_{i=1}^{D} \sum_{j \neq i} C_{ij}^2,
\end{equation}
trong đó số hạng thứ nhất (invariance term) bắt các đặc trưng tương quan chéo trên đường chéo chính tiến về 1 (tối đa hóa độ tương đồng của hai view), số hạng thứ hai (redundancy reduction term) ép các đặc trưng ngoài đường chéo chính về 0 với hệ số phạt $\lambda > 0$ (giảm thiểu thông tin trùng lặp giữa các chiều biểu diễn).

\section{Experimental Configuration}

\subsection{S3 Dataset Details}
Bộ dữ liệu S3 gồm ảnh mặt cắt ngang gỗ ở mức macro của 19 loài thuộc 6 chi: \textit{Afzelia}, \textit{Dalbergia}, \textit{Guibourtia}, \textit{Peltogyne}, \textit{Pterocarpus} và \textit{Sindora}. Dữ liệu có cấu trúc phân cấp tự nhiên: mỗi class tương ứng với một loài gỗ; bên trong mỗi class có nhiều subfolder; mỗi subfolder đại diện cho một mẫu vật lý hoặc một phiên chụp độc lập của cùng một block/tiêu bản gỗ. Các ảnh trong cùng subfolder thường là nhiều ảnh macro của cùng mặt cắt gỗ, được chụp hoặc cắt thành nhiều patch dưới điều kiện rất gần nhau về thiết bị, ánh sáng, bề mặt, hướng cắt và trạng thái chuẩn bị mẫu.

Cấu trúc này làm S3 khác với bộ dữ liệu ảnh tự nhiên thông thường. Nếu chia ngẫu nhiên ở cấp ảnh, các patch từ cùng một mẫu vật lý có thể xuất hiện đồng thời trong train và test, tạo điều kiện cho mô hình ghi nhớ đặc trưng riêng của phiên chụp như màu bề mặt, vết xước, nhiễu camera hoặc texture cục bộ lặp lại. Vì vậy, subfolder được xem là đơn vị group cơ sở trong toàn bộ nghiên cứu. Một split hợp lệ phải giữ nguyên mọi ảnh của cùng subfolder trong một tập duy nhất. Các mẫu chỉ định danh ở cấp chi, chẳng hạn \textit{Pterocarpus sp.}, được loại khỏi pipeline huấn luyện nhằm duy trì tính nhất quán của nhãn cấp loài và tránh gây nhiễu cho không gian embedding.

\subsection{Domain-Specific Data Augmentation}
Ảnh gỗ khác ảnh tự nhiên ở chỗ thông tin phân biệt nằm trong texture lặp, hướng thớ, cấu trúc mạch và màu sắc tự nhiên. Do đó, augmentation được thiết kế để bảo toàn marker giải phẫu nhưng làm mô hình bền vững trước điều kiện chụp. Các phép biến đổi phù hợp gồm rotation $0^\circ/90^\circ/180^\circ/270^\circ$, random rotation nhỏ, RandomResizedCrop, ColorJitter, RandomGrayscale, GaussianBlur, RandomAdjustSharpness và RandomErasing. Các biến đổi này mô phỏng khác biệt hướng cắt, tỷ lệ phóng đại, độ sáng, mất nét và vùng bề mặt bị che khuất.

\subsection{Benchmark Protocol}
Benchmark bao gồm 14 phương pháp metric learning: Vanilla Triplet, Semi-hard Triplet, Soft-Margin Triplet, Angular Loss, Multi-Similarity, Vanilla Contrastive, Hard Contrastive, Lifted Structured, Circle Loss, Proxy Anchor, ArcFace, SubCenter ArcFace, SoftTriple và SupCon. Nhóm self-supervised gồm SimCLR, BYOL, SimSiam và Barlow Twins. Tất cả phương pháp dùng cùng giao thức CEGS-Split để tránh so sánh lệch do khác biệt dữ liệu.

\subsection{Performance Evaluation Protocols}
Chất lượng biểu diễn được đánh giá bằng hai nhóm chỉ số. Nhóm retrieval gồm Recall@k, Precision@k, mAP và AUC. Với từng metric theo lớp $x_i$, báo cáo dùng cả macro average và harmonic mean có smoothing floor $\epsilon=0.01$:
\begin{equation}
\text{HM}=\frac{N}{\sum_{i=1}^{N}\frac{1}{x_i+\epsilon}},\quad \epsilon=0.01.
\end{equation}
Harmonic mean làm nổi bật các lớp yếu, tránh trường hợp mô hình đạt macro cao nhờ các lớp dễ nhưng thất bại ở các loài fine-grained khó.

Nhóm clustering gồm Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Dunn Index và Normalized Mutual Information. Với một mẫu $i$, Silhouette được tính:
\begin{equation}
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))},
\end{equation}
trong đó $a(i)$ là khoảng cách trung bình đến các mẫu cùng cụm và $b(i)$ là khoảng cách trung bình nhỏ nhất đến cụm khác. Davies-Bouldin Index là
\begin{equation}
\text{DBI}=\frac{1}{C}\sum_{i=1}^{C}\max_{j\ne i}\frac{\sigma_i+\sigma_j}{d(\mu_i,\mu_j)}.
\end{equation}
Dunn Index được định nghĩa:
\begin{equation}
\text{DI}=\frac{\min_{i\ne j}\delta(C_i,C_j)}{\max_k\Delta(C_k)}.
\end{equation}
NMI đo mức tương đồng thông tin giữa nhãn cụm dự đoán và nhãn loài thật:
\begin{equation}
\text{NMI}(Y,C)=\frac{2I(Y;C)}{H(Y)+H(C)}.
\end{equation}

\subsection{Implementation Hyperparameters}
Mô hình được tối ưu bằng AdamW với learning rate $10^{-4}$ và weight decay $10^{-2}$. Scheduler Cosine Annealing được dùng trong tối đa 100 epoch. Early stopping theo dõi harmonic mAP trên validation với patience 30 epoch. Embedding dimension mặc định là 256, freeze ratio mặc định là $90\%$, và backbone chính là ConvNeXt-Tiny. Các thí nghiệm ablation được thiết kế cho backbone, embedding dimension, freeze ratio, split method và loss function.

\section{Results and Discussion}

\subsection{Quantitative Evaluation}
Bảng \ref{tab:main_results} là khung báo cáo kết quả chính cho các phương pháp học biểu diễn. Khi điền số liệu thực nghiệm, mỗi hàng cần được báo cáo đồng thời theo macro và harmonic để phản ánh cả hiệu năng trung bình và độ công bằng giữa các loài khó.

\begin{table*}[htbp]
\centering
\caption{Performance Comparison of Representation Learning Methods on the Test Set (CEGS-Split)}
\label{tab:main_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccc}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{Recall@1 (\%)}} & \multicolumn{2}{c}{\textbf{Precision@1 (\%)}} & \multicolumn{2}{c}{\textbf{mAP (\%)}} & \multicolumn{5}{c}{\textbf{Clustering Metrics}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-12}
& Macro & Harmonic & Macro & Harmonic & Macro & Harmonic & Silhouette $\uparrow$ & DBI $\downarrow$ & CHI $\uparrow$ & Dunn $\uparrow$ & NMI $\uparrow$ \\
\midrule
\multicolumn{12}{l}{\textit{Classification Base}} \\
Focal Loss Baseline & - & - & - & - & - & - & - & - & - & - & - \\
\midrule
\multicolumn{12}{l}{\textit{Pairwise \& Triplet Metric Learning}} \\
Contrastive Loss & - & - & - & - & - & - & - & - & - & - & - \\
Hard Contrastive Loss & - & - & - & - & - & - & - & - & - & - & - \\
Triplet Loss & - & - & - & - & - & - & - & - & - & - & - \\
Semi-hard Triplet Loss & - & - & - & - & - & - & - & - & - & - & - \\
Soft-Margin Triplet & - & - & - & - & - & - & - & - & - & - & - \\
Lifted Structured Loss & - & - & - & - & - & - & - & - & - & - & - \\
Multi-Similarity Loss & - & - & - & - & - & - & - & - & - & - & - \\
Circle Loss & - & - & - & - & - & - & - & - & - & - & - \\
\midrule
\multicolumn{12}{l}{\textit{Angular Margin and Proxy-based Learning}} \\
Angular Loss & - & - & - & - & - & - & - & - & - & - & - \\
Proxy Anchor & - & - & - & - & - & - & - & - & - & - & - \\
ArcFace & - & - & - & - & - & - & - & - & - & - & - \\
SubCenter ArcFace & - & - & - & - & - & - & - & - & - & - & - \\
SoftTriple & - & - & - & - & - & - & - & - & - & - & - \\
\midrule
\multicolumn{12}{l}{\textit{Self-Supervised Learning}} \\
SupCon & - & - & - & - & - & - & - & - & - & - & - \\
SimCLR & - & - & - & - & - & - & - & - & - & - & - \\
BYOL & - & - & - & - & - & - & - & - & - & - & - \\
SimSiam & - & - & - & - & - & - & - & - & - & - & - \\
Barlow Twins & - & - & - & - & - & - & - & - & - & - & - \\
\midrule
\textbf{Proposed CEGS-Split (Ours)} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\subsection{Ablation on Data Splitting Impact}
Để định lượng tác động của rò rỉ dữ liệu, chúng tôi so sánh Random Split với CEGS-Split bằng một giao thức KNN trên embedding Swin-Large. Random Split được chạy trên ba seed $(42,123,456)$, trong khi CEGS-Split dùng cấu hình class-wise đã chọn. Các chỉ số gồm KNN test accuracy, F1-Macro, F1-Weighted, mean cosine similarity và max cosine similarity giữa train-test. Mean cosine similarity phản ánh mức gần trung bình giữa ảnh test và không gian train; max cosine similarity đặc biệt quan trọng vì nó phát hiện các test sample có láng giềng gần bất thường trong train, một dấu hiệu của near-duplicate hoặc rò rỉ ngữ nghĩa.

\begin{table}[htbp]
\centering
\caption{Data splitting ablation using KNN on Swin-Large embeddings}
\label{tab:split_ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Split Method} & \textbf{KNN Acc.} & \textbf{F1-Macro} & \textbf{F1-Weighted} & \textbf{Mean CosSim} & \textbf{Max CosSim} \\
\midrule
Random & $0.9581 \pm 0.0155$ & $0.9493 \pm 0.0179$ & $0.9571 \pm 0.0159$ & $0.5730 \pm 0.0037$ & $0.9161 \pm 0.0015$ \\
CEGS-Split & $0.9087 $ & $0.8858 $ & $0.9062 $ & $0.5711 $ & $0.8999 $ \\
\bottomrule
\end{tabular}%
}
\end{table}

Kết quả cho thấy Random Split đạt KNN accuracy trung bình $0.9581$, cao hơn CEGS-Split $0.9087$ một khoảng $0.0494$, tương đương mức overestimation $4.94\%$. F1-Macro cũng giảm từ $0.9493$ xuống $0.8858$ khi chuyển sang split cô lập mẫu vật, cho thấy sự sụt giảm không chỉ đến từ các lớp lớn mà còn ảnh hưởng rõ đến độ công bằng giữa các loài. Đồng thời, Random Split có mean cosine similarity cao hơn CEGS-Split $0.0019$ và max cosine similarity cao hơn $0.0162$ $(0.9161$ so với $0.8999)$. Điều này cho thấy test set của Random Split gần train hơn về mặt embedding, làm tăng khả năng mô hình hoặc KNN classifier hưởng lợi từ các mẫu cùng nguồn vật lý hoặc các patch gần trùng lặp. Vì vậy, CEGS-Split tạo ra đánh giá khó hơn nhưng trung thực hơn cho khả năng tổng quát hóa trên mẫu gỗ chưa từng thấy.

\subsection{Ablation on Model Components}
Các thí nghiệm ablation nên bao gồm: backbone ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny, Swin-Tiny và DINOv2-ViT-S/14; embedding dimension $64,128,256,512$; freeze ratio $0\%,50\%,75\%,90\%,100\%$; và nhóm loss gồm Triplet, ArcFace, Multi-Similarity, và SupCon. Cách trình bày nên dùng mean $\pm$ standard deviation trên nhiều seed và kiểm định thống kê cặp, chẳng hạn paired t-test hoặc Wilcoxon signed-rank test, để xác nhận cải thiện không chỉ là nhiễu thực nghiệm.

\section{Explainable AI and Embedding Analysis}

\subsection{t-SNE Clustering Visualization}
Phân tích t-SNE tập trung vào chi \textit{Afzelia}, gồm \textit{A. africana}, \textit{A. bella}, \textit{A. pachyloba} và \textit{A. quanzensis}. Đây là nhóm khó vì các loài có cấu trúc mạch gỗ và tia gỗ rất gần nhau. Ở baseline classification hoặc metric loss không dùng taxonomy, embedding của bốn loài có thể chồng lấn mạnh, tạo thành vùng chuyển tiếp mơ hồ. Sau khi tối ưu hóa không gian nhúng bằng metric learning, kỳ vọng là các mẫu vẫn giữ cấu trúc cấp chi nhưng tách thành bốn cụm loài rõ hơn, chứng minh vai trò của không gian nhúng biểu diễn đặc tính mịn.

\subsection{Attention Mapping via Grad-CAM}
Grad-CAM and Finer-CAM được dùng để kiểm tra mô hình dựa vào vùng ảnh nào khi phân biệt loài. Trước fine-tuning, backbone thường chú ý vào màu sắc chung, vùng nền hoặc artefact bề mặt. Sau khi tối ưu metric learning, bản đồ kích hoạt cần dịch chuyển về các marker giải phẫu như kích thước và phân bố lỗ mạch, mô hình nhu mô quanh mạch, tia gỗ và cấu trúc thớ. Nếu heatmap tập trung vào các vùng này, mô hình có tính diễn giải sinh học tốt hơn và ít phụ thuộc vào shortcut.

\section{Conclusion and Future Work}
Nghiên cứu này trình bày một khung nhận diện loài gỗ fine-grained kết hợp chia dữ liệu chống rò rỉ và học metric. CEGS-Split chuyển quy trình đánh giá từ random image-level split sang specimen-level isolation, giúp kết quả phản ánh khả năng tổng quát hóa trên mẫu vật lý chưa từng thấy.

Trong tương lai, nghiên cứu cần mở rộng sang cross-dataset validation với các bộ dữ liệu gỗ khác như UFPR, bổ sung kiểm định thống kê đa seed, đánh giá robustness với thay đổi thiết bị chụp và phát triển giao thức open-set để nhận biết loài chưa xuất hiện trong tập huấn luyện. Một hướng tiếp theo cũng quan trọng là tích hợp mô tả giải phẫu gỗ từ chuyên gia vào quá trình học, nhằm tăng tính diễn giải và độ tin cậy trong ứng dụng pháp y lâm nghiệp.

\begin{thebibliography}{00}
\bibitem{b1} CITES, ``Convention on International Trade in Endangered Species of Wild Fauna and Flora,'' 1973.
\bibitem{b2} A. L. Hafemann, L. S. Oliveira, and P. Cavalin, ``Forest species identification using deep convolutional neural networks,'' in \textit{Proceedings of the International Conference on Pattern Recognition (ICPR)}, 2014, pp. 1107--1112.
\bibitem{b3} J. Deng, J. Guo, N. Xue, and S. Zafeiriou, ``ArcFace: Additive Angular Margin Loss for Deep Face Recognition,'' in \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2019, pp. 4690--4699.
\bibitem{b4} Y. Wang, Q. Yao, and J. T. Kwok, ``Multi-similarity loss with general pair weighting for deep metric learning,'' in \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2019, pp. 5022--5030.
\bibitem{b5} P. Khosla et al., ``Supervised contrastive learning,'' \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol. 33, pp. 18661--18673, 2020.
\bibitem{b6} J. Zbontar et al., ``Barlow Twins: Self-supervised learning via redundancy reduction,'' in \textit{Proceedings of the International Conference on Machine Learning (ICML)}, 2021.
\bibitem{b7} Q. Qian et al., ``SoftTriple Loss: Deep Metric Learning Without Triplet Sampling,'' in \textit{Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, 2019.
\bibitem{b8} Y. Sun et al., ``Circle Loss: A unified perspective of pair similarity optimization,'' in \textit{CVPR}, 2020.
\end{thebibliography}

\end{document}
```
