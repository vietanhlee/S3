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
