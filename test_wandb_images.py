import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import wandb

# 1. Đọc file .env
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")

if api_key is None:
    raise ValueError("Không tìm thấy WANDB_API_KEY trong file .env. Vui lòng kiểm tra lại file .env tại thư mục gốc.")

# 2. Đăng nhập và khởi tạo WandB
wandb.login(key=api_key)
wandb.init(
    project="S3-Wood-Recognition-Test",
    name="test-image-logging",
)

# 3. Tạo một biểu đồ đơn giản bằng matplotlib làm ví dụ
print("Đang tạo biểu đồ matplotlib...")
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Sóng hình Sin", color="crimson", linewidth=2)
plt.title("WandB Image Logging Test - Antigravity")
plt.xlabel("Trục X")
plt.ylabel("Trục Y")
plt.legend()
plt.grid(True)

# 4. Đẩy trực tiếp hình ảnh biểu đồ lên WandB
print("Đang upload ảnh lên WandB...")
wandb.log({"test_sin_wave": wandb.Image(plt)})
plt.close()

# 5. Đóng tiến trình run
wandb.finish()
print("\n[Thành công] Đã hoàn tất! Hãy mở liên kết WandB hiển thị ở trên terminal để xem biểu đồ trong tab 'Media' hoặc 'Workspace'.")
