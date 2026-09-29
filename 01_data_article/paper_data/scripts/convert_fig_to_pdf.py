"""
Script chuyển đổi toàn bộ ảnh (PNG, JPG) trong thư mục fig sang định dạng PDF
Chuẩn publication cho bài báo khoa học (Lossless, High Resolution).

Sử dụng:
    python convert_fig_to_pdf.py
hoặc:
    python convert_fig_to_pdf.py --input_dir fig
"""

import os
import sys
import argparse
from pathlib import Path

def convert_images_to_pdf(fig_dir: Path):
    if not fig_dir.exists() or not fig_dir.is_dir():
        print(f"[ERROR] Thư mục không tồn tại: {fig_dir}")
        return

    # Danh sách các định dạng ảnh cần chuyển
    valid_extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted([
        f for f in fig_dir.iterdir() 
        if f.suffix.lower() in valid_extensions
    ])

    if not image_files:
        print(f"[INFO] Không tìm thấy file ảnh PNG/JPG nào trong thư mục: {fig_dir}")
        return

    print(f"============================================================")
    print(f" BẮT ĐẦU CHUYỂN ĐỔI ẢNH TRONG THƯ MỤC: {fig_dir.resolve()}")
    print(f" Tìm thấy {len(image_files)} file ảnh cần chuyển đổi sang PDF.")
    print(f"============================================================")

    # Thử import img2pdf (giải pháp lossless tốt nhất, không nén lại ảnh)
    has_img2pdf = False
    try:
        import img2pdf
        has_img2pdf = True
        print("[INFO] Đã phát hiện thư viện 'img2pdf': Chuyển đổi Lossless 100% (Bit-exact).")
    except ImportError:
        print("[INFO] Không tìm thấy 'img2pdf', tự động dùng thư viện 'Pillow' (PIL).")
        try:
            from PIL import Image
        except ImportError:
            print("[ERROR] Cả 'img2pdf' và 'Pillow' đều chưa được cài đặt!")
            print("Vui lòng cài đặt ít nhất một trong hai bằng lệnh:")
            print("    pip install img2pdf")
            print("  hoặc:")
            print("    pip install pillow")
            return

    success_count = 0
    for img_path in image_files:
        pdf_path = img_path.with_suffix(".pdf")
        src_size_kb = img_path.stat().st_size / 1024

        try:
            if has_img2pdf:
                with open(pdf_path, "wb") as f_out:
                    f_out.write(img2pdf.convert(str(img_path)))
            else:
                # Fallback dùng Pillow
                from PIL import Image
                with Image.open(img_path) as img:
                    # Chuyển RGBA sang RGB với nền trắng để tránh lỗi PDF
                    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        bg.paste(img, mask=img.split()[3])
                        bg.save(pdf_path, "PDF", resolution=300.0)
                    else:
                        img_rgb = img.convert("RGB")
                        img_rgb.save(pdf_path, "PDF", resolution=300.0)

            dst_size_kb = pdf_path.stat().st_size / 1024
            print(f" [OK] {img_path.name} ({src_size_kb:.1f} KB) -> {pdf_path.name} ({dst_size_kb:.1f} KB)")
            success_count += 1

        except Exception as e:
            print(f" [FAIL] Lỗi khi chuyển đổi {img_path.name}: {e}")

    print(f"============================================================")
    print(f" HOÀN TẤT: Đã chuyển đổi thành công {success_count}/{len(image_files)} file sang PDF.")
    print(f" Các file PDF mới đã được lưu trong: {fig_dir.resolve()}")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển toàn bộ ảnh trong thư mục fig sang định dạng PDF")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=str(Path(__file__).parent / "fig"),
        help="Đường dẫn đến thư mục chứa ảnh (mặc định: thư mục fig cùng cấp script)"
    )
    args = parser.parse_args()
    convert_images_to_pdf(Path(args.input_dir))
