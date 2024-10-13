import os
import cv2

DATA_DIR = './data'  # Thư mục chứa ảnh của từng nhãn

def adjust_brightness(image, factor):
    """Điều chỉnh độ sáng của ảnh."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def rotate_image(image, angle):
    """Xoay ảnh theo góc cho trước."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

# Duyệt qua từng nhãn trong thư mục
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue  # Bỏ qua nếu không phải thư mục nhãn

    # Kiểm tra số lượng ảnh trong thư mục nhãn
    num_images = len([f for f in os.listdir(label_dir) if f.endswith(('.jpg', '.png'))])
    if num_images >= 500:
        print(f"Skipping label {label} because it already contains {num_images} images.")
        continue  # Bỏ qua nếu số lượng ảnh >= 500

    # Duyệt qua từng ảnh trong thư mục của nhãn
    for filename in os.listdir(label_dir):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path)

            # Tăng sáng
            bright_img = adjust_brightness(img, factor=1.5)
            cv2.imwrite(os.path.join(label_dir, f"bright_{filename}"), bright_img)

            # Giảm sáng
            dark_img = adjust_brightness(img, factor=0.5)
            cv2.imwrite(os.path.join(label_dir, f"dark_{filename}"), dark_img)

            # Xoay theo chiều kim đồng hồ 15 độ
            rotated_cw = rotate_image(img, angle=-10)
            cv2.imwrite(os.path.join(label_dir, f"rotated_cw_{filename}"), rotated_cw)

            # Xoay ngược chiều kim đồng hồ 15 độ
            rotated_ccw = rotate_image(img, angle=10)
            cv2.imwrite(os.path.join(label_dir, f"rotated_ccw_{filename}"), rotated_ccw)

            print(f"Processed {filename} in label {label}.")
