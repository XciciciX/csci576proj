import cv2
import numpy as np

# 读取一张 reconstructed 图片
img = cv2.imread('reconstructed.png')

if img is None:
    print("Error: Could not read image")
    exit()

print(f"Image shape: {img.shape}")

# 统计不同像素值的分布
unique_values = []
for threshold in [0, 5, 10, 15, 20, 30, 50]:
    # 检查所有通道都小于threshold的像素
    dark_mask = np.all(img <= threshold, axis=2)
    dark_count = np.count_nonzero(dark_mask)
    dark_percent = 100 * dark_count / (img.shape[0] * img.shape[1])
    print(f"Pixels with all channels <= {threshold:3d}: {dark_count:8d} ({dark_percent:.2f}%)")

# 找出最暗的像素值
print("\nDarkest pixel values (first 20):")
flat_img = img.reshape(-1, 3)
sorted_by_brightness = flat_img[np.argsort(flat_img.sum(axis=1))]
for i in range(min(20, len(sorted_by_brightness))):
    print(f"  {i+1}: {sorted_by_brightness[i]}")

# 检查边缘附近的像素
print("\nSampling pixels at piece boundaries (assuming 4x4 grid):")
h, w = img.shape[:2]
piece_h = h // 4
piece_w = w // 4

# 检查每个piece边界处的像素
for row in range(4):
    for col in range(4):
        # 每个piece的边缘位置
        y = row * piece_h
        x = col * piece_w

        # 采样几个边缘像素
        if y > 0 and x > 0:
            pixel = img[y, x]
            if pixel.sum() < 30:  # 如果是暗色
                print(f"  Piece ({row},{col}) at ({y},{x}): {pixel}")
