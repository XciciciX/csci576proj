import cv2
import numpy as np

# 读取三张图片
img8 = cv2.imread('reconstructed_rank8.png')
img9 = cv2.imread('reconstructed_rank9.png')
img10 = cv2.imread('reconstructed_rank10.png')

print(f"Image 8 shape: {img8.shape}")
print(f"Image 9 shape: {img9.shape}")
print(f"Image 10 shape: {img10.shape}")

# 计算差异
diff_8_9 = cv2.absdiff(img8, img9)
diff_8_10 = cv2.absdiff(img8, img10)
diff_9_10 = cv2.absdiff(img9, img10)

# 统计不同像素的数量
num_diff_8_9 = np.count_nonzero(diff_8_9)
num_diff_8_10 = np.count_nonzero(diff_8_10)
num_diff_9_10 = np.count_nonzero(diff_9_10)

total_pixels = img8.shape[0] * img8.shape[1] * img8.shape[2]

print(f"\nDifference between rank 8 and 9: {num_diff_8_9}/{total_pixels} pixels ({100*num_diff_8_9/total_pixels:.2f}%)")
print(f"Difference between rank 8 and 10: {num_diff_8_10}/{total_pixels} pixels ({100*num_diff_8_10/total_pixels:.2f}%)")
print(f"Difference between rank 9 and 10: {num_diff_9_10}/{total_pixels} pixels ({100*num_diff_9_10/total_pixels:.2f}%)")

# 保存差异图
cv2.imwrite('diff_8_9.png', diff_8_9)
cv2.imwrite('diff_8_10.png', diff_8_10)
cv2.imwrite('diff_9_10.png', diff_9_10)

print("\nSaved difference images: diff_8_9.png, diff_8_10.png, diff_9_10.png")
