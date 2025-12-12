import cv2
import numpy as np
import math

import sys
import os
from puzzleSolver import PuzzleSolver



MIN_COMPONENT_AREA = 10    # 过滤太小的噪声连通域



def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def segment_pieces(image):
    """
    Segment individual puzzle pieces from a large image with a black background.
    Assumes the background is close to pure black, and puzzle pieces are colored.
    Returns: a list [piece_img, ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Simple threshold to separate foreground (puzzle pieces) from background
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    pieces = []
    for label in range(1, num_labels):  # 0 is the background
        x, y, w, h, area = stats[label]
        if area < MIN_COMPONENT_AREA:
            continue

        piece = image[y:y + h, x:x + w].copy()
        pieces.append(piece)

    print(f"[INFO] Detected {len(pieces)} pieces.")
    return pieces


#TODO: consider how to make rotate to translate
# The current version is blurry
def rectify_piece(piece_img, smooth=True):
    """
    使用透视变换将歪斜的方块矫正为矩形。
    先放大 2 倍以提高采样精度，变换后再缩回原尺寸。
    """
    gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return piece_img

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # 计算四条边长度
    pts = np.array(box, dtype="float32")
    edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    min_idx = np.argmin(edges)
    max_idx = np.argmax(edges)
    short_len = edges[min_idx]
    long_len = edges[max_idx]

    # 判断是否为规则矩形且无旋转（近似）
    h0, w0 = piece_img.shape[:2]
    # minAreaRect的角度为0或90且box与原图shape近似时，直接返回原图
    angle = rect[2]
    if (abs(short_len - h0) < 2 and abs(long_len - w0) < 2 and (abs(angle) < 1 or abs(abs(angle)-90)<1)) \
        or (abs(short_len - w0) < 2 and abs(long_len - h0) < 2 and (abs(angle) < 1 or abs(abs(angle)-90)<1)):
        # 没有旋转/透视，is_rotated=False
        return piece_img, False

    # 重新排列src点，使长边映射到目标长边，短边映射到目标短边
    start_idx = max_idx
    src_pts = np.array([pts[start_idx],
                        pts[(start_idx+1)%4],
                        pts[(start_idx+2)%4],
                        pts[(start_idx+3)%4]], dtype="float32")

    w, h = int(round(long_len)), int(round(short_len))
    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    except cv2.error:
        return piece_img, False

    big = cv2.resize(piece_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    M_scaled = M.copy()
    M_scaled[0, 2] *= 2
    M_scaled[1, 2] *= 2

    rectified = cv2.warpPerspective(
        big, M_scaled, (w * 2, h * 2),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    rectified = cv2.resize(rectified, (w, h), interpolation=cv2.INTER_AREA)

    if smooth:
        blur = cv2.GaussianBlur(rectified, (3, 3), sigmaX=1.0)
        rectified = cv2.addWeighted(rectified, 1.2, blur, -0.2, 0)

    # 保持shape不变，将内容放大一点点，舍弃边缘
    if rectified is not None:
        h, w = rectified.shape[:2]
        scale = 1.02  # 放大4%，可调整
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 先放大
        enlarged = cv2.resize(rectified, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # 再中心裁剪回原shape
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        rectified = enlarged[start_y:start_y + h, start_x:start_x + w]
    # 经过透视变换，is_rotated=True
    return rectified, True


def save_rectified_pieces(pieces, rectified_pieces, out_dir="debug_rectified"):
    os.makedirs(out_dir, exist_ok=True)

    for i, (orig, recti) in enumerate(zip(pieces, rectified_pieces)):
        # 原图（segment 出来的）
        orig_path = os.path.join(out_dir, f"{i:03d}_orig.png")
        cv2.imwrite(orig_path, orig)

        # 转正后的
        rect_path = os.path.join(out_dir, f"{i:03d}_rectified.png")
        cv2.imwrite(rect_path, recti)

    print(f"[INFO] Saved {len(rectified_pieces)} rectified pieces to: {out_dir}")



def main(input_path, output_image_path, output_anim_dir=None):
    img = load_image(input_path)
    pieces = segment_pieces(img)

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return
    

    # 获取rectified和is_rotated
    rectified_results = [rectify_piece(p) for p in pieces]
    rectified_pieces = [r[0] for r in rectified_results]
    is_rotated = any(r[1] for r in rectified_results)
    save_rectified_pieces(pieces, rectified_pieces, out_dir="debug_rectified")

    num_pieces = len(pieces)
    side = int(round(math.sqrt(num_pieces)))
    
    # 4, 4 in this case
    grid_rows = side
    grid_cols = side

    solver = PuzzleSolver(rectified_pieces, grid_rows, grid_cols, is_rotated)
    # best_layout, best_cost = solver.solve()
    img = solver.solve_packing_2()
    cv2.imwrite(output_image_path, img)
    print(f"[INFO] Saved solved puzzle image to {output_image_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python puzzle_solver.py <input_image> <output_image> [output_anim_dir]")
    else:
        input_path = sys.argv[1]
        output_image_path = sys.argv[2]
        output_anim_dir = sys.argv[3] if len(sys.argv) >= 4 else None
        main(input_path, output_image_path, output_anim_dir)
