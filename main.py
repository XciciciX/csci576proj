import cv2
import numpy as np
import math
from collections import namedtuple
import itertools
import sys
import os
from functools import reduce
from get_frame import solve_packing
from math import gcd
from typing import List, Tuple, Dict, Optional

from compute_similarity import compute_piece_edge_descriptors
from puzzleSolver import PuzzleSolver

# ---- 配置参数 ----
EDGE_STRIP_WIDTH = 10        # 用于提取边缘条带的宽度 (像素)
COLOR_BINS = 8              # HSV 每个通道的 bin 数
GRAD_BINS = 8               # 梯度方向直方图 bin 数
ALPHA = 0.5                 # 颜色差权重
BETAB = 0.5                 # 梯度差权重
MIN_COMPONENT_AREA = 10    # 过滤太小的噪声连通域

# Store info of a piece in a specific rotation
PieceRot = namedtuple("PieceRot", ["piece_idx", "img", "edges", "shape"])

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
    gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return piece_img

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # order four corners
    pts = np.array(box, dtype="float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype="float32")

    w, h = int(rect[1][0]), int(rect[1][1])

    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")

    # upscale sampling resolution
    big = cv2.resize(piece_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    M = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(
        big, M, (w*2, h*2),  # double resolution
        flags=cv2.INTER_CUBIC
    )

    # shrink back
    rectified = cv2.resize(rectified, (w, h), interpolation=cv2.INTER_AREA)

    # sharpening pass
    if smooth:
        blur = cv2.GaussianBlur(rectified, (0,0), sigmaX=1.1)
        rectified = cv2.addWeighted(rectified, 1.4, blur, -0.4, 0)

    return rectified


#TODO: change and consider irregular shapes


# ---------- 构建所有旋转版本 ----------

def build_all_rotations(pieces):
    """
    对每个 piece 生成 4 个旋转版本，并计算每个版本的 edge 描述子。
    返回：
        all_rots: (piece_idx, rot_idx) -> PieceRot
    """
    all_rots = []
    for i, p in enumerate(pieces):
        # for rot in range(4):
        #     img_rot = rotate_piece(p, rot)
        edges = compute_piece_edge_descriptors(p)
        all_rots.append(PieceRot(piece_idx=i, img=p, edges=edges, shape=p.shape[:2]))
    print(f"[INFO] Built {len(all_rots)} rotated versions.")
    return all_rots




def render_frame_layout(
    frame: List[Dict],
    piece_images: List[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    cell_h: int,
    cell_w: int,
) -> np.ndarray:
    """
    根据 solve_packing 的一个 frame 解，把真实的 piece 图片拼到一个大画布上。

    frame: solve_packing 返回的某个 solution（list[dict]）
    piece_images: 原始/rectified 的每块小图列表，第 i 个对应 piece_index = i
    canvas_h, canvas_w: 画布的网格尺寸（solve_packing 用的 H, W）
    cell_h, cell_w: 每个网格 cell 对应的像素高 / 像素宽（你用 gcd 算出来的）

    返回：
        一张大图 np.ndarray，尺寸约为 (canvas_h * cell_h, canvas_w * cell_w, 3)
    """
    H = canvas_h * cell_h
    W = canvas_w * cell_w
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for slot in frame:
        idx   = slot["piece_index"]
        top   = slot["top"]
        left  = slot["left"]
        h_g   = slot["height"]   # grid 单位高度
        w_g   = slot["width"]    # grid 单位宽度

        # 目标像素尺寸 = grid 尺寸 * 每个 cell 的像素尺寸
        target_h = h_g * cell_h
        target_w = w_g * cell_w

        piece_img = piece_images[idx]

        # resize 到对应的矩形区域大小
        piece_resized = cv2.resize(
            piece_img,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

        y0 = top * cell_h
        x0 = left * cell_w

        canvas[y0:y0 + target_h, x0:x0 + target_w, :] = piece_resized

    return canvas


def render_layout(best_layout, all_rots, piece_size):
    """
    根据 best_layout 把拼图拼好。
    best_layout: 2D: (piece_idx, rot_idx)
    piece_size: (h, w)
    """
    rows = len(best_layout)
    cols = len(best_layout[0])
    ph, pw = piece_size

    canvas = np.zeros((rows * ph, cols * pw, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]
            piece_rot = all_rots[piece_idx]
            img = rotate_piece(piece_rot.img, rot_idx)

            # 确保大小相同
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            y0 = r * ph
            x0 = c * pw
            canvas[y0:y0 + ph, x0:x0 + pw, :] = img

    return canvas


def rotate_piece(img, rot_idx):
    """
    rot_idx = 0,1,2,3 分别对应 0°,270°,180°,90°
    顺时针旋转
    """
    return np.rot90(img, rot_idx, axes=(0, 1))

def render_animation_sequence(best_layout, all_rots, piece_size, bg_color=(0, 0, 0)):
    """
    简单制作一个“拼图动画序列”：一帧一块地放上去。
    返回：frames 列表，每个是一张图。
    """
    rows = len(best_layout)
    cols = len(best_layout[0])
    ph, pw = piece_size

    h = rows * ph
    w = cols * pw

    frames = []
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # 一块块放
    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]
            piece_rot = all_rots[(piece_idx, rot_idx)]
            img = cv2.resize(piece_rot.img, (pw, ph), interpolation=cv2.INTER_AREA)
            y0 = r * ph
            x0 = c * pw
            canvas[y0:y0 + ph, x0:x0 + pw, :] = img
            frames.append(canvas.copy())

    return frames

# ---------- 用一张 input 图 → 做 frame packing ----------

def _gcd_list(nums: List[int]) -> int:
    return reduce(gcd, nums)

# ---------- 主入口 ----------

def main(input_path, output_image_path, output_anim_dir=None):
    img = load_image(input_path)
    pieces = segment_pieces(img)

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return
    # piece_size = pieces[0].shape[:2]  # 假设所有 piece 大小相同
    # rectified_pieces = [rectify_piece(p) for p in pieces]
    
    # 2. 只取像素尺寸 (h, w)
    piece_sizes_px = [(p.shape[0], p.shape[1]) for p in pieces]
    print("[INFO] piece_sizes_px =", piece_sizes_px)

    hs = [h for h, w in piece_sizes_px]
    ws = [w for h, w in piece_sizes_px]

    # 用 gcd 把像素转成“格子单位”
    gcd_h = _gcd_list(hs)
    gcd_w = _gcd_list(ws)

    canvas_h_pixels, canvas_w_pixels = 400, 400

    # 确保画布像素维度能被 gcd 整除，否则退化到 cell=1 像素
    if canvas_h_pixels % gcd_h != 0:
        print(f"[WARN] canvas_h_pixels {canvas_h_pixels} 不能被 gcd_h {gcd_h} 整除，使用 cell_h=1")
        gcd_h = 1
    if canvas_w_pixels % gcd_w != 0:
        print(f"[WARN] canvas_w_pixels {canvas_w_pixels} 不能被 gcd_w {gcd_w} 整除，使用 cell_w=1")
        gcd_w = 1

    canvas_h = canvas_h_pixels // gcd_h
    canvas_w = canvas_w_pixels // gcd_w
    piece_sizes_grid = [(h // gcd_h, w // gcd_w) for (h, w) in piece_sizes_px]

    print(f"[INFO] cell size = ({gcd_h}, {gcd_w}) pixels")
    print(f"[INFO] canvas grid size = ({canvas_h}, {canvas_w})")
    print(f"[INFO] piece sizes in grid = {piece_sizes_grid}")
    
    total_piece_area_grid = sum(h * w for h, w in piece_sizes_grid)
    canvas_area_grid = canvas_h * canvas_w
    print(f"[INFO] total_piece_area_grid = {total_piece_area_grid}, "
          f"canvas_area_grid = {canvas_area_grid}")
    
     # 3. solve_packing 纯几何拼板（只看 frame）
    print("[INFO] Running solve_packing (frame only)...")


    

    # num_pieces = len(rectified_pieces)
    # side = int(round(math.sqrt(num_pieces)))
    
    # grid_rows = side
    # grid_cols = side

    # all_rots = build_all_rotations(rectified_pieces)

    # solver = PuzzleSolver(all_rots, grid_rows, grid_cols)
    # best_layout, best_cost = solver.solve()

    # if best_layout is None:
    #     print("[ERROR] No layout found.")
    #     return
    num_pieces = len(pieces)
    side = int(round(math.sqrt(num_pieces)))
    
    grid_rows = side
    grid_cols = side

    all_rots = build_all_rotations(pieces)

    solver = PuzzleSolver(all_rots, grid_rows, grid_cols)
    solutions = solver.irr_solve(canvas_h, canvas_w, piece_sizes_grid)
    print(f"[INFO] Found {len(solutions)} geometric layout(s).")

    for i, sol in enumerate(solutions):
        frame_image = render_frame_layout(
            sol,
            pieces,
            canvas_h,
            canvas_w,
            gcd_h,
            gcd_w
        )
        frame_path = f"{output_image_path}_frame_{i:02d}.png"
        cv2.imwrite(frame_path, frame_image)
        print(f"[INFO] Saved frame layout image to {frame_path}")
    

    # 默认假设是正方形布局：rows = cols = sqrt(N)

    # TODO: change this

    # "The image sizes will be same as the test samples you have"
    # 
    # 找出所有 (r, c)，使得 r * c == num_pieces

    # 对每个 (r, c) 都跑一次 PuzzleSolver 选 cost 最小的那个

   
    # print(f"[INFO] Best layout cost: {best_cost:.4f}")

    # solved_image = render_layout(best_layout, all_rots, piece_size)
    # cv2.imwrite(output_image_path, solved_image)
    # print(f"[INFO] Saved solved puzzle image to {output_image_path}")

    # if output_anim_dir is not None:
    #     os.makedirs(output_anim_dir, exist_ok=True)
    #     frames = render_animation_sequence(best_layout, all_rots, piece_size)
    #     for i, frame in enumerate(frames):
    #         frame_path = os.path.join(output_anim_dir, f"frame_{i:03d}.png")
    #         cv2.imwrite(frame_path, frame)
    #     print(f"[INFO] Saved {len(frames)} animation frames to {output_anim_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python puzzle_solver.py <input_image> <output_image> [output_anim_dir]")
    else:
        input_path = sys.argv[1]
        output_image_path = sys.argv[2]
        output_anim_dir = sys.argv[3] if len(sys.argv) >= 4 else None
        main(input_path, output_image_path, output_anim_dir)
