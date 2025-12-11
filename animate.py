# ============================================================
# main_solve_and_animate.py 
#
# 使用方式：
#   python main_solve_and_animate.py starry_night_rotate.png solved.png anim.gif
#   （需保證目前資料夾有 starry_night.png 作為原圖提示）
#
# 流程：
#   1. 從輸入黑底散落圖中偵測 16 塊拼圖 (connected components)
#   2. 用 minAreaRect 校正每一塊為正方形 tile (TILE_SIZE x TILE_SIZE)
#   3. 對「ref tile」與「所有 fragment tile 的 4 種旋轉」抽取特徵：
#        - RGB Histogram (16 bins * 3 channel)
#        - HOG
#        - LBP histogram
#   4. 建立 cost_matrix[i][j] = ref_i 和 fragment_j(最佳旋轉) 的距離
#   5. 用 bitmask DP 解 assignment（等價 Hungarian，n=16 也很快）
#   6. 依照 assignment + rotation 組回 4x4 拼圖 → saved as solved.png
#   7. 用 puzzle_anim.make_puzzle_animation 產生動畫，初始為散落狀態
#
# 依賴：
#   pip install opencv-python numpy scikit-image
#   puzzle_anim.py 需於同資料夾並提供 make_puzzle_animation(...)
# ============================================================

import cv2
import numpy as np
import math
import sys
import random
from pathlib import Path

from skimage.feature import hog, local_binary_pattern

from puzzle_anim import make_puzzle_animation


# --------------------- 全域參數調整區 ------------------------
MIN_COMPONENT_AREA = 800       # 連通域面積閾值：太小視為雜訊
TILE_SIZE          = 128       # 校正後 tile 的邊長
EDGE_STRIP_WIDTH   = 28        # 雖然這版不再用到 strip，但保留以防調整

RANDOM_SEED        = 123       # 固定 seed 方便重現
# ------------------------------------------------------------


# ============================================================
# 基礎工具：載入、分割、校正
# ============================================================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[ERROR] Cannot load image: {path}")
    return img


def segment_tiles(img: np.ndarray):
    """
    利用黑背景做 connected components，切出每塊 tile。
    回傳：list[(tile_img, center_x, center_y)]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        th, connectivity=8
    )

    tiles = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < MIN_COMPONENT_AREA:
            continue
        crop = img[y:y+h, x:x+w].copy()
        cx, cy = centroids[i]
        tiles.append((crop, float(cx), float(cy)))

    print(f"[INFO] Detected {len(tiles)} tiles in fragment image.")
    return tiles


def rectify_tile(tile: np.ndarray) -> np.ndarray:
    """
    用 minAreaRect 把 tile 校正成水平矩形，再縮放成固定大小正方形。
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(tile, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)

    w, h = rect[1]
    w = int(max(w, 1))
    h = int(max(h, 1))

    # 排序成 TL, TR, BR, BL
    s = box.sum(axis=1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tile, M, (w, h))

    warped = cv2.resize(warped, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
    return warped


# ============================================================
# 原圖切割：Starry Night → 4x4 參考 tile
# ============================================================

def cut_reference_tiles(ref_img: np.ndarray, N: int = 4):
    """
    將原圖 resize 為 (N*TILE_SIZE, N*TILE_SIZE)，再切成 N x N tile。
    """
    H, W, _ = ref_img.shape
    target_size = (N * TILE_SIZE, N * TILE_SIZE)
    ref_resized = cv2.resize(ref_img, target_size, interpolation=cv2.INTER_AREA)

    tiles = []
    h_step = TILE_SIZE
    w_step = TILE_SIZE
    for r in range(N):
        for c in range(N):
            y0 = r * h_step
            x0 = c * w_step
            crop = ref_resized[y0:y0+h_step, x0:x0+w_step].copy()
            tiles.append(crop)
    print(f"[INFO] Cut reference image into {len(tiles)} tiles ({N}x{N}).")
    return tiles


# ============================================================
# Tile 特徵：RGB Hist + HOG + LBP
# ============================================================

def tile_feature_advanced(tile: np.ndarray) -> np.ndarray:
    """
    對整塊 tile 抽特徵：
      - RGB histogram (16 bins * 3 = 48 維)
      - HOG descriptor (~36 維，視圖像尺寸略有變化)
      - LBP histogram (16 bins)
    """
    # --- RGB Histogram ---
    hist_list = []
    for ch in range(3):
        h = cv2.calcHist([tile], [ch], None, [16], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist_list.append(h)
    hist_rgb = np.concatenate(hist_list)  # (48,)

    # --- HOG ---
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    ph = max(1, tile.shape[0] // 4)
    pw = max(1, tile.shape[1] // 4)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(ph, pw),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    ).astype(np.float32)
    if hog_feat.shape[0] > 40:
        hog_feat = hog_feat[:40]

    # --- LBP ---
    lbp = local_binary_pattern(gray, P=8, R=2, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 16))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    feat = np.concatenate([hist_rgb, hog_feat, lbp_hist]).astype(np.float32)
    return feat



def tile_pixel_cost(ref_tile, frag_tile):
    """
    直接用像素（在 Lab 色彩空間）算兩塊的 MSE。
    為了穩定，先縮小到 64x64。
    """
    ref = cv2.resize(ref_tile, (64, 64), interpolation=cv2.INTER_AREA)
    frag = cv2.resize(frag_tile, (64, 64), interpolation=cv2.INTER_AREA)

    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab).astype(np.float32)
    frag_lab = cv2.cvtColor(frag, cv2.COLOR_BGR2Lab).astype(np.float32)

    diff = ref_lab - frag_lab
    mse = np.mean(diff * diff)
    return float(mse)

# ============================================================
# 建立 cost matrix + 最佳旋轉
# ============================================================

def build_cost_matrix_with_rot(ref_tiles, frag_tiles):
    """
    ref_tiles : list of N^2 reference tiles (BGR)
    frag_tiles: list of N^2 fragment tiles (BGR, 已校正但尚未旋轉回正)

    回傳：
      cost_matrix[i][j] = ref_i 與 frag_j（某個最佳旋轉）的「綜合成本」
      best_rot[i][j]    = 使成本最小的旋轉 (0,1,2,3)
    """
    n = len(ref_tiles)
    assert n == len(frag_tiles), "ref tiles 與 fragment tiles 數量不一致"

    # 先抽 feature，之後統一做 z-score
    ref_feats_raw = []
    frag_feats_raw = [[None]*4 for _ in range(n)]
    all_feats = []

    for i in range(n):
        f = tile_feature_advanced(ref_tiles[i])
        ref_feats_raw.append(f)
        all_feats.append(f)

    for j in range(n):
        tile = frag_tiles[j]
        for rot in range(4):
            if rot == 0:
                t = tile
            else:
                t = np.rot90(tile, rot).copy()
            f = tile_feature_advanced(t)
            frag_feats_raw[j][rot] = f
            all_feats.append(f)

    all_feats_arr = np.stack(all_feats, axis=0)
    mean = all_feats_arr.mean(axis=0, keepdims=True)
    std = all_feats_arr.std(axis=0, keepdims=True) + 1e-6

    ref_feats = [((f - mean) / std).astype(np.float32) for f in ref_feats_raw]
    frag_feats = [
        [((frag_feats_raw[j][rot] - mean) / std).astype(np.float32) for rot in range(4)]
        for j in range(n)
    ]

    # ---- 這裡開始：同時考慮「特徵距離」與「像素 MSE」 ----
    ALPHA_FEAT = 0.3   # 特徵的權重
    ALPHA_PIX  = 0.7   # 像素 MSE 的權重（Starry Night 紋理多，這個可以放大）

    cost_matrix = np.zeros((n, n), dtype=np.float32)
    best_rot = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        fi = ref_feats[i]
        ref_tile = ref_tiles[i]
        for j in range(n):
            best_c = float("inf")
            best_r = 0
            frag_base = frag_tiles[j]
            for rot in range(4):
                fj = frag_feats[j][rot]

                # feature 距離
                diff_feat = fi - fj
                c_feat = float(np.sqrt(np.sum(diff_feat * diff_feat)))

                # 像素 MSE（直接跟旋轉後的 tile 比）
                if rot == 0:
                    frag_rot = frag_base
                else:
                    frag_rot = np.rot90(frag_base, rot).copy()
                c_pix = tile_pixel_cost(ref_tile, frag_rot)

                # 綜合成本
                c = ALPHA_FEAT * c_feat + ALPHA_PIX * c_pix

                if c < best_c:
                    best_c = c
                    best_r = rot

            cost_matrix[i, j] = best_c
            best_rot[i, j] = best_r

    print("[INFO] Cost matrix (feature + pixel) & best rotations computed.")
    return cost_matrix, best_rot


# ============================================================
# Bitmask DP Assignment（最小權重匹配）
# ============================================================

def solve_assignment_dp(cost_matrix):
    """
    cost_matrix: shape (n, n)， rows=ref tiles, cols=fragment tiles
    回傳：
      assignment[i] = 選到的 column j
      best_cost     = 總成本
    """
    n = cost_matrix.shape[0]
    size = 1 << n
    dp = [float("inf")] * size
    prev_mask = [-1] * size
    prev_col = [-1] * size

    dp[0] = 0.0

    for mask in range(size):
        i = bin(mask).count("1")  # 已指派的 row 數 = 下一個 row index
        if i >= n:
            continue
        cur_cost = dp[mask]
        if cur_cost == float("inf"):
            continue

        for j in range(n):
            if (mask & (1 << j)) == 0:
                new_mask = mask | (1 << j)
                new_cost = cur_cost + float(cost_matrix[i, j])
                if new_cost < dp[new_mask]:
                    dp[new_mask] = new_cost
                    prev_mask[new_mask] = mask
                    prev_col[new_mask] = j

    full_mask = size - 1
    best_cost = dp[full_mask]
    if best_cost == float("inf"):
        raise RuntimeError("Assignment DP failed: no finite solution")

    assignment = [None] * n
    mask = full_mask
    while mask:
        j = prev_col[mask]
        pm = prev_mask[mask]
        i = bin(mask).count("1") - 1
        assignment[i] = j
        mask = pm

    print(f"[INFO] Assignment DP done. best_cost = {best_cost:.4f}")
    return assignment, best_cost


# ============================================================
# 由 assignment & rotation 組回 layout / 影像 / 動畫
# ============================================================

def build_layout_from_assignment(assignment, best_rot, N=4):
    """
    assignment[i] = column j (fragment index)
    best_rot[i][j] = rotation for that pair
    ref tile index i 對應 grid: r = i // N, c = i % N
    """
    n = len(assignment)
    layout = [[None for _ in range(N)] for _ in range(N)]
    for i in range(n):
        r = i // N
        c = i % N
        j = assignment[i]
        rot = int(best_rot[i, j])
        layout[r][c] = (j, rot)
    return layout


def build_solved_image(layout, tiles_rectified):
    N = len(layout)
    h, w, _ = tiles_rectified[0].shape
    canvas = np.zeros((N*h, N*w, 3), dtype=np.uint8)
    for r in range(N):
        for c in range(N):
            idx, rot = layout[r][c]
            tile = tiles_rectified[idx]
            if rot != 0:
                tile = np.rot90(tile, rot).copy()
            y0 = r * h
            x0 = c * w
            canvas[y0:y0+h, x0:x0+w] = tile
    return canvas


def build_animation_pieces(fragment_img, tiles_rectified, centers, layout):
    """
    利用原始 fragment image 裡的中心點當作散落位置。
    layout 告訴我們每塊 tile 最終的位置與旋轉。
    """
    h0, w0, _ = tiles_rectified[0].shape
    H, W, _ = fragment_img.shape
    N = len(layout)

    # fragment index -> (grid_x, grid_y, rot)
    grid_info = {}
    for r in range(N):
        for c in range(N):
            idx, rot = layout[r][c]
            grid_x = c * w0
            grid_y = r * h0
            grid_info[idx] = (grid_x, grid_y, rot)

    pieces_anim = []
    for idx, (cx, cy) in enumerate(centers):
        tile = tiles_rectified[idx]
        grid_x, grid_y, rot = grid_info[idx]
        if rot != 0:
            tile = np.rot90(tile, rot).copy()

        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        scatter_x = float(cx - w0/2)
        scatter_y = float(cy - h0/2)

        pieces_anim.append({
            "img": tile_rgb,
            "grid_x": float(grid_x),
            "grid_y": float(grid_y),
            "scatter_x": scatter_x,
            "scatter_y": scatter_y,
            "start_angle": float(random.uniform(-160, 160)),
        })

    return pieces_anim, (W, H)


# ============================================================
# 主程式
# ============================================================

def main():
    if len(sys.argv) < 4:
        print("Usage: python main_solve_and_animate.py fragment.png solved.png anim.gif")
        print("Note: starry_night.png must exist in the same directory.")
        return

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fragment_path = sys.argv[1]  # 例如 starry_night_rotate.png
    solved_path   = sys.argv[2]
    anim_path     = sys.argv[3]

    # --------------------------------------------
    # 1) 讀取 fragment image & 分割 tile
    # --------------------------------------------
    frag_img = load_image(fragment_path)
    tiles_raw = segment_tiles(frag_img)
    if len(tiles_raw) == 0:
        print("[ERROR] No tiles found in fragment image.")
        return

    tiles_rectified = []
    centers = []
    for tile, cx, cy in tiles_raw:
        tiles_rectified.append(rectify_tile(tile))
        centers.append((cx, cy))

    n_tiles = len(tiles_rectified)
    N = int(round(math.sqrt(n_tiles)))
    assert N * N == n_tiles == 16, f"[ERROR] Expect 16 tiles, got {n_tiles}"

    # --------------------------------------------
    # 2) 讀取 Starry Night 原圖並切格
    # --------------------------------------------
    ref_img = load_image("starry_night.png")
    ref_tiles = cut_reference_tiles(ref_img, N=N)

    # --------------------------------------------
    # 3) 建立 cost matrix & best rotation
    # --------------------------------------------
    cost_matrix, best_rot = build_cost_matrix_with_rot(ref_tiles, tiles_rectified)

    # --------------------------------------------
    # 4) 透過 bitmask DP 解 assignment
    # --------------------------------------------
    assignment, best_cost = solve_assignment_dp(cost_matrix)
    print(f"[INFO] Final assignment cost = {best_cost:.4f}")

    # --------------------------------------------
    # 5) 轉為 layout 並生成 solved 圖
    # --------------------------------------------
    layout = build_layout_from_assignment(assignment, best_rot, N=N)
    solved_img = build_solved_image(layout, tiles_rectified)
    cv2.imwrite(solved_path, solved_img)
    print(f"[INFO] Saved solved image -> {solved_path}")

    # --------------------------------------------
    # 6) 準備動畫：起點 = fragment.png, 終點 = 原圖
    # --------------------------------------------
    pieces_anim, (W, H) = build_animation_pieces(
        frag_img, tiles_rectified, centers, layout
    )

    frag_rgb = cv2.cvtColor(frag_img, cv2.COLOR_BGR2RGB)
    solved_rgb = cv2.cvtColor(solved_img, cv2.COLOR_BGR2RGB)
    solved_rgb = cv2.resize(solved_rgb, (W, H))

    reference_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.resize(reference_rgb, (W, H))

    frames = []

    num_frames_hold = 5     # 一開始完全靜止顯示 fragment（你的 rotate 圖）
    num_frames_move = 45    # 碎片滑回去的影格數
    num_frames_fade = 20    # 拼圖 → 原圖 淡入影格數

    # -------- (0) 起始：完全是你提供的 fragment.png --------
    for _ in range(num_frames_hold):
        frames.append(frag_rgb.copy())

    # -------- (1) 散落碎片 → grid（只平移，不再旋轉，避免多邊形） --------
    for t in range(num_frames_move):
        alpha = t / (num_frames_move - 1)
        frame = np.zeros((H, W, 3), dtype=np.uint8)  # 黑底

        for p in pieces_anim:
            # 位置插值
            x = (1 - alpha) * p["scatter_x"] + alpha * p["grid_x"]
            y = (1 - alpha) * p["scatter_y"] + alpha * p["grid_y"]

            tile_img = p["img"]   # 已經是正確朝向的方形 tile
            h, w, _ = tile_img.shape

            xi, yi = int(x), int(y)
            if 0 <= xi < W - w and 0 <= yi < H - h:
                frame[yi:yi+h, xi:xi+w] = tile_img

        frames.append(frame)

    # -------- (2) grid 拼圖 → 淡入真正原圖 --------
    for t in range(num_frames_fade):
        alpha = t / (num_frames_fade - 1)
        blended = ((1 - alpha) * solved_rgb + alpha * reference_rgb).astype(np.uint8)
        frames.append(blended)

    # -------- 寫出 GIF --------
    import imageio
    imageio.mimsave(anim_path, frames, fps=10)
    print(f"[INFO] Saved improved animation -> {anim_path}")

if __name__ == "__main__":
    main()
