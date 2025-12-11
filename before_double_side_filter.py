import cv2
import numpy as np
import math
from collections import namedtuple
import itertools
import sys
import os

# ---- 配置参数 ----
EDGE_STRIP_WIDTH = 12        # 用于提取边缘条带的宽度 (像素)
COLOR_BINS = 8              # HSV 每个通道的 bin 数
GRAD_BINS = 8               # 梯度方向直方图 bin 数
ALPHA = 0.4                 # 颜色差权重
BETAB = 0.6                 # 梯度差权重
MIN_COMPONENT_AREA = 10    # 过滤太小的噪声连通域
# MAX_SEARCH_SOLUTIONS = 100000    # 只找一个最优解，够用了

# Store info of a piece in a specific rotation
PieceRot = namedtuple("PieceRot", ["piece_idx", "img", "edges"])



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
    
    # 获取宽高（可能有旋转）
    w, h = rect[1]
    w, h = int(max(w, 1)), int(max(h, 1))
    
    # 对角点进行排序：top-left, top-right, bottom-right, bottom-left
    pts = np.array(box, dtype="float32")
    
    # 计算中心
    center = pts.mean(axis=0)
    
    # 按角度排序（从上-左，顺时针）
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    pts_sorted = pts[sorted_indices]
    
    # 找到最接近左上角的点作为起点
    distances = np.linalg.norm(pts_sorted - pts_sorted[0], axis=1)
    # 排序后应该是：左上 -> 右上 -> 右下 -> 左下
    src_pts = pts_sorted.astype("float32")
    
    # 目标矩形的四个角（左上、右上、右下、左下）
    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")
    
    try:
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    except cv2.error:
        # 如果变换矩阵计算失败，返回原图
        return piece_img
    
    # 先放大 2 倍以提高采样精度
    big = cv2.resize(piece_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    M_scaled = M.copy()
    M_scaled[0, 2] *= 2  # 调整平移参数以适应放大后的图像
    M_scaled[1, 2] *= 2
    
    # 执行透视变换
    rectified = cv2.warpPerspective(
        big, M_scaled, (w * 2, h * 2),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    # 缩回原尺寸
    rectified = cv2.resize(rectified, (w, h), interpolation=cv2.INTER_AREA)
    
    # 可选：锐化处理
    if smooth:
        blur = cv2.GaussianBlur(rectified, (3, 3), sigmaX=1.0)
        rectified = cv2.addWeighted(rectified, 1.2, blur, -0.2, 0)
    
    return rectified


#TODO: change and consider irregular shapes




def compute_color_hist(strip, bins=COLOR_BINS):
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2],
                        None,
                        [bins, bins, bins],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0,
                         norm_type=cv2.NORM_L1)
    return hist.flatten()



def compute_grad_hist(gray_strip, mag_strip, ang_strip, bins=GRAD_BINS):
    """
    梯度方向直方图，angle 在 [0, 2pi)，以 mag 为权重。
    """
    # 拉平成 1D
    angles = ang_strip.flatten()
    mags = mag_strip.flatten()

    # 映射到 [0, 2pi)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    hist = np.zeros(bins, dtype=np.float32)
    bin_width = 2 * np.pi / bins

    for a, m in zip(angles, mags):
        b = int(a // bin_width)
        if 0 <= b < bins:
            hist[b] += m

    # 归一化
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def compute_piece_edge_descriptors(piece_img, edge_strip_width=EDGE_STRIP_WIDTH):
    """
    对一个 piece（已经是某个固定旋转）的四条边，计算：
    - 颜色直方图
    - 梯度方向直方图
    返回 dict: {"top": desc, "right": desc, "bottom": desc, "left": desc}
    其中 desc = {"color": color_vec, "grad": grad_vec}
    """
    h, w, _ = piece_img.shape
    k = min(edge_strip_width, h // 3, w // 3)  # 防止太大

    # 获取边缘 strip
    top_strip = piece_img[0:k, :, :]
    bottom_strip = piece_img[h - k:h, :, :]
    left_strip = piece_img[:, 0:k, :]
    right_strip = piece_img[:, w - k:w, :]

    # 梯度在整张 piece 上算一次，然后取对应 strip
    gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx)

    top_mag = mag[0:k, :]
    top_ang = ang[0:k, :]
    bottom_mag = mag[h - k:h, :]
    bottom_ang = ang[h - k:h, :]
    left_mag = mag[:, 0:k]
    left_ang = ang[:, 0:k]
    right_mag = mag[:, w - k:w]
    right_ang = ang[:, w - k:w]

    edges = {}

    # top
    edges[0] = [compute_color_hist(top_strip), compute_grad_hist(gray[0:k, :], top_mag, top_ang)]
    # right
    edges[1] = [compute_color_hist(right_strip), compute_grad_hist(gray[:, w - k:w], right_mag, right_ang)] 
    # bottom
    edges[2] = [compute_color_hist(bottom_strip), compute_grad_hist(gray[h - k:h, :], bottom_mag, bottom_ang)]
    # left
    edges[3] = [compute_color_hist(left_strip), compute_grad_hist(gray[:, 0:k], left_mag, left_ang)]    


    return edges

def color_distance(c1, c2):
    # 直方图已经 normalize 了
    inter = np.minimum(c1, c2).sum()
    # 交集越大，相似度越高 → 距离越小
    return 1.0 - inter

def grad_distance(g1, g2):
    inter = np.minimum(g1, g2).sum()
    return 1.0 - inter



def edge_distance(descA, descB, alpha=ALPHA, beta=BETAB):
    color_diff = color_distance(descA[0], descB[0])
    grad_diff  = grad_distance(descA[1],  descB[1])
    return alpha * color_diff + beta * grad_diff



# ---------- 构建所有旋转版本 ----------

def build_all_rotations(norm_pieces):
    """
    对每个 piece 生成 4 个旋转版本，并计算每个版本的 edge 描述子。
    返回：
        all_rots: (piece_idx, rot_idx) -> PieceRot
    """
    all_rots = []
    for i, p in enumerate(norm_pieces):
        # for rot in range(4):
        #     img_rot = rotate_piece(p, rot)
        edges = compute_piece_edge_descriptors(p)
        all_rots.append(PieceRot(piece_idx=i, img=p, edges=edges))
    print(f"[INFO] Built {len(all_rots)} rotated versions.")
    return all_rots



# ---------- 布局搜索（DFS + 剪枝） ----------

class PuzzleSolver:
    def __init__(self, all_rots, grid_rows, grid_cols):
        self.all_rots = all_rots               # dict[(piece_idx, rot_idx)] -> PieceRot
        self.num_pieces = len(all_rots)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.positions = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]

        self.best_cost = float("inf")
        self.best_layout = None  # 2D: (piece_idx, rot_idx)

        # 当前状态
        self.current_layout = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        self.used_piece = [False] * self.num_pieces
        self.current_cost = 0.0

        self.solutions_found = 0

        self.sim = []
        self.candidates = []


    def solve(self):
        
        self._get_score()
        self._build_candidates(top_k=10)
        self._dfs(0)
        return self.best_layout, self.best_cost

    def _get_score(self):
        # Get similarity scores between all edges first
        sim = np.full((self.num_pieces, self.num_pieces, 4, 4), np.inf, dtype=np.float32)


        for i in range(self.num_pieces):
            pi = self.all_rots[i]
            for j in range(self.num_pieces):
                if i == j:
                    continue
                pj = self.all_rots[j]
                for rot1 in range(4):
                    for rot2 in range(4):
                        sim[i][j][rot1][rot2] = edge_distance(
                            pi.edges[rot1], pj.edges[rot2]
                        )
        self.sim = sim

    def _build_candidates(self, top_k=5):
        """
        对于每条边 (i, edge_i)，选出 cost 最小的 top_k 个 (j, edge_j)。
        candidates[i][edge_i] 是一个 set，元素是 (j, edge_j)。
        """
        num_pieces = self.num_pieces
        candidates = [[set() for _ in range(4)] for _ in range(num_pieces)]

        for i in range(num_pieces):
            for edge_i in range(4):
                # 收集所有 (j, edge_j, cost)
                triplets = []
                for j in range(num_pieces):
                    if i == j:
                        continue
                    for edge_j in range(4):
                        cost = self.sim[i, j, edge_i, edge_j]
                        triplets.append((cost, j, edge_j))

                # 按 cost 排序，取前 top_k
                triplets.sort(key=lambda x: x[0])
                for t in triplets[:top_k]:
                    _, j, edge_j = t
                    candidates[i][edge_i].add((j, edge_j))

        self.candidates = candidates
        print("[INFO] Built candidate neighbor sets with top_k =", top_k)

        
    # dfs search top-k smallest edge difference
    def _dfs(self, pos_idx):
        """
        深度优先 + branch-and-bound：
        - 不再用 solutions_found / MAX_SEARCH_SOLUTIONS 提前退出
        - 只保留基于 current_cost / best_cost 的安全剪枝
        """
        # 所有位置都填满了，检查一次完整布局
        if pos_idx == len(self.positions):
            if self.current_cost < self.best_cost:
                self.best_cost = self.current_cost
                self.best_layout = [row[:] for row in self.current_layout]
                print(f"[INFO] Found new best layout, cost={self.best_cost:.4f}")
            return

        r, c = self.positions[pos_idx]

        # 尝试放每一个尚未使用的 piece
        for piece_idx in range(self.num_pieces):
            if self.used_piece[piece_idx]:
                continue

            # 如果当前是“只平移不旋转”的例子，可以把 range(4) 改成 [0]
            for rot in range(4): # 0 : 0, 1 : 270, 2 : 180, 3 : 90
                # key = (piece_idx, rot)
                # if key not in self.all_rots:
                #     continue

            

                # 只考虑与已放好的“上”和“左”的匹配代价
                add_cost = 0.0

# Translate: If we have n pieces, calculate all scores, finish the puzzle from a random start.

                # TOP
                if r > 0 and self.current_layout[r - 1][c] is not None:
                    up_piece_idx, up_rot = self.current_layout[r - 1][c]
                    edge_up_down = (up_rot + 2) % 4   # 上块的 bottom
                    edge_cur_top = rot                # 当前块的 top
                    if (piece_idx, edge_cur_top) not in self.candidates[up_piece_idx][edge_up_down]:
                        continue
                    # for up_rot: the up_piece_idx edge will be 
                    # up_rot :  0 1 2 3
                    # edge_idx: 2 3 0 1
                    add_cost += self.sim[up_piece_idx, piece_idx, edge_up_down, edge_cur_top]

                # LEFT
                if c > 0 and self.current_layout[r][c - 1] is not None:
                    left_piece_idx, left_rot = self.current_layout[r][c - 1]
                    edge_left_right = (left_rot + 1) % 4      # 左块的 right
                    edge_cur_left   = (rot + 3) % 4
                    if (piece_idx, edge_cur_left) not in self.candidates[left_piece_idx][edge_left_right]:
                        continue
                    # for left_rot: the left_piece_idx edge will be 
                    # left_rot: 0 1 2 3
                    # edge_idx: 1 2 3 0
                    add_cost += self.sim[left_piece_idx, piece_idx, edge_left_right, edge_cur_left]
                    
                new_cost = self.current_cost + add_cost

                # branch-and-bound 剪枝：当前 partial cost 已经 >= best，就没必要继续
                if new_cost >= self.best_cost:
                    continue
                    
                if add_cost > 1:
                    continue

                # 选择当前 piece+rot 放到 (r, c)
                self.current_layout[r][c] = (piece_idx, rot)
                self.used_piece[piece_idx] = True
                prev_cost = self.current_cost
                self.current_cost = new_cost

                # 递归到下一个格子
                self._dfs(pos_idx + 1)

                # 回溯
                self.current_layout[r][c] = None
                self.used_piece[piece_idx] = False
                self.current_cost = prev_cost

def render_layout(best_layout, all_rots):
    """
    根据 best_layout 把拼图拼好。
    所有块都会被 resize 成同一个尺寸：(max_h, max_w)，
    其中 max_h / max_w 是所有旋转后拼图块中最大的高 / 宽。
    """
    rows = len(best_layout)
    cols = len(best_layout[0])

    # 1. 先遍历一遍，找所有旋转后 piece 的最大高度和宽度
    max_h = 0
    max_w = 0
    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]
            piece_rot = all_rots[piece_idx]
            img = rotate_piece(piece_rot.img, rot_idx)
            h, w = img.shape[:2]
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w

    # 2. 画布大小 = rows * max_h, cols * max_w
    total_h = rows * max_h
    total_w = cols * max_w
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # 3. 把每块 resize 成 (max_w, max_h)，按规则网格贴到对应位置
    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]
            piece_rot = all_rots[piece_idx]
            img = rotate_piece(piece_rot.img, rot_idx)

            # 这里直接拉伸到统一尺寸（如果想等比例 + padding 再说）
            img_resized = cv2.resize(img, (max_w, max_h), interpolation=cv2.INTER_AREA)

            y0 = r * max_h
            x0 = c * max_w
            canvas[y0:y0 + max_h, x0:x0 + max_w, :] = img_resized
    
    square_size = max(total_h, total_w)
    canvas = cv2.resize(canvas, (square_size, square_size), interpolation=cv2.INTER_AREA)

    return canvas

# # ---------- 重建图片 & 简易动画 ----------

# def render_layout(best_layout, all_rots, piece_size):
#     """
#     根据 best_layout 把拼图拼好。
#     best_layout: 2D: (piece_idx, rot_idx)
#     piece_size: (h, w)
#     """
#     rows = len(best_layout)
#     cols = len(best_layout[0])
#     ph, pw = piece_size

#     canvas = np.zeros((rows * ph, cols * pw, 3), dtype=np.uint8)

#     for r in range(rows):
#         for c in range(cols):
#             piece_idx, rot_idx = best_layout[r][c]
#             piece_rot = all_rots[piece_idx]
#             img = rotate_piece(piece_rot.img, rot_idx)

#             # 确保大小相同
#             img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
#             # ph, pw = img.shape[:2]
#             y0 = r * ph
#             x0 = c * pw
#             canvas[y0:y0 + ph, x0:x0 + pw, :] = img

#     return canvas


def rotate_piece(img, rot_idx):
    """
    rot_idx = 0,1,2,3 分别对应 0°,270°,180°,90°
    顺时针旋转
    """
    return np.rot90(img, rot_idx, axes=(0, 1))


def generate_mp4_animation_cpp_style(original_image, best_layout, all_rots, rectified_pieces, piece_size, output_path,
                                     fps=10, duration_per_piece=0.5, hold_final=2.0, hold_initial=1.0):
    """
    生成MP4动画：按照C++ AnimationGenerator的方式
    - 从原始input图片开始
    - 逐个piece从原始位置移动到目标位置

    参数：
    - original_image: 原始输入图片（带黑底的打乱拼图）
    - best_layout: 最佳布局方案
    - all_rots: 所有piece的旋转版本
    - rectified_pieces: 矫正后的pieces (unused)
    - piece_size: (h, w) piece尺寸
    - output_path: 输出视频路径
    - fps: 帧率
    - duration_per_piece: 每个piece动画时长（秒）
    - hold_final: 最终状态停留时间（秒）
    - hold_initial: 初始原始图片停留时间（秒）
    """
    try:
        import imageio
    except ImportError:
        print("[ERROR] imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return

    rows = len(best_layout)
    cols = len(best_layout[0])
    ph, pw = piece_size

    # 计算canvas大小，取原始图片和grid大小的最大值
    canvas_h = max(original_image.shape[0], rows * ph)
    canvas_w = max(original_image.shape[1], cols * pw)

    frames = []
    frames_per_piece = max(1, int(fps * duration_per_piece))

    print(f"\n[INFO] Generating MP4 animation (C++ style - starting from original image)...")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}, Duration per piece: {duration_per_piece}s")
    print(f"  Canvas size: {canvas_w}x{canvas_h}")

    # 1. 初始帧：显示原始input图片
    initial_frames_count = int(fps * hold_initial)

    # 将原始图片放在canvas中央
    initial_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_offset = (canvas_h - original_image.shape[0]) // 2
    x_offset = (canvas_w - original_image.shape[1]) // 2
    initial_canvas[y_offset:y_offset+original_image.shape[0],
                   x_offset:x_offset+original_image.shape[1]] = original_image

    initial_canvas_rgb = cv2.cvtColor(initial_canvas, cv2.COLOR_BGR2RGB)

    for _ in range(initial_frames_count):
        frames.append(initial_canvas_rgb.copy())

    print(f"  Added {initial_frames_count} initial frames showing original image")

    # 2. 从原始图片中检测每个piece的中心位置
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    centers = {}
    for i in range(1, num_labels):
        if stats[i][4] >= MIN_COMPONENT_AREA:
            cx, cy = centroids[i]
            # 考虑offset
            centers[len(centers)] = (cx + x_offset, cy + y_offset)

    print(f"  Detected {len(centers)} piece centers in original image")

    # 3. 构建piece信息列表（按layout顺序）
    piece_infos = []

    # Grid位置的offset（居中）
    grid_y_offset = (canvas_h - rows * ph) // 2
    grid_x_offset = (canvas_w - cols * pw) // 2

    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]

            # 目标位置（grid中心，考虑offset）
            grid_center_x = grid_x_offset + c * pw + pw / 2
            grid_center_y = grid_y_offset + r * ph + ph / 2

            # 原始位置（中心）
            if piece_idx in centers:
                start_x, start_y = centers[piece_idx]
            else:
                start_x, start_y = grid_center_x, grid_center_y

            # 获取piece图像（旋转后）
            piece_rot = all_rots[piece_idx]
            tile = rotate_piece(piece_rot.img, rot_idx)
            tile = cv2.resize(tile, (pw, ph), interpolation=cv2.INTER_AREA)

            piece_infos.append({
                'piece_idx': piece_idx,
                'rot_idx': rot_idx,
                'img': tile,
                'start_center': (start_x, start_y),
                'end_center': (grid_center_x, grid_center_y),
            })

    # 4. 逐个piece生成动画帧
    total_frames = initial_frames_count

    for current_piece_idx in range(len(piece_infos)):
        if current_piece_idx % 4 == 0:
            print(f"  Animating piece {current_piece_idx+1}/{len(piece_infos)}")

        for f in range(frames_per_piece):
            progress = f / max(1, frames_per_piece - 1) if frames_per_piece > 1 else 1.0
            # Ease-in-out (C++ style: smoothT = t * t * (3.0 - 2.0 * t))
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)

            # 黑色背景
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 绘制所有pieces
            for i, piece_info in enumerate(piece_infos):
                if i < current_piece_idx:
                    # 已完成：在最终位置
                    current_pos = piece_info['end_center']
                elif i == current_piece_idx:
                    # 当前动画：插值位置
                    start_x, start_y = piece_info['start_center']
                    end_x, end_y = piece_info['end_center']
                    current_x = start_x + (end_x - start_x) * smooth_progress
                    current_y = start_y + (end_y - start_y) * smooth_progress
                    current_pos = (current_x, current_y)
                else:
                    # 未开始：在原始位置
                    current_pos = piece_info['start_center']

                piece_img = piece_info['img'].copy()
                x = int(current_pos[0] - pw / 2)
                y = int(current_pos[1] - ph / 2)

                # 确保在边界内
                if 0 <= x <= canvas_w - pw and 0 <= y <= canvas_h - ph:
                    canvas[y:y+ph, x:x+pw] = piece_img

            frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            total_frames += 1

    # 5. 最终状态：停留
    final_frames_count = int(fps * hold_final)
    final_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for piece_info in piece_infos:
        x = int(piece_info['end_center'][0] - pw / 2)
        y = int(piece_info['end_center'][1] - ph / 2)
        piece_img = piece_info['img']

        if 0 <= x <= canvas_w - pw and 0 <= y <= canvas_h - ph:
            final_canvas[y:y+ph, x:x+pw] = piece_img

    final_canvas_rgb = cv2.cvtColor(final_canvas, cv2.COLOR_BGR2RGB)

    for _ in range(final_frames_count):
        frames.append(final_canvas_rgb.copy())
        total_frames += 1

    print(f"  Added {final_frames_count} final frames")

    # 6. 保存MP4
    print(f"  Total frames: {total_frames}")
    print(f"  Saving MP4...")

    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)

    print(f"[INFO] Animation saved to {output_path}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")



def generate_mp4_animation(original_image, best_layout, all_rots, rectified_pieces, piece_size, output_path,
                          fps=10, num_frames_hold=10, num_frames_move=60, num_frames_final=20):
    """
    生成MP4动画：参考animate.py的方式，所有pieces同时从原始位置移动到目标位置

    参数：
    - original_image: 原始输入图片（带黑底的打乱拼图）
    - best_layout: 最佳布局方案
    - all_rots: 所有piece的旋转版本
    - rectified_pieces: 矫正后的pieces (unused but kept for compatibility)
    - piece_size: (h, w) piece尺寸
    - output_path: 输出视频路径
    - fps: 帧率
    - num_frames_hold: 初始状态停留帧数
    - num_frames_move: 移动动画帧数
    - num_frames_final: 最终状态停留帧数
    """
    try:
        import imageio
    except ImportError:
        print("[ERROR] imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return

    rows = len(best_layout)
    cols = len(best_layout[0])
    ph, pw = piece_size

    canvas_h = rows * ph
    canvas_w = cols * pw

    frames = []

    print(f"\n[INFO] Generating MP4 animation (animate.py style)...")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps}, Canvas: {canvas_w}x{canvas_h}")

    # 1. 从原始图片中检测每个piece的中心位置
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    centers = []
    for i in range(1, num_labels):
        if stats[i][4] >= MIN_COMPONENT_AREA:
            cx, cy = centroids[i]
            centers.append((cx, cy))

    print(f"  Detected {len(centers)} piece centers")

    # 2. 构建pieces_anim：每个piece的信息
    pieces_anim = []

    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]

            # 目标位置（grid左上角）
            grid_x = c * pw
            grid_y = r * ph

            # 原始位置（scatter位置，左上角）
            if piece_idx < len(centers):
                cx, cy = centers[piece_idx]
                scatter_x = cx - pw / 2
                scatter_y = cy - ph / 2
            else:
                scatter_x = grid_x
                scatter_y = grid_y

            # 获取旋转后的piece图像
            piece_rot = all_rots[piece_idx]
            tile = rotate_piece(piece_rot.img, rot_idx)
            tile = cv2.resize(tile, (pw, ph), interpolation=cv2.INTER_AREA)
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

            pieces_anim.append({
                "img": tile_rgb,
                "grid_x": float(grid_x),
                "grid_y": float(grid_y),
                "scatter_x": float(scatter_x),
                "scatter_y": float(scatter_y),
            })

    # 3. 初始帧：显示原始打乱的fragment图片
    frag_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    frag_rgb_resized = cv2.resize(frag_rgb, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

    for _ in range(num_frames_hold):
        frames.append(frag_rgb_resized.copy())

    print(f"  Added {num_frames_hold} initial frames")

    # 4. 移动动画：所有pieces同时从scatter位置移动到grid位置
    print(f"  Generating {num_frames_move} movement frames...")

    for t in range(num_frames_move):
        alpha = t / max(1, num_frames_move - 1)

        # 黑色背景
        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for p in pieces_anim:
            # 位置插值
            x = (1 - alpha) * p["scatter_x"] + alpha * p["grid_x"]
            y = (1 - alpha) * p["scatter_y"] + alpha * p["grid_y"]

            tile_img = p["img"]
            h, w, _ = tile_img.shape

            xi, yi = int(x), int(y)

            # 确保在边界内
            if 0 <= xi <= canvas_w - w and 0 <= yi <= canvas_h - h:
                frame[yi:yi+h, xi:xi+w] = tile_img

        frames.append(frame)

    # 5. 最终状态：显示完成的拼图
    final_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for p in pieces_anim:
        x = int(p["grid_x"])
        y = int(p["grid_y"])
        tile_img = p["img"]
        h, w, _ = tile_img.shape

        if 0 <= x <= canvas_w - w and 0 <= y <= canvas_h - h:
            final_canvas[y:y+h, x:x+w] = tile_img

    for _ in range(num_frames_final):
        frames.append(final_canvas.copy())

    print(f"  Added {num_frames_final} final frames")

    # 6. 保存为MP4
    total_frames = len(frames)
    print(f"  Total frames: {total_frames}")
    print(f"  Saving MP4...")

    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)

    print(f"[INFO] Animation saved to {output_path}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")



def render_animation_sequence(best_layout, all_rots, original_pieces, piece_size=None, bg_color=(0, 0, 0), initial_frames=10, final_frames=20):
    """
    制作拼图还原动画序列：
    1. 显示初始打乱的图片（带黑底）
    2. 逐块还原到正确位置

    参数：
    - best_layout: 最佳布局
    - all_rots: 所有旋转版本
    - original_pieces: 原始检测到的pieces（用于显示初始状态）
    - piece_size: (h, w) 如果为None则使用原始尺寸
    - bg_color: 背景色
    - initial_frames: 初始状态显示的帧数
    - final_frames: 最终完成状态显示的帧数

    返回：frames 列表
    """
    rows = len(best_layout)
    cols = len(best_layout[0])
    frames = []

    if piece_size is None:
        # 使用原始尺寸，计算canvas大小
        row_heights = []
        col_widths = []

        for r in range(rows):
            max_h = 0
            for c in range(cols):
                piece_idx, rot_idx = best_layout[r][c]
                piece_rot = all_rots[piece_idx]
                img = rotate_piece(piece_rot.img, rot_idx)
                max_h = max(max_h, img.shape[0])
            row_heights.append(max_h)

        for c in range(cols):
            max_w = 0
            for r in range(rows):
                piece_idx, rot_idx = best_layout[r][c]
                piece_rot = all_rots[piece_idx]
                img = rotate_piece(piece_rot.img, rot_idx)
                max_w = max(max_w, img.shape[1])
            col_widths.append(max_w)

        canvas_h = sum(row_heights)
        canvas_w = sum(col_widths)
    else:
        ph, pw = piece_size
        canvas_h = rows * ph
        canvas_w = cols * pw
        row_heights = [ph] * rows
        col_widths = [pw] * cols

    # 创建初始状态：显示所有打乱的pieces（在原始位置，带黑底）
    initial_canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    # 将原始pieces放在grid中（打乱的状态）
    piece_idx_counter = 0
    for r in range(rows):
        for c in range(cols):
            if piece_idx_counter < len(original_pieces):
                piece = original_pieces[piece_idx_counter]

                # 计算位置
                y0 = sum(row_heights[:r])
                x0 = sum(col_widths[:c])

                if piece_size is not None:
                    piece = cv2.resize(piece, (col_widths[c], row_heights[r]), interpolation=cv2.INTER_AREA)
                else:
                    # 居中放置
                    h, w = piece.shape[:2]
                    y_offset = (row_heights[r] - h) // 2
                    x_offset = (col_widths[c] - w) // 2
                    y0 += y_offset
                    x0 += x_offset

                h, w = piece.shape[:2]
                initial_canvas[y0:y0 + h, x0:x0 + w, :] = piece
                piece_idx_counter += 1

    # 添加初始状态的多帧（让观众看清初始状态）
    for _ in range(initial_frames):
        frames.append(initial_canvas.copy())

    # 逐步还原：一块块放到正确位置
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            piece_idx, rot_idx = best_layout[r][c]
            piece_rot = all_rots[piece_idx]
            img = rotate_piece(piece_rot.img, rot_idx)

            # 计算位置
            y0 = sum(row_heights[:r])
            x0 = sum(col_widths[:c])

            if piece_size is not None:
                img = cv2.resize(img, (col_widths[c], row_heights[r]), interpolation=cv2.INTER_AREA)
                canvas[y0:y0 + row_heights[r], x0:x0 + col_widths[c], :] = img
            else:
                h, w = img.shape[:2]
                y_offset = (row_heights[r] - h) // 2
                x_offset = (col_widths[c] - w) // 2
                canvas[y0 + y_offset:y0 + y_offset + h, x0 + x_offset:x0 + x_offset + w, :] = img

            frames.append(canvas.copy())

    # 添加最终状态的多帧（让观众看清最终结果）
    for _ in range(final_frames):
        frames.append(canvas.copy())

    return frames


# ---------- 主入口 ----------

def main(input_path, output_image_path, output_anim_path=None, anim_style='simultaneous'):
    """
    主函数
    参数:
    - input_path: 输入图片路径（打乱的拼图）
    - output_image_path: 输出还原图片路径
    - output_anim_path: 输出动画路径（.mp4文件），如果为None则不生成动画
    - anim_style: 动画风格 - 'simultaneous' (所有pieces同时移动) 或 'sequential' (逐个piece移动，C++风格)
    """
    # 保存原始图片用于动画
    original_img = load_image(input_path)

    pieces = segment_pieces(original_img)

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return
    piece_size = pieces[0].shape[:2]  # 假设所有 piece 大小相同
    rectified_pieces = [rectify_piece(p) for p in pieces]

    # 默认假设是正方形布局：rows = cols = sqrt(N)
    num_pieces = len(rectified_pieces)
    side = int(round(math.sqrt(num_pieces)))

    grid_rows = side
    grid_cols = side

    all_rots = build_all_rotations(rectified_pieces)

    solver = PuzzleSolver(all_rots, grid_rows, grid_cols)
    best_layout, best_cost = solver.solve()

    if best_layout is None:
        print("[ERROR] No layout found.")
        return

    print(f"[INFO] Best layout cost: {best_cost:.4f}")

    solved_image = render_layout(best_layout, all_rots)
    cv2.imwrite(output_image_path, solved_image)
    print(f"[INFO] Saved solved puzzle image to {output_image_path}")

    # 生成MP4动画
    if output_anim_path is not None:
        if anim_style == 'sequential' or anim_style == 'cpp':
            # C++风格：逐个piece动画
            generate_mp4_animation_cpp_style(
                original_image=original_img,
                best_layout=best_layout,
                all_rots=all_rots,
                rectified_pieces=rectified_pieces,
                piece_size=piece_size,
                output_path=output_anim_path,
                fps=10,
                duration_per_piece=0.5,
                hold_final=2.0
            )
        else:
            # animate.py风格：所有pieces同时移动
            generate_mp4_animation(
                original_image=original_img,
                best_layout=best_layout,
                all_rots=all_rots,
                rectified_pieces=rectified_pieces,
                piece_size=piece_size,
                output_path=output_anim_path,
                fps=10,
                num_frames_hold=10,
                num_frames_move=60,
                num_frames_final=20
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python solve_resize_problem.py <input_image> <output_image> [output_animation.mp4] [--style=simultaneous|sequential]")
        print("\nExamples:")
        print("  python solve_resize_problem.py input.png output.png")
        print("  python solve_resize_problem.py input.png output.png animation.mp4")
        print("  python solve_resize_problem.py input.png output.png animation.mp4 --style=sequential")
        print("\nAnimation styles:")
        print("  simultaneous - All pieces move at once (default, animate.py style)")
        print("  sequential   - Pieces move one by one (C++ AnimationGenerator style)")
        print("\nNote: If animation path is provided, it must end with .mp4")
    else:
        input_path = sys.argv[1]
        output_image_path = sys.argv[2]
        output_anim_path = sys.argv[3] if len(sys.argv) >= 4 and not sys.argv[3].startswith('--') else None
        anim_style = 'simultaneous'

        # Parse style argument
        for arg in sys.argv[3:]:
            if arg.startswith('--style='):
                anim_style = arg.split('=')[1]

        if output_anim_path and not output_anim_path.endswith('.mp4'):
            print("[WARNING] Animation output should be .mp4 file")
            output_anim_path += '.mp4' if '.' not in output_anim_path else ''

        main(input_path, output_image_path, output_anim_path, anim_style=anim_style)
