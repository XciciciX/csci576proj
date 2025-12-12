import cv2
import numpy as np
import math
from collections import namedtuple
import itertools
import sys
import os
import random

# ---- 配置参数 ----
EDGE_STRIP_WIDTH = 10        # 用于提取边缘条带的宽度 (像素)
COLOR_BINS = 8              # HSV 每个通道的 bin 数
GRAD_BINS = 8               # 梯度方向直方图 bin 数
ALPHA = 0.45                 # 颜色差权重
BETAB = 0.55                 # 梯度差权重
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
    返回 dict: {edge_idx: desc}
    其中 desc = {"color": color_vec, "grad": grad_vec, "length": edge_length}
    length 直接用边的像素长度，后续用于只在长度接近时比较相似度。
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
    edges[0] = {
        "color": compute_color_hist(top_strip),
        "grad": compute_grad_hist(gray[0:k, :], top_mag, top_ang),
        "length": w
    }
    # right
    edges[1] = {
        "color": compute_color_hist(right_strip),
        "grad": compute_grad_hist(gray[:, w - k:w], right_mag, right_ang),
        "length": h
    }
    # bottom
    edges[2] = {
        "color": compute_color_hist(bottom_strip),
        "grad": compute_grad_hist(gray[h - k:h, :], bottom_mag, bottom_ang),
        "length": w
    }
    # left
    edges[3] = {
        "color": compute_color_hist(left_strip),
        "grad": compute_grad_hist(gray[:, 0:k], left_mag, left_ang),
        "length": h
    }


    return edges

def color_distance(c1, c2):
    # 直方图已经 normalize 了
    inter = np.minimum(c1, c2).sum()
    # 交集越大，相似度越高 → 距离越小
    return 1.0 - inter

def grad_distance(g1, g2):
    inter = np.minimum(g1, g2).sum()
    return 1.0 - inter


def edge_distance(eA, eB, alpha=ALPHA, beta=BETAB, len_tol=2):
    """
    只在边长接近时计算距离，否则返回 inf。
    """
    if abs(eA["length"] - eB["length"]) > len_tol:
        return float("inf")

    color_diff = color_distance(eA["color"], eB["color"])
    grad_diff  = grad_distance(eA["grad"],  eB["grad"])
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


# ---------- Irregular puzzle pipeline ----------

def cluster_rows_by_height(rectified_pieces, bin_size=10):
    """
    将高度相近的块聚类成行。
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for idx, img in enumerate(rectified_pieces):
        h, _ = img.shape[:2]
        key = int(round(h / bin_size))
        buckets[key].append(idx)

    keys_sorted = sorted(buckets.keys())
    row_clusters = [buckets[k] for k in keys_sorted]
    return row_clusters


def order_pieces_in_row(piece_indices, sim):
    """
    按左右边缘相似度给一行的块排顺序（贪心）。
    """
    if not piece_indices:
        return []

    used = {idx: False for idx in piece_indices}
    start = piece_indices[0]
    used[start] = True
    cur_piece = start
    cur_rot = 0
    row_order = [(cur_piece, cur_rot)]

    while len(row_order) < len(piece_indices):
        best_next = None
        best_rot = 0
        best_cost = float("inf")

        for j in piece_indices:
            if used[j]:
                continue
            for rot_j in range(4):
                edge_right = (cur_rot + 1) % 4
                edge_left_j = (rot_j + 3) % 4
                cost = sim[cur_piece, j, edge_right, edge_left_j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
                    best_rot = rot_j

        if best_next is None:
            # 退化情况：全是 inf，直接把剩余块拼上去
            for j in piece_indices:
                if not used[j]:
                    row_order.append((j, 0))
            break

        used[best_next] = True
        row_order.append((best_next, best_rot))
        cur_piece, cur_rot = best_next, best_rot

    return row_order


def row_pair_cost(rowA, rowB, sim, top_k_pairs=5):
    """
    计算 rowA 在上、rowB 在下 的匹配代价。
    """
    costs = []
    for (i, rot_i) in rowA:
        for (j, rot_j) in rowB:
            edge_bottom_i = (rot_i + 2) % 4
            edge_top_j = rot_j
            costs.append(sim[i, j, edge_bottom_i, edge_top_j])

    costs = [c for c in costs if not math.isinf(c)]
    if not costs:
        return float("inf")
    costs = sorted(costs)[:min(top_k_pairs, len(costs))]
    return sum(costs) / len(costs)


def order_rows_greedy(row_orders, sim):
    """
    根据上下匹配代价串联行顺序（贪心版）。
    """
    if not row_orders:
        return []

    used = [False] * len(row_orders)
    order = [0]
    used[0] = True

    while len(order) < len(row_orders):
        cur_idx = order[-1]
        best_idx = None
        best_cost = float("inf")

        for i, row in enumerate(row_orders):
            if used[i]:
                continue
            cost = row_pair_cost(row_orders[cur_idx], row, sim)
            if cost < best_cost:
                best_cost = cost
                best_idx = i

        if best_idx is None:
            break

        used[best_idx] = True
        order.append(best_idx)

    # 如果有没用到的行，按原顺序补上
    for i, flag in enumerate(used):
        if not flag:
            order.append(i)
    return order


def render_irregular_layout(best_layout, all_rots):
    """
    以实际宽度累加的方式渲染不规则 layout。
    """
    row_heights = []
    row_widths = []
    for row in best_layout:
        max_h = 0
        total_w = 0
        for piece_idx, rot_idx in row:
            img = rotate_piece(all_rots[piece_idx].img, rot_idx)
            h, w = img.shape[:2]
            max_h = max(max_h, h)
            total_w += w
        row_heights.append(max_h)
        row_widths.append(total_w)

    canvas_h = sum(row_heights)
    canvas_w = max(row_widths) if row_widths else 0
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    y_offset = 0
    for row_idx, row in enumerate(best_layout):
        x_offset = 0
        row_h = row_heights[row_idx]
        for piece_idx, rot_idx in row:
            img = rotate_piece(all_rots[piece_idx].img, rot_idx)
            h, w = img.shape[:2]
            y0 = y_offset + (row_h - h) // 2
            canvas[y0:y0 + h, x_offset:x_offset + w] = img
            x_offset += w
        y_offset += row_h

    square_size = max(canvas_h, canvas_w)
    if square_size > 0:
        canvas = cv2.resize(canvas, (square_size, square_size), interpolation=cv2.INTER_AREA)
    return canvas


class IrregularPuzzleSolver:
    """
    面向不规则行列数量/尺寸的简单启发式 solver。
    """
    def __init__(self, rectified_pieces, bin_size=10, len_tol=2):
        self.rectified_pieces = rectified_pieces
        self.bin_size = bin_size
        self.len_tol = len_tol
        self.all_rots = build_all_rotations(rectified_pieces)
        self.num_pieces = len(self.all_rots)
        self.sim = None

    def build_similarity(self):
        sim = np.full((self.num_pieces, self.num_pieces, 4, 4), np.inf, dtype=np.float32)
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                if i == j:
                    continue
                for edge_i in range(4):
                    for edge_j in range(4):
                        sim[i, j, edge_i, edge_j] = edge_distance(
                            self.all_rots[i].edges[edge_i],
                            self.all_rots[j].edges[edge_j],
                            len_tol=self.len_tol,
                        )
        self.sim = sim
        return sim

    def solve(self):
        if self.sim is None:
            self.build_similarity()

        # Step 1: 高度聚类
        row_clusters = cluster_rows_by_height(self.rectified_pieces, bin_size=self.bin_size)
        print(f"[INFO] Row clusters by height: {row_clusters}")

        # Step 2: 行内排序
        row_orders = []
        for cluster in row_clusters:
            ordered_row = order_pieces_in_row(cluster, self.sim)
            row_orders.append(ordered_row)
            print(f"[INFO] Ordered row from cluster {cluster}: {ordered_row}")

        # Step 3: 行间排序
        row_order_indices = order_rows_greedy(row_orders, self.sim)
        print(f"[INFO] Row order indices: {row_order_indices}")

        best_layout = [row_orders[i] for i in row_order_indices]
        return best_layout, {
            "row_clusters": row_clusters,
            "row_orders": row_orders,
            "row_sequence": row_order_indices,
            "sim": self.sim,
        }


class GreedyGrowSolver:
    """
    从一个 seed piece 开始，优先按行左右生长；
    一行结束后从该行最左边尝试上下开新行，再继续左右生长。
    """
    def __init__(self, all_rots, sim, cost_thresh=0.5, seed_piece=None):
        self.all_rots = all_rots
        self.sim = sim
        self.N = len(all_rots)
        self.cost_thresh = cost_thresh
        self.seed_piece = seed_piece

        self.pos = {}   # piece_idx -> (x, y)
        self.rot = {}   # piece_idx -> rot_idx
        self.used = []

    def _reset_state(self):
        self.pos = {}
        self.rot = {}
        self.used = [False] * self.N

    def init_seed(self, seed_rot=0):
        if self.seed_piece is not None:
            seed = self.seed_piece
        else:
            seed = random.randrange(self.N)
        self.pos[seed] = (0, 0)
        self.rot[seed] = seed_rot % 4
        self.used[seed] = True
        print(f"[INFO] GreedyGrow seed piece: {seed} rot={self.rot[seed]}")

    def _neighbor_coord(self, x, y, edge_rot):
        if edge_rot == 0:      # top
            return x, y - 1
        if edge_rot == 1:      # right
            return x + 1, y
        if edge_rot == 2:      # bottom
            return x, y + 1
        if edge_rot == 3:      # left
            return x - 1, y
        return x, y

    def _edge_indices_for_sim(self, edge_i_rot, rot_i, edge_j_rot, rot_j):
        edge_i_orig = (edge_i_rot + rot_i) % 4
        edge_j_orig = (edge_j_rot + rot_j) % 4
        return edge_i_orig, edge_j_orig

    def _occupied(self, x, y):
        return any((px, py) == (x, y) for px, py in self.pos.values())

    def _place_piece(self, piece_idx, rot_idx, x, y):
        if self._occupied(x, y):
            return False
        self.pos[piece_idx] = (x, y)
        self.rot[piece_idx] = rot_idx
        self.used[piece_idx] = True
        return True

    def _best_match(self, anchor_idx, anchor_edge_rot, neighbor_edge_rot, dx, dy):
        ax, ay = self.pos[anchor_idx]
        tx, ty = ax + dx, ay + dy
        if self._occupied(tx, ty):
            return None, float("inf"), (tx, ty)

        rot_i = self.rot[anchor_idx]
        best = None
        best_cost = float("inf")

        for j in range(self.N):
            if self.used[j]:
                continue
            for rot_j in range(4):
                edge_i_orig, edge_j_orig = self._edge_indices_for_sim(anchor_edge_rot, rot_i, neighbor_edge_rot, rot_j)
                cost = self.sim[anchor_idx, j, edge_i_orig, edge_j_orig]
                if cost < best_cost and cost < self.cost_thresh:
                    best_cost = cost
                    best = (j, rot_j)
        return best, best_cost, (tx, ty)

    def _extend_direction(self, start_idx, anchor_edge_rot, neighbor_edge_rot, dx, dy):
        """
        不断在给定方向上追加块，直到没有低于阈值的匹配。
        返回最后一个在该方向上的块 idx。
        """
        cur = start_idx
        last = cur
        while True:
            best, best_cost, (tx, ty) = self._best_match(cur, anchor_edge_rot, neighbor_edge_rot, dx, dy)
            if best is None:
                break
            j, rot_j = best
            placed = self._place_piece(j, rot_j, tx, ty)
            if not placed:
                break
            print(f"[STEP] Row extend: piece {j} (rot={rot_j}) at ({tx},{ty}) cost={best_cost:.4f}")
            cur = j
            last = j
        return last

    def _get_leftmost_in_row(self, row_y):
        candidates = [(x, p) for p, (x, y) in self.pos.items() if y == row_y]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _grow_once(self):
        while True:
            # 当前行横向生长：先向左再向右
            current_y = list(self.pos.values())[0][1]  # 取任意已放块的 y
            left_anchor = self._get_leftmost_in_row(current_y)
            if left_anchor is None:
                break

            # 向左扩张
            leftmost = self._extend_direction(left_anchor, anchor_edge_rot=3, neighbor_edge_rot=1, dx=-1, dy=0)
            # 向右扩张（从起点而不是最右，避免错过种子右边）
            self._extend_direction(left_anchor, anchor_edge_rot=1, neighbor_edge_rot=3, dx=1, dy=0)

            # 更新行最左块（扩张后可能改变）
            left_anchor = self._get_leftmost_in_row(current_y)
            # 尝试从左端往上/下接新行
            best_up, cost_up, (tx_up, ty_up) = self._best_match(left_anchor, anchor_edge_rot=0, neighbor_edge_rot=2, dx=0, dy=-1)
            best_down, cost_down, (tx_down, ty_down) = self._best_match(left_anchor, anchor_edge_rot=2, neighbor_edge_rot=0, dx=0, dy=1)

            candidate_choices = []
            if best_up is not None:
                candidate_choices.append((cost_up, best_up, tx_up, ty_up, 0, 2))
            if best_down is not None:
                candidate_choices.append((cost_down, best_down, tx_down, ty_down, 2, 0))

            candidate_choices = [c for c in candidate_choices if c[0] < self.cost_thresh]

            if not candidate_choices:
                print("[INFO] GreedyGrow stopped: no vertical match under threshold.")
                break

            candidate_choices.sort(key=lambda t: t[0])
            best_cost, (j, rot_j), tx, ty, anchor_edge_rot, neighbor_edge_rot = candidate_choices[0]

            placed = self._place_piece(j, rot_j, tx, ty)
            if placed:
                print(f"[STEP] New row seed: piece {j} (rot={rot_j}) at ({tx},{ty}) from edge {anchor_edge_rot}, cost={best_cost:.4f}")
            else:
                print("[WARN] Vertical placement collided; stopping.")
                break

            if all(self.used):
                print("[INFO] GreedyGrow placed all pieces.")
                break

        return self._build_layout_from_positions()

    def solve(self):
        best_layout = []
        best_info = {}
        best_count = -1

        for seed_piece in range(self.N):
            for seed_rot in range(4):
                self.seed_piece = seed_piece
                self._reset_state()
                self.init_seed(seed_rot=seed_rot)
                layout, info = self._grow_once()
                placed_count = sum(self.used)
                print(f"[INFO] Seed piece={seed_piece} rot={seed_rot} placed {placed_count}/{self.N} pieces.")

                if placed_count > best_count:
                    best_count = placed_count
                    best_layout = layout
                    best_info = info

                if placed_count == self.N:
                    return best_layout, best_info

        return best_layout, best_info

    def _build_layout_from_positions(self):
        if not self.pos:
            return [], {}

        # 归一化坐标从 (0,0) 开始
        xs = [x for x, _ in self.pos.values()]
        ys = [y for _, y in self.pos.values()]
        min_x, min_y = min(xs), min(ys)
        norm_pos = {p: (x - min_x, y - min_y) for p, (x, y) in self.pos.items()}

        # 用稀疏坐标生成按行的布局
        rows = {}
        for p, (x, y) in norm_pos.items():
            if y not in rows:
                rows[y] = []
            rows[y].append((x, p, self.rot[p]))

        best_layout = []
        for y in sorted(rows.keys()):
            row_cells = sorted(rows[y], key=lambda t: t[0])
            best_layout.append([(p, r) for _, p, r in row_cells])

        return best_layout, {"norm_pos": norm_pos, "rot": self.rot}


def render_grow_layout(norm_pos, rot_map, all_rots):
    """
    根据坐标/旋转渲染生长式 solver 的结果。
    """
    if not norm_pos:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    xs = sorted({x for x, _ in norm_pos.values()})
    ys = sorted({y for _, y in norm_pos.values()})

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    col_widths = [0] * len(xs)
    row_heights = [0] * len(ys)

    for p, (x, y) in norm_pos.items():
        idx_x = x_to_idx[x]
        idx_y = y_to_idx[y]
        img = rotate_piece(all_rots[p].img, rot_map[p])
        h, w = img.shape[:2]
        col_widths[idx_x] = max(col_widths[idx_x], w)
        row_heights[idx_y] = max(row_heights[idx_y], h)

    canvas_h = sum(row_heights)
    canvas_w = sum(col_widths)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for p, (x, y) in norm_pos.items():
        idx_x = x_to_idx[x]
        idx_y = y_to_idx[y]
        img = rotate_piece(all_rots[p].img, rot_map[p])
        h, w = img.shape[:2]

        x_offset = sum(col_widths[:idx_x])
        y_offset = sum(row_heights[:idx_y])

        # 居中贴合当前 row/col 的最大尺寸
        y0 = y_offset + (row_heights[idx_y] - h) // 2
        x0 = x_offset + (col_widths[idx_x] - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = img

    square_size = max(canvas_h, canvas_w)
    if square_size > 0:
        canvas = cv2.resize(canvas, (square_size, square_size), interpolation=cv2.INTER_AREA)
    return canvas



# ---------- 布局搜索（DFS + 剪枝） ----------

class PuzzleSolver:
    def __init__(self, all_rots, grid_rows, grid_cols, keep_top_n=5):
        self.all_rots = all_rots
        self.num_pieces = len(all_rots)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.positions = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]

        self.keep_top_n = keep_top_n
        self.best_solutions = []  # [(cost, layout), ...]
        self.seen_layouts = set()  # 👈 用于去重
        
        self.current_layout = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        self.used_piece = [False] * self.num_pieces
        self.current_cost = 0.0

        self.solutions_found = 0

        self.sim = []
        self.candidates = []

    def solve(self):
        self._get_score()
        self._build_candidates(top_k=9, mutual=False)
        self._dfs(0)
        print(f"best solutions: {self.best_solutions}")
        # 👇 验证并去除完全相同的 layout
        unique_solutions = []
        seen_layouts_final = set()

        for cost, layout in self.best_solutions:
            layout_tuple = self._layout_to_tuple(layout)
            if layout_tuple not in seen_layouts_final:
                unique_solutions.append((cost, layout))
                seen_layouts_final.add(layout_tuple)
            else:
                print(f"[WARNING] Found duplicate layout with cost={cost:.4f}, removing...")

        if len(unique_solutions) < len(self.best_solutions):
            print(f"[INFO] Removed {len(self.best_solutions) - len(unique_solutions)} duplicate layouts")
            print(f"[INFO] Unique solutions: {len(unique_solutions)}/{len(self.best_solutions)}")

        # 👇 返回所有 top-N 结果
        return unique_solutions

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

    def _build_candidates(self, top_k=5, mutual=True):
        """
        mutual=True: 只保留双向都在 top_k 中的候选
        """
        num_pieces = self.num_pieces
        candidates = [[set() for _ in range(4)] for _ in range(num_pieces)]

        # 第一步：找每条边的 top_k 候选
        raw_candidates = [[[] for _ in range(4)] for _ in range(num_pieces)]
        
        for i in range(num_pieces):
            for edge_i in range(4):
                triplets = []
                for j in range(num_pieces):
                    if i == j:
                        continue
                    for edge_j in range(4):
                        cost = self.sim[i, j, edge_i, edge_j]
                        triplets.append((cost, j, edge_j))

                triplets.sort(key=lambda x: x[0])
                raw_candidates[i][edge_i] = triplets[:top_k]

        # 第二步：双向过滤
        for i in range(num_pieces):
            for edge_i in range(4):
                for (cost_ij, j, edge_j) in raw_candidates[i][edge_i]:
                    if mutual:
                        # 检查 j->i 是否也在 top_k 中
                        j_candidates = [t for t in raw_candidates[j][edge_j]]
                        is_mutual = any(t[1] == i and t[2] == edge_i for t in j_candidates)
                        
                        if is_mutual:
                            candidates[i][edge_i].add((j, edge_j))
                            # print(f"[MUTUAL] Piece {i} Edge {edge_i} ({EDGE_NAMES[edge_i]}) <-> "
                            #       f"Piece {j} Edge {edge_j} ({EDGE_NAMES[edge_j]}), cost={cost_ij:.4f}")
                    else:
                        candidates[i][edge_i].add((j, edge_j))

        self.candidates = candidates
        print(f"[INFO] Built candidate sets (mutual={mutual}, top_k={top_k})")
        
    # dfs search top-k smallest edge difference
    def _dfs(self, pos_idx, debug_level=0):
        """
        深度优先搜索，保留 top-N 个最佳结果（去重）
        """
        # 所有位置都填满了
        if pos_idx == len(self.positions):
            # 深拷贝 layout，确保每个 solution 都是独立的
            import copy
            current_layout_copy = copy.deepcopy(self.current_layout)

            # 👇 检查是否重复（包括旋转等价）
            if self._is_duplicate_layout(current_layout_copy):
                if debug_level >= 1:
                    print(f"[DEBUG] Skipped duplicate layout, cost={self.current_cost:.4f}")
                return

            # 👇 重要：把当前 layout 的所有等价形式都加入 seen_layouts
            self._mark_layout_as_seen(current_layout_copy)

            # 加入候选结果
            self.best_solutions.append((self.current_cost, current_layout_copy))

            # 按 cost 排序，只保留 top-N
            self.best_solutions.sort(key=lambda x: x[0])
            if len(self.best_solutions) > self.keep_top_n:
                self.best_solutions = self.best_solutions[:self.keep_top_n]
            
            # 打印信息
            self.solutions_found += 1
            rank = next((i for i, (c, _) in enumerate(self.best_solutions, 1) 
                        if c == self.current_cost), None)
            
            if rank == 1:
                print(f"[INFO] 🏆 Found new BEST solution #{self.solutions_found}, cost={self.current_cost:.4f}")
            elif rank and rank <= self.keep_top_n:
                print(f"[INFO] Found solution #{self.solutions_found}, cost={self.current_cost:.4f}, rank={rank}/{self.keep_top_n}")
            else:
                print(f"[INFO] Found solution #{self.solutions_found}, cost={self.current_cost:.4f}, (not in top-{self.keep_top_n})")
            
            return

        r, c = self.positions[pos_idx]

        # 剪枝
        worst_cost_in_top_n = self.best_solutions[-1][0] if len(self.best_solutions) == self.keep_top_n else float('inf')

        for piece_idx in range(self.num_pieces):
            if self.used_piece[piece_idx]:
                continue

            for rot in range(4):
                add_cost = 0.0

                # TOP
                if r > 0 and self.current_layout[r - 1][c] is not None:
                    up_piece_idx, up_rot = self.current_layout[r - 1][c]
                    edge_up_down = (up_rot + 2) % 4
                    edge_cur_top = rot
                    if (piece_idx, edge_cur_top) not in self.candidates[up_piece_idx][edge_up_down]:
                        continue
                    add_cost += self.sim[up_piece_idx, piece_idx, edge_up_down, edge_cur_top]

                # LEFT
                if c > 0 and self.current_layout[r][c - 1] is not None:
                    left_piece_idx, left_rot = self.current_layout[r][c - 1]
                    edge_left_right = (left_rot + 1) % 4
                    edge_cur_left = (rot + 3) % 4
                    if (piece_idx, edge_cur_left) not in self.candidates[left_piece_idx][edge_left_right]:
                        continue
                    add_cost += self.sim[left_piece_idx, piece_idx, edge_left_right, edge_cur_left]
                    
                new_cost = self.current_cost + add_cost

                # 剪枝
                if new_cost >= worst_cost_in_top_n:
                    continue

                # 选择
                self.current_layout[r][c] = (piece_idx, rot)
                self.used_piece[piece_idx] = True
                prev_cost = self.current_cost
                self.current_cost = new_cost

                self._dfs(pos_idx + 1, debug_level=debug_level)

                # 回溯
                self.current_layout[r][c] = None
                self.used_piece[piece_idx] = False
                self.current_cost = prev_cost

    def _mark_layout_as_seen(self, layout):
        """
        将 layout 的所有等价形式标记为已见过
        """
        # 1. 原始 layout
        self.seen_layouts.add(self._layout_to_tuple(layout))
        
        # 2. 所有旋转版本（如果你想去除旋转等价）
        for rot_times in range(1, 4):  # 旋转 90°, 180°, 270°
            rotated = self._rotate_layout_90(layout, rot_times)
            self.seen_layouts.add(self._layout_to_tuple(rotated))
        
        # 3. 镜像版本（可选）
        # flipped_h = self._flip_layout_horizontal(layout)
        # self.seen_layouts.add(self._layout_to_tuple(flipped_h))
        # for rot_times in range(1, 4):
        #     rotated = self._rotate_layout_90(flipped_h, rot_times)
        #     self.seen_layouts.add(self._layout_to_tuple(rotated))

    def _layout_to_tuple(self, layout):
        """
        将 layout 转换为可哈希的 tuple，用于去重
        """
        result = []
        for row in layout:
            row_tuple = []
            for cell in row:
                if cell is not None:
                    piece_idx, rot = cell
                    row_tuple.append((piece_idx, rot))
                else:
                    row_tuple.append((-1, -1))
            result.append(tuple(row_tuple))
        return tuple(result)

    def _is_duplicate_layout(self, layout):
        """
        检查 layout 是否已经存在（考虑旋转和镜像）
        """
        layout_tuple = self._layout_to_tuple(layout)
        
        # 1. 检查完全相同
        if layout_tuple in self.seen_layouts:
            return True
        
        # 2. 检查整体旋转 90°/180°/270°
        for rot_times in range(1, 4):
            rotated = self._rotate_layout_90(layout, rot_times)
            if self._layout_to_tuple(rotated) in self.seen_layouts:
                return True
        
        # 3. 检查镜像（可选）
        # flipped_h = self._flip_layout_horizontal(layout)
        # if self._layout_to_tuple(flipped_h) in self.seen_layouts:
        #     return True
        # for rot_times in range(1, 4):
        #     rotated = self._rotate_layout_90(flipped_h, rot_times)
        #     if self._layout_to_tuple(rotated) in self.seen_layouts:
        #         return True
        
        return False

    def _rotate_layout_90(self, layout, times=1):
        """
        将整个 layout 顺时针旋转 90° * times
        """
        result = [row[:] for row in layout]
        for _ in range(times % 4):
            rows = len(result)
            cols = len(result[0])
            new_layout = [[None] * rows for _ in range(cols)]
            
            for r in range(rows):
                for c in range(cols):
                    if result[r][c] is not None:
                        piece_idx, rot = result[r][c]
                        # 旋转后位置：(r, c) -> (c, rows-1-r)
                        # 块本身也要旋转：rot -> (rot + 1) % 4
                        new_layout[c][rows - 1 - r] = (piece_idx, (rot + 1) % 4)
            
            result = new_layout
        
        return result

    def _flip_layout_horizontal(self, layout):
        """
        水平翻转 layout
        """
        rows = len(layout)
        cols = len(layout[0])
        flipped = [[None] * cols for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                if layout[r][c] is not None:
                    piece_idx, rot = layout[r][c]
                    # 水平翻转：(r, c) -> (r, cols-1-c)
                    # 旋转也要调整（0<->2, 1<->3）
                    new_rot = (4 - rot) % 4 if rot % 2 == 1 else rot
                    flipped[r][cols - 1 - c] = (piece_idx, new_rot)
        
        return flipped

def calculate_layout_cost(layout, sim):
    """
    计算给定 layout 的总 cost

    Args:
        layout: 2D list of (piece_idx, rot_idx) tuples
        sim: similarity matrix [num_pieces, num_pieces, 4, 4]

    Returns:
        total_cost: float
    """
    rows = len(layout)
    cols = len(layout[0])
    total_cost = 0.0

    for r in range(rows):
        for c in range(cols):
            piece_idx, rot = layout[r][c]

            # 检查上方邻居
            if r > 0 and layout[r - 1][c] is not None:
                up_piece_idx, up_rot = layout[r - 1][c]
                # 上方 piece 的下边 (edge 2) 对应当前 piece 的上边 (edge 0)
                edge_up_down = (up_rot + 2) % 4  # 下边 = (rot + 2) % 4
                edge_cur_top = rot  # 上边 = rot
                total_cost += sim[up_piece_idx, piece_idx, edge_up_down, edge_cur_top]

            # 检查左方邻居
            if c > 0 and layout[r][c - 1] is not None:
                left_piece_idx, left_rot = layout[r][c - 1]
                # 左方 piece 的右边 (edge 1) 对应当前 piece 的左边 (edge 3)
                edge_left_right = (left_rot + 1) % 4  # 右边 = (rot + 1) % 4
                edge_cur_left = (rot + 3) % 4  # 左边 = (rot + 3) % 4
                total_cost += sim[left_piece_idx, piece_idx, edge_left_right, edge_cur_left]

    return total_cost

def optimize_layout_by_row_permutation(layout, sim, all_rots, max_rotations=4):
    """
    优化 layout：保持每行的 pieces 不变，尝试所有可能的排列和旋转组合

    Args:
        layout: 2D list of (piece_idx, rot_idx) tuples
        sim: similarity matrix [num_pieces, num_pieces, 4, 4]
        all_rots: list of PieceRot objects
        max_rotations: 每个 piece 尝试的旋转次数 (0-3)

    Returns:
        best_layout: 优化后的 layout
        best_cost: 优化后的 cost
    """
    from itertools import permutations, product
    import copy

    best_layout = copy.deepcopy(layout)
    best_cost = calculate_layout_cost(layout, sim)

    rows = len(layout)
    cols = len(layout[0])

    print(f"\n[OPTIMIZE] Starting row-wise optimization...")
    print(f"[OPTIMIZE] Initial cost: {best_cost:.4f}")

    # 对每一行进行优化
    for row_idx in range(rows):
        print(f"\n[OPTIMIZE] Optimizing row {row_idx}...")

        # 获取这一行的所有 piece_idx（不考虑旋转）
        row_pieces = [layout[row_idx][c][0] for c in range(cols)]

        # 尝试所有可能的排列
        best_row_layout = best_layout[row_idx][:]
        improved = False

        num_perms = 0
        for perm in permutations(row_pieces):
            # 对每个排列，尝试所有旋转组合
            for rotations in product(range(max_rotations), repeat=cols):
                num_perms += 1

                # 创建新的 layout
                test_layout = copy.deepcopy(best_layout)
                test_layout[row_idx] = [(perm[c], rotations[c]) for c in range(cols)]

                # 计算 cost
                test_cost = calculate_layout_cost(test_layout, sim)

                # 如果找到更好的解
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_layout = test_layout
                    best_row_layout = test_layout[row_idx][:]
                    improved = True
                    print(f"  [IMPROVE] Found better layout! New cost: {best_cost:.4f}")

        if improved:
            print(f"  [RESULT] Row {row_idx} optimized. Tried {num_perms} permutations.")
        else:
            print(f"  [RESULT] Row {row_idx} unchanged. Tried {num_perms} permutations.")

    print(f"\n[OPTIMIZE] Final cost: {best_cost:.4f}")
    print(f"[OPTIMIZE] Improvement: {calculate_layout_cost(layout, sim) - best_cost:.4f}")

    return best_layout, best_cost

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
                    canvas[y:y+ph, x:x+pw] = piece_img  # 👈 修复：w -> pw

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
            final_canvas[y:y+ph, x:x+pw] = piece_img  # 👈 修复：w -> pw

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

def main_irregular(input_path, output_image_path, bin_size=10, len_tol=2, use_grow=False, cost_thresh=0.5):
    """
    不规则尺寸的启发式求解入口。
    """
    original_img = load_image(input_path)
    pieces = segment_pieces(original_img)

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return

    rectified_pieces = [rectify_piece(p) for p in pieces]
    if use_grow:
        all_rots = build_all_rotations(rectified_pieces)
        tmp_solver = IrregularPuzzleSolver(rectified_pieces, bin_size=bin_size, len_tol=len_tol)
        sim = tmp_solver.build_similarity()
        grow_solver = GreedyGrowSolver(all_rots, sim, cost_thresh=cost_thresh)
        best_layout, info = grow_solver.solve()
        render_info = {"norm_pos": info.get("norm_pos"), "rot": info.get("rot")}
        solved_image = render_grow_layout(render_info["norm_pos"], render_info["rot"], all_rots)
    else:
        solver = IrregularPuzzleSolver(rectified_pieces, bin_size=bin_size, len_tol=len_tol)
        best_layout, info = solver.solve()
        solved_image = render_irregular_layout(best_layout, solver.all_rots)

    if not best_layout:
        print("[ERROR] Failed to build irregular layout.")
        return

    cv2.imwrite(output_image_path, solved_image)

    print(f"[INFO] Irregular layout saved to {output_image_path}")
    if use_grow:
        print(f"[INFO] GreedyGrow placed {len(best_layout)} rows of pieces.")
    else:
        print(f"[INFO] Row clusters: {info['row_clusters']}")
        print(f"[INFO] Row order: {info['row_sequence']}")
    return best_layout, info


def main(input_path, output_image_path, output_anim_path=None, anim_style='simultaneous', keep_top_n=5):
    """
    主函数
    参数:
    - keep_top_n: 保留 top-N 个最佳结果
    """
    original_img = load_image(input_path)
    pieces = segment_pieces(original_img)

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return
    
    piece_size = pieces[0].shape[:2]
    rectified_pieces = [rectify_piece(p) for p in pieces]

    num_pieces = len(rectified_pieces)
    side = int(round(math.sqrt(num_pieces)))
    grid_rows = side
    grid_cols = side

    all_rots = build_all_rotations(rectified_pieces)

    # 👇 传入 keep_top_n 参数
    solver = PuzzleSolver(all_rots, grid_rows, grid_cols, keep_top_n=keep_top_n)
    best_solutions = solver.solve()

    if not best_solutions:
        print("[ERROR] No layout found.")
        return

    # 👇 对最佳解决方案进行行内优化
    print("\n" + "="*60)
    print("Starting row-wise optimization on best solution...")
    print("="*60)

    best_cost, best_layout = best_solutions[0]
    optimized_layout, optimized_cost = optimize_layout_by_row_permutation(
        best_layout, solver.sim, all_rots, max_rotations=4
    )

    # 如果优化后的结果更好，替换最佳解决方案
    if optimized_cost < best_cost:
        print(f"\n[SUCCESS] Optimization improved cost from {best_cost:.4f} to {optimized_cost:.4f}")
        best_solutions[0] = (optimized_cost, optimized_layout)
    else:
        print(f"\n[INFO] Optimization did not improve the solution (cost remains {best_cost:.4f})")

    # 👇 打印所有 top-N 结果
    print(f"\n{'='*60}")
    print(f"Found {len(best_solutions)} solutions:")
    print(f"{'='*60}")
    
    for i, (cost, layout) in enumerate(best_solutions, 1):
        print(f"  Rank {i}: cost={cost:.4f}")

    # 👇 保存所有 top-N 结果的图片
    base_name = output_image_path.rsplit('.', 1)[0]
    ext = output_image_path.rsplit('.', 1)[1] if '.' in output_image_path else 'png'

    ground_truth = [[(10, 0), (5, 0), (0, 0), (14, 0)], 
                [(15, 0), (9, 0), (11, 0), (8, 0)], 
                [(4, 0), (6, 0), (1, 0), (12, 0)], 
                [(7, 0), (3, 0), (13, 0), (2, 0)]]
    ground_truth_image = render_layout(ground_truth, all_rots)
    cv2.imwrite(f"{base_name}_ground_truth.{ext}", ground_truth_image)
    ground_truth_cost = calculate_layout_cost(ground_truth, solver.sim)
    print(f"[INFO] Saved ground truth layout with cost={ground_truth_cost:.4f} to {base_name}_ground_truth.{ext}")

    # 验证 ground truth 的每条边是否在候选集中
    print("\n[DEBUG] Checking if ground truth edges are in candidate sets:")
    all_edges_in_candidates = True
    rows = len(ground_truth)
    cols = len(ground_truth[0])

    for r in range(rows):
        for c in range(cols):
            piece_idx, rot = ground_truth[r][c]

            # 检查上方邻居
            if r > 0:
                up_piece_idx, up_rot = ground_truth[r - 1][c]
                edge_up_down = (up_rot + 2) % 4
                edge_cur_top = rot
                if (piece_idx, edge_cur_top) not in solver.candidates[up_piece_idx][edge_up_down]:
                    print(f"  ✗ Position ({r},{c}): Piece {up_piece_idx} edge {edge_up_down} -> Piece {piece_idx} edge {edge_cur_top} NOT in candidates")
                    all_edges_in_candidates = False

            # 检查左方邻居
            if c > 0:
                left_piece_idx, left_rot = ground_truth[r][c - 1]
                edge_left_right = (left_rot + 1) % 4
                edge_cur_left = (rot + 3) % 4
                if (piece_idx, edge_cur_left) not in solver.candidates[left_piece_idx][edge_left_right]:
                    print(f"  ✗ Position ({r},{c}): Piece {left_piece_idx} edge {edge_left_right} -> Piece {piece_idx} edge {edge_cur_left} NOT in candidates")
                    all_edges_in_candidates = False

    if all_edges_in_candidates:
        print("  ✓ All ground truth edges are in candidate sets!")
    else:
        print("  ✗ Some ground truth edges are NOT in candidate sets - that's why it wasn't found!")
    print()

    for i, (cost, layout) in enumerate(best_solutions, 1):
        solved_image = render_layout(layout, all_rots)
        
        if i == 1:
            # 第一名用原始文件名
            output_path = output_image_path
        else:
            # 其他结果加后缀
            output_path = f"{base_name}_rank{i}.{ext}"
        
        cv2.imwrite(output_path, solved_image)
        print(f"[INFO] Saved rank {i} solution to {output_path}")

    # 👇 只为最佳结果生成动画
    if output_anim_path is not None:
        best_layout = best_solutions[0][1]
        
        if anim_style == 'sequential' or anim_style == 'cpp':
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
        print("Usage: python solve_resize_problem.py <input_image> <output_image> [output_animation.mp4] [--style=simultaneous|sequential] [--top-n=5]")
        print("Add --irregular to run the irregular heuristic pipeline.")
        print("Optional: --bin-size=<int> (default=10), --len-tol=<int> (default=2), --grow (use greedy region grow), --cost-thresh=<float> (default=0.5)")
        print("\nExamples:")
        print("  python solve_resize_problem.py input.png output.png --top-n=10")
        print("  python solve_resize_problem.py input.png output.png animation.mp4 --top-n=3")
        print("  python solve_resize_problem.py input.png output.png animation.mp4 --style=sequential --top-n=5")
        print("  python solve_resize_problem.py input.png output.png --irregular --bin-size=12 --len-tol=3")
        print("  python solve_resize_problem.py input.png output.png --irregular --grow --cost-thresh=0.45")
    else:
        input_path = sys.argv[1]
        output_image_path = sys.argv[2]
        output_anim_path = sys.argv[3] if len(sys.argv) >= 4 and not sys.argv[3].startswith('--') else None
        anim_style = 'simultaneous'
        keep_top_n = 5  # 默认保留 5 个
        irregular = False
        bin_size = 10
        len_tol = 2
        use_grow = False
        cost_thresh = 0.5

        # Parse arguments
        for arg in sys.argv[3:]:
            if arg.startswith('--style='):
                anim_style = arg.split('=')[1]
            elif arg.startswith('--top-n='):
                keep_top_n = int(arg.split('=')[1])
            elif arg == '--irregular':
                irregular = True
            elif arg.startswith('--bin-size='):
                bin_size = int(arg.split('=')[1])
            elif arg.startswith('--len-tol='):
                len_tol = int(arg.split('=')[1])
            elif arg == '--grow':
                use_grow = True
            elif arg.startswith('--cost-thresh='):
                cost_thresh = float(arg.split('=')[1])

        if irregular:
            if output_anim_path is not None:
                print("[WARNING] Animation is not generated in irregular mode; ignoring animation output argument.")
            main_irregular(
                input_path,
                output_image_path,
                bin_size=bin_size,
                len_tol=len_tol,
                use_grow=use_grow,
                cost_thresh=cost_thresh,
            )
            sys.exit(0)

        if output_anim_path and not output_anim_path.endswith('.mp4'):
            print("[WARNING] Animation output should be .mp4 file")
            output_anim_path += '.mp4' if '.' not in output_anim_path else ''

        main(input_path, output_image_path, output_anim_path, anim_style=anim_style, keep_top_n=keep_top_n)
