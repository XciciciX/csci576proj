import cv2
import numpy as np
import math
from collections import namedtuple
import itertools
import sys
import os

# ---- 配置参数 ----
EDGE_STRIP_WIDTH = 10        # 用于提取边缘条带的宽度 (像素)
COLOR_BINS = 10              # HSV 每个通道的 bin 数
GRAD_BINS = 10               # 梯度方向直方图 bin 数
ALPHA = 0.5                 # 颜色差权重
BETAB = 0.5                 # 梯度差权重
MIN_COMPONENT_AREA = 10    # 过滤太小的噪声连通域
# MAX_SEARCH_SOLUTIONS = 100000    # 只找一个最优解，够用了

# 一个简单的结构体保存 piece 的旋转版本信息
PieceRot = namedtuple("PieceRot", ["piece_idx", "rot_idx", "img", "edges"])


# ---------- 工具函数 ----------

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def segment_pieces(image):
    """
    从黑底大图中分割出每个 puzzle piece。
    假设背景接近纯黑，拼图是彩色块。
    返回：列表 [piece_img, ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 简单阈值分离前景（拼图块）和背景
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 去噪一下
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    pieces = []
    for label in range(1, num_labels):  # 0 是背景
        x, y, w, h, area = stats[label]
        if area < MIN_COMPONENT_AREA:
            continue
        piece = image[y:y + h, x:x + w].copy()
        pieces.append(piece)

    print(f"[INFO] Detected {len(pieces)} pieces.")
    return pieces


def normalize_pieces(pieces):
    """
    把所有 piece 调整到相同大小（简单 resize）。
    这里用中位数宽高作为目标尺寸。
    返回：normalized_pieces, (target_h, target_w)
    """
    hs = [p.shape[0] for p in pieces]
    ws = [p.shape[1] for p in pieces]
    target_h = int(np.median(hs))
    target_w = int(np.median(ws))

    norm_pieces = []
    for p in pieces:
        resized = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
        norm_pieces.append(resized)

    print(f"[INFO] Normalized piece size to {target_w}x{target_h}")
    return norm_pieces, (target_h, target_w)


def rotate_piece(img, rot_idx):
    """
    rot_idx = 0,1,2,3 分别对应 0°,90°,180°,270°
    """
    k = rot_idx % 4
    return np.rot90(img, k, axes=(1, 0))  # 逆时针 90*k（注意 axes）


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
    edges["top"] = {
        "color": compute_color_hist(top_strip),
        "grad": compute_grad_hist(gray[0:k, :], top_mag, top_ang)
    }
    # bottom
    edges["bottom"] = {
        "color": compute_color_hist(bottom_strip),
        "grad": compute_grad_hist(gray[h - k:h, :], bottom_mag, bottom_ang)
    }
    # left
    edges["left"] = {
        "color": compute_color_hist(left_strip),
        "grad": compute_grad_hist(gray[:, 0:k], left_mag, left_ang)
    }
    # right
    edges["right"] = {
        "color": compute_color_hist(right_strip),
        "grad": compute_grad_hist(gray[:, w - k:w], right_mag, right_ang)
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



def edge_distance(descA, descB, alpha=ALPHA, beta=BETAB):
    color_diff = color_distance(descA["color"], descB["color"])
    grad_diff  = grad_distance(descA["grad"],  descB["grad"])
    return alpha * color_diff + beta * grad_diff



# ---------- 构建所有旋转版本 ----------

def build_all_rotations(norm_pieces):
    """
    对每个 piece 生成 4 个旋转版本，并计算每个版本的 edge 描述子。
    返回：
        all_rots: (piece_idx, rot_idx) -> PieceRot
    """
    all_rots = {}
    for i, p in enumerate(norm_pieces):
        for rot in range(4):
            img_rot = rotate_piece(p, rot)
            edges = compute_piece_edge_descriptors(img_rot)
            all_rots[(i, rot)] = PieceRot(piece_idx=i, rot_idx=rot, img=img_rot, edges=edges)
    print(f"[INFO] Built {len(all_rots)} rotated versions.")
    return all_rots


# ---------- 布局搜索（DFS + 剪枝） ----------

class PuzzleSolver:
    def __init__(self, all_rots, grid_rows, grid_cols):
        self.all_rots = all_rots               # dict[(piece_idx, rot_idx)] -> PieceRot
        self.num_pieces = len(set(i for (i, _) in all_rots.keys()))
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

    def solve(self):
        self._dfs(0)
        return self.best_layout, self.best_cost

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
            for rot in [0]:
                key = (piece_idx, rot)
                if key not in self.all_rots:
                    continue

                piece_rot = self.all_rots[key]

                # 只考虑与已放好的“上”和“左”的匹配代价
                add_cost = 0.0

                # 上方
                if r > 0 and self.current_layout[r - 1][c] is not None:
                    up_piece_idx, up_rot = self.current_layout[r - 1][c]
                    up_rot_obj = self.all_rots[(up_piece_idx, up_rot)]
                    add_cost += edge_distance(
                        up_rot_obj.edges["bottom"],
                        piece_rot.edges["top"]
                    )

                # 左方
                if c > 0 and self.current_layout[r][c - 1] is not None:
                    left_piece_idx, left_rot = self.current_layout[r][c - 1]
                    left_rot_obj = self.all_rots[(left_piece_idx, left_rot)]
                    add_cost += edge_distance(
                        left_rot_obj.edges["right"],
                        piece_rot.edges["left"]
                    )

                new_cost = self.current_cost + add_cost

                # branch-and-bound 剪枝：当前 partial cost 已经 >= best，就没必要继续
                if new_cost >= self.best_cost:
                    continue
                    
                if add_cost > 0.8:
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



# ---------- 重建图片 & 简易动画 ----------

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
            piece_rot = all_rots[(piece_idx, rot_idx)]
            img = piece_rot.img
            # 确保大小相同
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            y0 = r * ph
            x0 = c * pw
            canvas[y0:y0 + ph, x0:x0 + pw, :] = img

    return canvas


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


# ---------- 主入口 ----------

def main(input_path, output_image_path, output_anim_dir=None):
    img = load_image(input_path)
    pieces = segment_pieces(img)
    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return

    norm_pieces, piece_size = normalize_pieces(pieces)

    # 默认假设是正方形布局：rows = cols = sqrt(N)
    num_pieces = len(norm_pieces)
    side = int(round(math.sqrt(num_pieces)))
    if side * side != num_pieces:
        print(f"[WARN] Number of pieces is {num_pieces}, not a perfect square. "
              "You may need to specify grid size manually.")
    grid_rows = side
    grid_cols = side

    all_rots = build_all_rotations(norm_pieces)

    solver = PuzzleSolver(all_rots, grid_rows, grid_cols)
    best_layout, best_cost = solver.solve()

    if best_layout is None:
        print("[ERROR] No layout found.")
        return

    print(f"[INFO] Best layout cost: {best_cost:.4f}")

    solved_image = render_layout(best_layout, all_rots, piece_size)
    cv2.imwrite(output_image_path, solved_image)
    print(f"[INFO] Saved solved puzzle image to {output_image_path}")

    if output_anim_dir is not None:
        os.makedirs(output_anim_dir, exist_ok=True)
        frames = render_animation_sequence(best_layout, all_rots, piece_size)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_anim_dir, f"frame_{i:03d}.png")
            cv2.imwrite(frame_path, frame)
        print(f"[INFO] Saved {len(frames)} animation frames to {output_anim_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python puzzle_solver.py <input_image> <output_image> [output_anim_dir]")
    else:
        input_path = sys.argv[1]
        output_image_path = sys.argv[2]
        output_anim_dir = sys.argv[3] if len(sys.argv) >= 4 else None
        main(input_path, output_image_path, output_anim_dir)
