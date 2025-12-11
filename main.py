import cv2
import numpy as np
import math
from collections import namedtuple
import itertools
import sys
import os

# ---- 配置参数 ----
EDGE_STRIP_WIDTH = 10        # 用于提取边缘条带的宽度 (像素)
COLOR_BINS = 8              # HSV 每个通道的 bin 数
GRAD_BINS = 8               # 梯度方向直方图 bin 数
ALPHA = 0.5                 # 颜色差权重
BETAB = 0.5                 # 梯度差权重
MIN_COMPONENT_AREA = 10    # 过滤太小的噪声连通域
# MAX_SEARCH_SOLUTIONS = 100000    # 只找一个最优解，够用了

# Store info of a piece in a specific rotation
PieceRot = namedtuple("PieceRot", ["piece_idx", "img", "edges", "edge_lengths"])



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

        # Calculate edge lengths
        # edges: {0: top, 1: right, 2: bottom, 3: left}
        h, w = p.shape[:2]
        edge_lengths = {
            0: w,  # top edge length = width
            1: h,  # right edge length = height
            2: w,  # bottom edge length = width
            3: h   # left edge length = height
        }

        all_rots.append(PieceRot(piece_idx=i, img=p, edges=edges, edge_lengths=edge_lengths))
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
        self._build_candidates(top_k=6)
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

    def _build_candidates(self, top_k=5, edge_length_tolerance=2):
        """
        对于每条边 (i, edge_i)，选出 cost 最小的 top_k 个 (j, edge_j)。
        只考虑边缘长度兼容的边（length差距在tolerance以内）。
        candidates[i][edge_i] 是一个 set，元素是 (j, edge_j)。
        """
        num_pieces = self.num_pieces
        candidates = [[set() for _ in range(4)] for _ in range(num_pieces)]

        empty_candidate_count = 0

        for i in range(num_pieces):
            pi = self.all_rots[i]
            for edge_i in range(4):
                # 收集所有 (j, edge_j, cost)，但只考虑边缘长度兼容的
                triplets = []
                edge_i_length = pi.edge_lengths[edge_i]

                for j in range(num_pieces):
                    if i == j:
                        continue
                    pj = self.all_rots[j]
                    for edge_j in range(4):
                        edge_j_length = pj.edge_lengths[edge_j]

                        # 只有边缘长度兼容时才考虑
                        if abs(edge_i_length - edge_j_length) <= edge_length_tolerance:
                            cost = self.sim[i, j, edge_i, edge_j]
                            triplets.append((cost, j, edge_j))

                # 按 cost 排序，取前 top_k
                triplets.sort(key=lambda x: x[0])
                for t in triplets[:top_k]:
                    _, j, edge_j = t
                    candidates[i][edge_i].add((j, edge_j))

                # 检查是否有空候选
                if len(candidates[i][edge_i]) == 0:
                    empty_candidate_count += 1

        self.candidates = candidates
        print(f"[INFO] Built candidate neighbor sets with top_k={top_k}, edge_length_tolerance={edge_length_tolerance}")
        if empty_candidate_count > 0:
            print(f"[WARNING] {empty_candidate_count} edges have no compatible candidates!")

        
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
            piece_rot = all_rots[piece_idx]
            img = rotate_piece(piece_rot.img, rot_idx)

            # 确保大小相同
            img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            # ph, pw = img.shape[:2]
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


# ---------- 主入口 ----------

def main(input_path, output_image_path, output_anim_dir=None):
    img = load_image(input_path)
    pieces = segment_pieces(img)

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return
    piece_size = pieces[0].shape[:2]  # 假设所有 piece 大小相同
    rectified_pieces = [rectify_piece(p) for p in pieces]
    
    

    # 默认假设是正方形布局：rows = cols = sqrt(N)

    # TODO: change this

    # "The image sizes will be same as the test samples you have"
    # 
    # 找出所有 (r, c)，使得 r * c == num_pieces

    # 对每个 (r, c) 都跑一次 PuzzleSolver 选 cost 最小的那个

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
