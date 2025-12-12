import numpy as np
from compute_similarity import edge_distance, compute_piece_edge_descriptors
from collections import namedtuple

# Store info of a piece in a specific rotation
PieceRot = namedtuple("PieceRot", ["piece_idx", "img", "edges", "shape"])

def rotate_piece(img, rot_idx):
    """
    rot_idx = 0,1,2,3 分别对应 0°,270°,180°,90°
    顺时针旋转
    """
    return np.rot90(img, rot_idx, axes=(0, 1))

def edge_len(shape, edge_idx):
    h, w = shape[:2]
    # 0 top, 1 right, 2 bottom, 3 left
    return w if edge_idx in (0, 2) else h


def crop_background(img: np.ndarray, bg_color=(0, 0, 0), tol: int = 0) -> np.ndarray:
    """
    裁掉四周与背景相同的区域（默认黑边）。
    tol: 容差，>0 时允许一点点噪声（比如压缩/插值导致的非0边缘）
    """
    if img is None or img.size == 0:
        return img

    bg = np.array(bg_color, dtype=img.dtype).reshape(1, 1, 3)
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16))  # 防止 uint8 下溢
    mask = (diff.max(axis=2) > tol)  # True 表示“不是背景”的像素

    if not mask.any():
        return img  # 全是背景，啥也裁不了

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy()

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

def max_rows_for_single_col(
        pieces,          # 原始 pieces: List[np.ndarray]
        idx_list,        # same_lists: List[int]
        max_H=400,
        max_W=400,
        sort_key="height_desc"  # 或 "width_desc" / "area_desc"
    ):
        # 收集 (idx, h, w)
        triples = []
        for idx in idx_list:
            h, w = pieces[idx].shape[:2]
            triples.append((idx, h, w))

        # 你可以换排序策略：让更“大”的先放，能更快剪枝
        if sort_key == "height_desc":
            triples.sort(key=lambda x: x[1], reverse=True)
        elif sort_key == "width_desc":
            triples.sort(key=lambda x: x[2], reverse=True)
        elif sort_key == "area_desc":
            triples.sort(key=lambda x: x[1]*x[2], reverse=True)

        used = []
        cur_H = 0
        cur_W = 0  # 单列时宽度是 max(w)

        for idx, h, w in triples:
            new_H = cur_H + h
            new_W = max(cur_W, w)
            if new_H > max_H or new_W > max_W:
                break
            used.append(idx)
            cur_H = new_H
            cur_W = new_W

        return used, (cur_H, cur_W)


import numpy as np
import cv2
from typing import List, Tuple

def split_image_bisect_until_max(
    img: np.ndarray,
    max_size: int = 400,
    overlap: int = 20,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    递归二分裁剪：把 img 拆成多张 <= max_size 的小图。
    返回 [(sub_img, (y_offset, x_offset)), ...]
    overlap: 两块之间重叠像素，减少把 piece 切断的概率。
    """
    H, W = img.shape[:2]
    out: List[np.ndarray] = []

    def rec(cur: np.ndarray, oy: int, ox: int):
        h, w = cur.shape[:2]
        if h <= max_size and w <= max_size:
            out.append(cur)
            return

        # 沿着更长的维度切
        if h >= w:
            cut = h // 2
            y0 = max(0, cut - overlap)
            y1 = min(h, cut + overlap)

            top = cur[:y1, ...]          # [0, y1)
            bot = cur[y0:, ...]          # [y0, h)

            rec(top, oy + 0,   ox + 0)
            rec(bot, oy + y0,  ox + 0)
        else:
            cut = w // 2
            x0 = max(0, cut - overlap)
            x1 = min(w, cut + overlap)

            left = cur[:, :x1, ...]      # [0, x1)
            right = cur[:, x0:, ...]     # [x0, w)

            rec(left,  oy + 0,  ox + 0)
            rec(right, oy + 0,  ox + x0)

    rec(img, 0, 0)
    return out


def split_until_all_segmented_pieces_under_max(
    img: np.ndarray,
    segment_fn,
    max_size: int = 400,
    overlap: int = 20,
) -> List[np.ndarray]:
    """
    输入一张大图：
    - 若长/宽 > max_size，先递归二分裁成 <= max_size 的小图
    - 每张小图跑 segment_fn，收集 pieces
    返回：pieces(list of np.ndarray)，保证每个 piece 的 h,w 都 <= max_size
    """
    tiles = split_image_bisect_until_max(img, max_size=max_size, overlap=overlap)

    all_pieces: List[np.ndarray] = []
    for tile, (oy, ox) in tiles:
        # 你已有的 segment_pieces(tile)
        pieces = segment_fn(tile)

        # 这里按需过滤掉“贴边的连通域碎片”（可选）
        # 如果你 segment_fn 没返回 bbox，就先不做这个过滤
        for p in pieces:
            ph, pw = p.shape[:2]
            if ph <= max_size and pw <= max_size:
                all_pieces.append(p)

    return all_pieces
