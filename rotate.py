import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

def _cluster_1d_by_tol(values: List[int], tol: int = 3) -> List[int]:
    """
    把 1D 长度按容差 tol 聚类，返回每簇中心（median）。
    例：values=[124,125,126, 147,146] tol=2 -> centers ~ [125,146]
    """
    if not values:
        return []
    v = sorted(int(x) for x in values)
    clusters = []
    cur = [v[0]]
    for x in v[1:]:
        if abs(x - cur[-1]) <= tol:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    centers = [int(np.median(c)) for c in clusters]
    return centers

def _snap_to_centers(x: int, centers: List[int]) -> int:
    if not centers:
        return int(x)
    return int(min(centers, key=lambda c: abs(c - x)))

def normalize_piece_sizes_global(
    pieces: List[np.ndarray],
    tol: int = 3,
    min_size: int = 1,
    save_debug_dir: Optional[str] = None,
) -> Tuple[List[np.ndarray], Dict[str, List[int]], List[Tuple[Tuple[int,int], Tuple[int,int]]]]:
    """
    方法B：先统计所有 piece 的 short/long 边长簇，再把每块 snap 到最近簇中心，并 resize。

    返回：
      - snapped_pieces: 对齐后的图片列表
      - centers: {"short": [...], "long": [...]}
      - mapping: [((h0,w0),(h1,w1)), ...] 原尺寸 -> 新尺寸
    """
    if not pieces:
        return [], {"short": [], "long": []}, []

    # 1) 统计 short/long
    shorts, longs = [], []
    sizes = []
    for img in pieces:
        h, w = img.shape[:2]
        sizes.append((h, w))
        shorts.append(min(h, w))
        longs.append(max(h, w))

    # 2) 聚类得到中心
    short_centers = _cluster_1d_by_tol(shorts, tol=tol)
    long_centers  = _cluster_1d_by_tol(longs,  tol=tol)

    centers = {"short": short_centers, "long": long_centers}

    # 3) 对每块 snap 并 resize（保持横/竖朝向不变）
    snapped = []
    mapping = []

    if save_debug_dir is not None:
        os.makedirs(save_debug_dir, exist_ok=True)

    for i, img in enumerate(pieces):
        h0, w0 = img.shape[:2]
        short0 = min(h0, w0)
        long0  = max(h0, w0)

        short1 = _snap_to_centers(short0, short_centers)
        long1  = _snap_to_centers(long0,  long_centers)

        # 防御：避免 0
        short1 = max(int(short1), min_size)
        long1  = max(int(long1),  min_size)

        # 保持当前朝向：横图(w>=h)就让 W=long, H=short；竖图反之
        if w0 >= h0:
            h1, w1 = short1, long1
        else:
            h1, w1 = long1, short1

        # 选择插值方式：放大用CUBIC，缩小用AREA
        interp = cv2.INTER_CUBIC if (h1 > h0 or w1 > w0) else cv2.INTER_AREA
        out = cv2.resize(img, (w1, h1), interpolation=interp)

        snapped.append(out)
        mapping.append(((h0, w0), (h1, w1)))

        if save_debug_dir is not None:
            cv2.imwrite(os.path.join(save_debug_dir, f"piece_{i:03d}_orig_{h0}x{w0}.png"), img)
            cv2.imwrite(os.path.join(save_debug_dir, f"piece_{i:03d}_snap_{h1}x{w1}.png"), out)

    return snapped, centers, mapping
