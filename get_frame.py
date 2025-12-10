#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import random
from math import gcd
from functools import reduce
from typing import List, Tuple, Dict, Optional



def solve_packing(
    canvas_h: int,
    canvas_w: int,
    pieces: List[Tuple[int, int]],
    max_solutions: Optional[int] = 10,
    allow_rotate: bool = True,
) -> List[List[Dict]]:
    """
    在一个 H×W 的整数网格画布上，铺一组长方形块，返回所有可行铺法（不重叠、刚好铺满）。

    这里做了去重：
    - 对于尺寸相同的 piece，只关心“格子上矩形尺寸的排布”是否不同。
    - 也就是说，如果两个解只是交换了相同尺寸的 index，会被视为同一个 frame，只保留一个。

    同时在 DFS 中加入一个基于“同高/同宽优先”的 heuristic：
    - 在格子 (r,c) 放块时：
        * 如果左边已有块，则优先尝试“高度 = 左边块高度”的 piece；
        * 如果上边已有块，则优先尝试“宽度 = 上边块宽度”的 piece；
    """

    n = len(pieces)
    total_area = sum(h * w for h, w in pieces)
    canvas_area = canvas_h * canvas_w
    if total_area != canvas_area:
        print(f"[WARN] total piece area ({total_area}) != canvas area ({canvas_area}), "
              "可能不存在完全铺满的解。")

    # 画布：-1 表示空，>=0 表示对应的 piece_index
    canvas = [[-1] * canvas_w for _ in range(canvas_h)]
    used = [False] * n
    # 记录当前每个 piece 放置时的 (h, w)，用于生成解和 frame
    current_orient: List[Optional[Tuple[int, int]]] = [None] * n
    solutions: List[List[Dict]] = []

    # 用于“按 frame 去重”的集合
    # 元素是：tuple(tuple(row), ...) 形式的 type_id 网格
    seen_frames = set()

    # ====== 预计算：按原始尺寸分组 ======
    height_to_indices: Dict[int, List[int]] = {}
    width_to_indices: Dict[int, List[int]] = {}
    for idx, (ph, pw) in enumerate(pieces):
        height_to_indices.setdefault(ph, []).append(idx)
        width_to_indices.setdefault(pw, []).append(idx)

    def find_empty():
        """找到第一个空格子 (r, c)，找不到则返回 (None, None)"""
        for r in range(canvas_h):
            for c in range(canvas_w):
                if canvas[r][c] == -1:
                    return r, c
        return None, None

    def can_place(idx: int, r: int, c: int, h: int, w: int) -> bool:
        """检查 piece idx 放在 (r,c) 顶点、高度 h、宽度 w 是否会出界或重叠"""
        if r + h > canvas_h or c + w > canvas_w:
            return False
        for i in range(r, r + h):
            row = canvas[i]
            for j in range(c, c + w):
                if row[j] != -1:
                    return False
        return True

    def place(idx: int, r: int, c: int, h: int, w: int, val: int):
        """在 canvas 上填充/清空某个块"""
        for i in range(r, r + h):
            for j in range(c, c + w):
                canvas[i][j] = val

    def build_frame_signature() -> Tuple[Tuple[int, ...], ...]:
        """
        基于当前 canvas + current_orient，构建“按尺寸的 frame 网格签名”。
        """
        shape_to_id: Dict[Tuple[int, int], int] = {}
        next_id = 0

        frame_grid = [[-1] * canvas_w for _ in range(canvas_h)]

        for y in range(canvas_h):
            for x in range(canvas_w):
                idx = canvas[y][x]
                if idx == -1:
                    frame_grid[y][x] = -1
                else:
                    h, w = current_orient[idx]
                    key = (h, w)
                    if key not in shape_to_id:
                        shape_to_id[key] = next_id
                        next_id += 1
                    frame_grid[y][x] = shape_to_id[key]

        return tuple(tuple(row) for row in frame_grid)

    def dfs():
        # 控制解的数量（按“不同 frame”来数）
        if max_solutions is not None and len(solutions) >= max_solutions:
            return

        r, c = find_empty()
        # 没有空格子了 → 找到一种完整拼法
        if r is None:
            sig = build_frame_signature()
            if sig in seen_frames:
                return
            seen_frames.add(sig)

            sol: List[Dict] = []
            for i in range(n):
                h, w = current_orient[i]
                # 找这个 piece 在 canvas 上的左上角
                tl = None
                for y in range(canvas_h):
                    for x in range(canvas_w):
                        if canvas[y][x] == i:
                            tl = (y, x)
                            break
                    if tl is not None:
                        break
                sol.append(
                    {
                        "piece_index": i,
                        "top": tl[0],
                        "left": tl[1],
                        "height": h,
                        "width": w,
                    }
                )
            solutions.append(sol)
            return

        # ========= 根据左/上邻居，生成“优先考虑的 piece 顺序” =========
        left_h = left_w = None
        top_h = top_w = None

        # 左邻居
        if c > 0 and canvas[r][c - 1] != -1:
            idx_left = canvas[r][c - 1]
            left_h, left_w = current_orient[idx_left]

        # 上邻居
        if r > 0 and canvas[r - 1][c] != -1:
            idx_top = canvas[r - 1][c]
            top_h, top_w = current_orient[idx_top]

        ordered_indices: List[int] = []
        seen_idx = set()

        # 1) 优先：原始高度 = left_h 的块
        if left_h is not None:
            for idx in height_to_indices.get(left_h, []):
                if not used[idx] and idx not in seen_idx:
                    ordered_indices.append(idx)
                    seen_idx.add(idx)

        # 2) 其次：原始宽度 = top_w 的块
        if top_w is not None:
            for idx in width_to_indices.get(top_w, []):
                if not used[idx] and idx not in seen_idx:
                    ordered_indices.append(idx)
                    seen_idx.add(idx)

        # 3) 最后：其他所有未使用的块
        for idx in range(n):
            if not used[idx] and idx not in seen_idx:
                ordered_indices.append(idx)
                seen_idx.add(idx)

        # 本格子“尺寸去重”：同样尺寸 (h,w) 在这个格子只尝试一次
        seen_shapes_this_cell = set()

        # 按 ordered_indices 的顺序 DFS
        for idx in ordered_indices:
            ph, pw = pieces[idx]

            # 决定这个 piece 的所有可选朝向
            if allow_rotate and ph != pw:
                orientations = [(ph, pw), (pw, ph)]
            else:
                orientations = [(ph, pw)]

            for h, w in orientations:
                if not can_place(idx, r, c, h, w):
                    continue

                shape_key = (h, w)
                if shape_key in seen_shapes_this_cell:
                    continue
                seen_shapes_this_cell.add(shape_key)

                # 放下去
                used[idx] = True
                current_orient[idx] = (h, w)
                place(idx, r, c, h, w, idx)

                # 递归
                dfs()

                # 回溯
                place(idx, r, c, h, w, -1)
                used[idx] = False
                current_orient[idx] = None

    dfs()
    return solutions



# ---------- 把一个解画成“彩色矩形示意图”（只看 frame 排列） ----------

def visualize_layout(solution, canvas_h, canvas_w, scale=10, draw_border=True):
    """
    把 solve_packing 返回的某个 solution 画成一张图片。
    不用真实拼图，只画彩色矩形，重点是看“大小排列方式”。

    solution : solve_packing 返回的某个解（solutions[k]）
    canvas_h, canvas_w : 画布尺寸（单位：格子）
    scale   : 每个“格子”放大到多少像素（越大图越清晰）
    draw_border : 是否在每块外面画边框
    """
    H = canvas_h * scale
    W = canvas_w * scale
    img = np.full((H, W, 3), 255, dtype=np.uint8)  # 白底

    # 给每个 piece_index 随机一个颜色
    colors = {}
    rng = random.Random(42)

    for p in solution:
        idx = p["piece_index"]
        top = p["top"]
        left = p["left"]
        h = p["height"]
        w = p["width"]

        if idx not in colors:
            colors[idx] = (
                rng.randint(50, 230),
                rng.randint(50, 230),
                rng.randint(50, 230),
            )
        color = colors[idx]

        # 放大到像素坐标
        y0 = top * scale
        x0 = left * scale
        y1 = (top + h) * scale
        x1 = (left + w) * scale

        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, thickness=-1)

        if draw_border:
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), thickness=1)

    return img


# ---------- 用一张 input 图 → 做 frame packing ----------

def _gcd_list(nums: List[int]) -> int:
    return reduce(gcd, nums)


def run_frame_packing_for_image(
    input_path: str,
    canvas_h_pixels: int,
    canvas_w_pixels: int,
    max_solutions: int = 5,
    allow_rotate: bool = True,
    out_dir: str = "frame_packing_vis",
):
    """
    用一张 input 图片：
      1. load_image + segment_pieces 得到所有拼图块
      2. 只取每块的 (h, w) 尺寸作为 frame
      3. 把像素尺寸离散化到网格（用 gcd 做 cell size）
      4. solve_packing 做几何拼板
      5. 把每个可行解画成“彩色矩形示意图”（只看 frame 排列）

    参数：
        input_path       : 黑底多块矩形拼图的图片路径
        canvas_h_pixels,
        canvas_w_pixels  : 画布的像素高度和宽度（你已知的完整图尺寸）
        max_solutions    : 最多枚举多少种拼法
        allow_rotate     : 是否允许 90° 旋转
        out_dir          : 输出示意图存放目录
    """
    # 1. 读图 + 分割
    img = load_image(input_path)
    pieces = segment_pieces(img)   # [piece_img, ...]

    if len(pieces) == 0:
        print("[ERROR] No pieces detected.")
        return

    # 2. 只取像素尺寸 (h, w)
    piece_sizes_px = [(p.shape[0], p.shape[1]) for p in pieces]
    print("[INFO] piece_sizes_px =", piece_sizes_px)

    hs = [h for h, w in piece_sizes_px]
    ws = [w for h, w in piece_sizes_px]

    # 用 gcd 把像素转成“格子单位”
    gcd_h = _gcd_list(hs)
    gcd_w = _gcd_list(ws)

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
    solutions = solve_packing(
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        pieces=piece_sizes_grid,
        max_solutions=max_solutions,
        allow_rotate=allow_rotate
    )
    print(f"[INFO] Found {len(solutions)} geometric layout(s).")

    if len(solutions) == 0:
        print("[WARN] No feasible packing found.")
        return

    # 4. 画出每一个解的“frame 排列图”
    os.makedirs(out_dir, exist_ok=True)

    for k, sol in enumerate(solutions):
        print(f"\n--- 解 {k} ---")
        for p in sol:
            print(
                f"piece {p['piece_index']} at (top={p['top']}, left={p['left']}), "
                f"size={p['height']}x{p['width']} (grid)"
            )

        vis = visualize_layout(sol, canvas_h, canvas_w, scale=20, draw_border=True)
        out_path = os.path.join(out_dir, f"frame_layout_{k:02d}.png")
        cv2.imwrite(out_path, vis)
        print(f"[INFO] Saved frame layout #{k} to {out_path}")
