from typing import List, Tuple, Dict, Optional
import numpy as np

# Heuristic; 1. same height/width within tol are considered compatible for adjacency
# Heuristic; 2. two pieces top k similarity

def dims_compatible_for_neighbor(p1, p2, tol=2):
    h1, w1 = p1
    h2, w2 = p2

    # 横向相邻（左右） → 高度要差不多
    same_height = abs(h1 - h2) <= tol
    # 纵向相邻（上下） → 宽度要差不多
    same_width  = abs(w1 - w2) <= tol

    return same_height or same_width


def compute_pair_lb(similarities: np.ndarray) -> np.ndarray:
    """
    similarities: (N, N, 4, 4)
    pair_lb[i, j] = min over all edge pairs 的最小 cost
    """
    pair_lb = np.min(similarities, axis=(2, 3))  # (N, N)
    N = pair_lb.shape[0]
    for i in range(N):
        pair_lb[i, i] = np.inf  # 自己和自己不用配
    return pair_lb


def shapes_compatible_for_rectangle(p1: Tuple[int, int],
                                    p2: Tuple[int, int]) -> bool:
    """
    判断两个 piece 尺寸是否“有可能”通过某种旋转后合成一个大矩形：
    也就是它们的高/宽里，有至少一个维度可以对齐。
    p1, p2: (h, w)
    """
    h1, w1 = p1
    h2, w2 = p2
    for a in (h1, w1):
        for b in (h2, w2):
            if a == b:
                return True
    return False


def build_strong_pairs_topk_for_frame(
    pair_lb: np.ndarray,
    pieces: List[Tuple[int, int]],
    per_piece_k: int = 2,
    max_pairs: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """
    从 pair_lb 中选“强配对”：
    - 对每个 i，找与它尺寸兼容且 pair_lb 最小的 per_piece_k 个 j
    - 合并成无向对 (i,j)，去重后按 cost 排序
    - 可选截断到 max_pairs

    返回:
        strong_pairs: [(i, j, cost), ...]   其中 i < j
    """
    N = pair_lb.shape[0]
    pair_best: Dict[Tuple[int, int], float] = {}

    for i in range(N):
        # 先筛尺寸兼容的 j
        neighbors = []
        for j in range(N):
            if i == j:
                continue
            if not np.isfinite(pair_lb[i, j]):
                continue
            # 尺寸兼容：有至少一个维度可以对齐（考虑旋转）
            if not shapes_compatible_for_rectangle(pieces[i], pieces[j]):
                continue
            neighbors.append((pair_lb[i, j], j))

        neighbors.sort(key=lambda x: x[0])
        for cost, j in neighbors[:per_piece_k]:
            a, b = sorted((i, j))
            key = (a, b)
            if key not in pair_best or cost < pair_best[key]:
                pair_best[key] = cost

    strong_pairs = [(i, j, cost) for (i, j), cost in pair_best.items()]
    strong_pairs.sort(key=lambda x: x[2])

    if max_pairs is not None and len(strong_pairs) > max_pairs:
        strong_pairs = strong_pairs[:max_pairs]

    return strong_pairs


def check_strong_pairs_adjacency(
    solution: List[Dict],
    strong_pairs: List[Tuple[int, int, float]],
) -> bool:
    """
    要求：对每个强配对 (i,j)，在 frame 里的摆放必须是：
    - 左右刚好拼成一个大矩形，或者
    - 上下刚好拼成一个大矩形

    solution: solve_packing 输出的单个 frame 解
              每个元素包含 piece_index, top, left, height, width
    """
    boxes = {p["piece_index"]: p for p in solution}

    def forms_big_rectangle(A: Dict, B: Dict) -> bool:
        At, Al, Ah, Aw = A["top"], A["left"], A["height"], A["width"]
        Bt, Bl, Bh, Bw = B["top"], B["left"], B["height"], B["width"]
        Ab, Ar = At + Ah, Al + Aw
        Bb, Br = Bt + Bh, Bl + Bw

        # 左右相邻且高度完全一致（合成一个宽为 Aw+Bw 的大矩形）
        # A 在左，B 在右
        left_right_1 = (Ar == Bl) and (At == Bt) and (Ab == Bb)
        # B 在左，A 在右
        left_right_2 = (Br == Al) and (At == Bt) and (Ab == Bb)

        # 上下相邻且宽度完全一致（合成一个高为 Ah+Bh 的大矩形）
        # A 在上，B 在下
        top_bottom_1 = (Ab == Bt) and (Al == Bl) and (Ar == Br)
        # B 在上，A 在下
        top_bottom_2 = (Bb == At) and (Al == Bl) and (Ar == Br)

        return left_right_1 or left_right_2 or top_bottom_1 or top_bottom_2

    for i, j, _ in strong_pairs:
        A = boxes[i]
        B = boxes[j]
        if not forms_big_rectangle(A, B):
            return False
    return True


def solve_packing(
    canvas_h: int,
    canvas_w: int,
    pieces: List[Tuple[int, int]],
    max_solutions: Optional[int] = 10,
    allow_rotate: bool = True,
    similarities: Optional[np.ndarray] = None,
    per_piece_k: int = 2,
    max_strong_pairs: Optional[int] = None,
) -> List[List[Dict]]:
    """
    在一个 H×W 的整数网格画布上，铺一组长方形块，返回所有可行铺法（不重叠、刚好铺满）。

    这里做了两类去重 + 剪枝：
    - 对于尺寸相同的 piece，只关心“格子上矩形尺寸的排布”是否不同（frame 去重）。
    - 如果提供了 similarities (N,N,4,4)，会从中抽取 top-k 的“强配对”，
      要求每个强配对在 frame 中必须以“合成大矩形”的方式相邻，否则丢弃该 frame。

    参数：
        canvas_h, canvas_w : 画布高度 / 宽度（单位：格子）
        pieces             : 每块拼图的尺寸列表 [(h1, w1), (h2, w2), ...]（单位：格子）
        max_solutions      : 最多返回多少种“不同 frame”拼法（None 表示不限制）
        allow_rotate       : 是否允许拼图块旋转 90°（交换 h / w）
        similarities       : 可选，(N,N,4,4) 的相似度矩阵
        per_piece_k        : 每个 piece 取多少个 top-k 强配对
        max_strong_pairs   : 全局最多保留多少个强配对

    返回：
        solutions: List[solution]
    """
    n = len(pieces)
    total_area = sum(h * w for h, w in pieces)
    canvas_area = canvas_h * canvas_w
    if total_area != canvas_area:
        print(f"[WARN] total piece area ({total_area}) != canvas area ({canvas_area}), "
              "可能不存在完全铺满的解。")

    # 根据 similarities 生成 strong_pairs（如果有）
    strong_pairs: List[Tuple[int, int, float]] = []
    if similarities is not None:
        pair_lb = compute_pair_lb(similarities)
        strong_pairs = build_strong_pairs_topk_for_frame(
            pair_lb,
            pieces,
            per_piece_k=per_piece_k,
            max_pairs=max_strong_pairs,
        )
        print(f"[INFO] Using {len(strong_pairs)} strong pairs as frame constraints.")

    # 画布：-1 表示空，>=0 表示对应的 piece_index
    canvas = [[-1] * canvas_w for _ in range(canvas_h)]
    used = [False] * n
    current_orient: List[Optional[Tuple[int, int]]] = [None] * n
    solutions: List[List[Dict]] = []

    seen_frames = set()

    def find_empty():
        for r in range(canvas_h):
            for c in range(canvas_w):
                if canvas[r][c] == -1:
                    return r, c
        return None, None

    def can_place(idx: int, r: int, c: int, h: int, w: int) -> bool:
        if r + h > canvas_h or c + w > canvas_w:
            return False
        for i in range(r, r + h):
            row = canvas[i]
            for j in range(c, c + w):
                if row[j] != -1:
                    return False
        return True

    def place(idx: int, r: int, c: int, h: int, w: int, val: int):
        for i in range(r, r + h):
            for j in range(c, c + w):
                canvas[i][j] = val

    def build_frame_signature() -> Tuple[Tuple[int, ...], ...]:
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

    def build_solution_from_canvas() -> List[Dict]:
        sol: List[Dict] = []
        for i in range(n):
            h, w = current_orient[i]
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
        return sol

    def dfs():
        if max_solutions is not None and len(solutions) >= max_solutions:
            return

        r, c = find_empty()
        if r is None:
            sig = build_frame_signature()
            if sig in seen_frames:
                return

            sol = build_solution_from_canvas()

            # ★ 如果有 strong_pairs：要求每一对在 frame 中必须“矩形相邻”
            if strong_pairs and not check_strong_pairs_adjacency(sol, strong_pairs):
                return

            seen_frames.add(sig)
            solutions.append(sol)
            return

        seen_shapes_this_cell = set()

        for idx in range(n):
            if used[idx]:
                continue

            ph, pw = pieces[idx]
            shape_key = (ph, pw)
            if shape_key in seen_shapes_this_cell:
                continue
            seen_shapes_this_cell.add(shape_key)

            if allow_rotate and ph != pw:
                orientations = [(ph, pw), (pw, ph)]
            else:
                orientations = [(ph, pw)]

            for h, w in orientations:
                if not can_place(idx, r, c, h, w):
                    continue

                used[idx] = True
                current_orient[idx] = (h, w)
                place(idx, r, c, h, w, idx)

                dfs()

                # Early exit if we've found enough solutions
                if max_solutions is not None and len(solutions) >= max_solutions:
                    used[idx] = False
                    current_orient[idx] = None
                    return

                place(idx, r, c, h, w, -1)
                used[idx] = False
                current_orient[idx] = None

    dfs()
    return solutions
