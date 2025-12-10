from compute_similarity import edge_distance
import numpy as np

from typing import List, Tuple, Dict, Optional
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
        self._build_candidates(top_k=5)
        self._dfs(0)
        return self.best_layout, self.best_cost
    
    def irr_solve(self, canvas_h, canvas_w, pieces):
    
        self._get_score()
        
        return self._irregular(canvas_h, canvas_w, pieces)
        # return self.best_layout, self.best_cost


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

    def _irregular(self, canvas_h, canvas_w, pieces):
        solutions = solve_packing(
            canvas_h,
            canvas_w,
            pieces,
            max_solutions=10,          # 最多保留 200 个 frame
            allow_rotate=False,
            # similarities=self.sim,  # ★ 加上相似度
            # per_piece_k=2,              # 每个 piece 选 2 个强配对
            # max_strong_pairs=None,      # 或者设为 3*len(pieces) 控制一下
        )
        return solutions
    

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






    

