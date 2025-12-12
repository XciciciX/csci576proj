from compute_similarity import edge_distance, compute_piece_edge_descriptors
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Dict, Optional
from utils import rotate_piece, crop_background, edge_len


# Store info of a piece in a specific rotation
PieceRot = namedtuple("PieceRot", ["piece_idx", "img", "edges", "shape"])

# ---------- 布局搜索（DFS + 剪枝） ----------

class Solver:
    def __init__(self, all_rots, grid_rows, grid_cols):
        self.all_rots = all_rots
        self.num_pieces = len(all_rots)
    
        # DFS Usage
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.positions = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]

        self.best_cost = float("inf")
        self.best_layout = None  # 2D: (piece_idx, rot_idx)

        self.current_layout = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.used_piece = [False] * self.num_pieces
        self.current_cost = 0.0

        self.solutions_found = 0

        self.sim = []
        self.candidates = []


    
        # Grouping
        self.height_to_indices: Dict[int, List[int]] = {}
        self.width_to_indices: Dict[int, List[int]] = {}

        self.used_index = set()



    def solve(self, n):
        
        self._get_score()
        self._build_candidates(top_k=15)
        self._dfs(0, n)
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
                for e1 in range(4):
                    for e2 in range(4):
                        if edge_len(pi.shape, e1) == edge_len(pj.shape, e2):
                            sim[i][j][e1][e2] = edge_distance(pi.edges[e1], pj.edges[e2])
                        else:
                            sim[i][j][e1][e2] = np.inf
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
    def _dfs(self, pos_idx, n):
        """
        深度优先 + branch-and-bound：
        - 不再用 solutions_found / MAX_SEARCH_SOLUTIONS 提前退出
        - 只保留基于 current_cost / best_cost 的安全剪枝
        """
        # 所有位置都填满了，检查一次完整布局
        if pos_idx == n:
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
                # 只考虑与已放好的“上”和“左”的匹配代价
                add_cost = 0.0

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
                self._dfs(pos_idx + 1, n)

                # 回溯
                self.current_layout[r][c] = None
                self.used_piece[piece_idx] = False
                self.current_cost = prev_cost