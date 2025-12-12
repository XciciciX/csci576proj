from compute_similarity import edge_distance, compute_piece_edge_descriptors
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Dict, Optional
from utils import rotate_piece, crop_background


# Store info of a piece in a specific rotation
PieceRot = namedtuple("PieceRot", ["piece_idx", "img", "edges", "shape"])

# ---------- 布局搜索（DFS + 剪枝） ----------

class PuzzleSolver:
    def __init__(self, pieces, grid_rows, grid_cols):
        self.all_rots = {}
        self.pieces = pieces               # dict[(piece_idx, rot_idx)] -> PieceRot
        self.num_pieces = len(pieces)
    
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

    # TODO: find all rectangle
    def find_largest_filled_rectangle(self) -> Tuple[int,int,int,int]:
        """
        在当前的 `self.current_layout` 中寻找最大的、完全被填充的矩形区域。

        返回：(top, left, height, width), sublayout, pieces_in_rect_set, leftover_piece_indices
        - sublayout 是矩形区域内的 (piece_idx, rot) 二维列表
        - pieces_in_rect_set 是该矩形包含的 piece indices 的集合
        - leftover_piece_indices 是未包含在该矩形中的 piece indices 列表

        算法：把布局视为二值矩阵（1=已放置，0=空），对每一行维护柱状图高度，使用单调栈求解每行的最大矩形。
        """
        rows = self.grid_rows
        cols = self.grid_cols
        # 二值矩阵：1 表示已放置
        mat = [[1 if self.best_layout[r][c] is not None else 0 for c in range(cols)] for r in range(rows)]

        heights = [0] * cols
        max_area = 0
        result_rect = (0, 0, 0, 0)
        top_row, left_col, h_val, w_val = 0, 0, 0, 0

        for r in range(rows):
            for c in range(cols):
                heights[c] = heights[c] + 1 if mat[r][c] == 1 else 0

            # largest rectangle in histogram 'heights'
            stack = []  # store indices
            c = 0
            while c <= cols:
                h = heights[c] if c < cols else 0
                if not stack or h >= heights[stack[-1]]:
                    stack.append(c)
                    c += 1
                else:
                    top_idx = stack.pop()
                    width = c if not stack else c - stack[-1] - 1
                    area = heights[top_idx] * width
                    if area > max_area:
                        max_area = area
                        h_val = heights[top_idx]
                        w_val = width
                        top_row = r - h_val + 1
                        left_col = stack[-1] + 1 if stack else 0
                        result_rect = (top_row, left_col, h_val, w_val)
        print(f"[INFO] result_rect: {result_rect}, max_area: {max_area}")
        sublayout = []
        pieces_in_rect = set()
        pRow = 0
        pCol = 0
        for rr in range(top_row, top_row + h_val):
            row_list = []
            pRow = 0
            for cc in range(left_col, left_col + w_val):
                cell = self.best_layout[rr][cc]
                row_list.append(cell)
                #TODO: rotate
                pRow = self.all_rots[cell[0]].shape[1]
                pCol += self.all_rots[cell[0]].shape[0]
                if cell is not None:
                    pieces_in_rect.add(cell[0])
            sublayout.append(row_list)
            
            print(f"[DEBUG] pCol after row {rr}: {pCol}")
        print(f"[INFO] Sublayout contents: {sublayout}")
        print(f"[INFO] Sublayout contents: {self.best_layout}")

       
        
        return (result_rect, sublayout, pieces_in_rect, (pRow, pCol))
    
    def group_pieces(self):
        # 构建 sublayout 和 pieces 集合
        result_rect, sublayout, pieces_in_rect, (pRow, pCol) = self.find_largest_filled_rectangle()
        top_row, left_col, h_val, w_val = result_rect

        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        # canvas = []

        print(f"[INFO] Grouping pieces into one rectangle of size {h_val} x {w_val}...")
        print(f"[INFO] Sublayout: {sublayout}")
        for r in range(len(sublayout)):
            for c in range(len(sublayout[0])):
                print(f"[DEBUG] Placing piece: {sublayout[r][c]}")
                piece_idx, rot_idx = sublayout[r][c]
                piece_rot = self.all_rots[piece_idx]
                img = rotate_piece(piece_rot.img, rot_idx)
                print(f"[DEBUG] Placing piece shape: {img.shape}")
                ph, pw = img.shape[:2]
                print(f"[DEBUG] Piece shape: {ph} x {pw}")
                y0 = r * ph
                x0 = c * pw
                # canvas = canvas.append(np.zeros((ph, pw, 3), dtype=np.uint8))
                canvas[y0:y0+ph, x0:x0+pw, :] = img
        canvas = crop_background(canvas, bg_color=(0,0,0), tol=0)

        return (canvas, pieces_in_rect)


    # def get_new_pieces():
    #     # 1. Get groups of pieces from self.current_layout, make sure is a rectangle

    #      if max_area > 0 and h_val > 0 and w_val > 0:


    
    def solve_packing_2(self):
        # get a union set with same height or width
        # [H/W] -> set(piece index)
        # Every piece may be original w or h, need to be recorded in as well
        # make sure each piece only appear once
        # get one dict: [H/W] -> set(piece index)

        while True:
            if len(self.pieces) <= 1:
                break
            self.all_rots = self.build_all_rotations(self.pieces)
            self.num_pieces = len(self.all_rots)
            prev_num_pieces = self.num_pieces
            self.find_same_HW()
            self.build_solution_same_HW()
            if self.num_pieces == prev_num_pieces:
                break
        return self.pieces[0]
       
        # First, we have original pieces.
        # Do find_same_HW, build_solution_same_HW until no more same H/W groups can be found.
        # Then, do frame() for all remaining pieces if there is any.

    def find_same_HW(self):
        # Build height_to_indices and width_to_indices
        for idx, prot in enumerate(self.all_rots):
            ph, pw = prot.shape
            self.height_to_indices.setdefault(ph, []).append(idx)
            self.width_to_indices.setdefault(pw, []).append(idx)
        # combine two dicts into one 
        # same_HW_dict: Dict[int, List[int]] = {}
        # for h, indices in height_to_indices.items():
        #     if len(indices) >= 2:
        #         same_HW_dict[h] = indices
        # for w, indices in width_to_indices.items():
        #     if w in same_HW_dict and len(indices) >= 2:
        #         same_HW_dict[w].append(indices)
        #     elif len(indices) >= 2:
        #         same_HW_dict[w] = indices
            
        
    
    def initialize_DFS_variables(self, sameH_lists):
        # TODO: change back
        self.grid_rows = 4
        self.grid_cols = 4
        print(f"[INFO] Initialized DFS grid size: {self.grid_rows} x {self.grid_cols}")
        self.positions = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]

        self.best_cost = float("inf")
        self.best_layout = None  # 2D: (piece_idx, rot_idx)

        # 当前状态
        self.current_layout = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.used_piece = [False] * self.num_pieces
        self.current_cost = 0.0

        self.solutions_found = 0
    
    def build_solution_same_HW(self):
        
         # 1. For every same H/W group, do DFS search
        #   a. check the index is not used
        #   b. >= 2
        # 2. Each time a solution is found, 
        #   a. record the best solution found as one big piece 
        #   b. record used pieces in the set
        #   c. put these pieces back to new pieces
        # 3. Finally, get a new pieces list with all pieces grouped + original pieces that cannot be grouped.
        # 4. Recalculate all_rots with new pieces list
       

        
        pieces = []
        

        all_lists = [self.height_to_indices, self.width_to_indices]
        for lst in all_lists:
            for HW, same_lists in lst.items():
                # delete used from sameH_lists
                same_lists = [idx for idx in same_lists if idx not in self.used_index]
                print(f"[INFO] Processing H/W={HW} group with pieces: {same_lists}")

                if len(same_lists) == 0:
                    continue
                if len(same_lists) == 1:
                    pieces.append(self.all_rots[same_lists[0]].img)
                    continue
               
                

                # Update variables
                self.initialize_DFS_variables(same_lists)
                self._get_score()
                self._build_candidates(top_k=6)
                self._dfs(0, len(same_lists))
                print(f"[INFO] Best cost for H/W={HW} group: {self.best_cost:.4f}")
                print(f"[INFO] Best layout for H/W={HW} group: {self.best_layout}")
                img, pieces_in_rect = self.group_pieces()

                self.used_index.update(pieces_in_rect)
                pieces.append(img)
        for idx in range(self.num_pieces):
            if idx not in self.used_index:
                pieces.append(self.all_rots[idx].img)
                
        self.pieces = pieces
        self.num_pieces = len(self.pieces)
        

        

        # For every same map key, we do one of DFS search that is similar to regular one

        # Input: one dict containing [H/W] -> set(piece index)
        # Search all possible solutions.
        # DFS: randomly pick one to start, try fill right all first, then down.
        # 1. (4 edges into consideration) Sim < 0.8
        # 2. The image size cannot be exceeded
        # 3. We can calculate all similarity with same size first

        # Then, we need to record the best solution found, see them as a big piece. If it is not a sqare,
        # cut it into squares.

        # Create a new pieces list, with the best solution found as one piece, and other pieces that are not used.
    
    
    # Repeat find_same_HW and build_solution_same_HW until the total N of the returned new pieces list is the same
    # return a new pieces list with all pieces that cannot be grouped.
    
    def frame(self):
        # For all pieces that cannot be grouped, do the regular frame solver.
        # Each time a possible frame is found, we need to check the score.
        # If score < threshold, we keep it as a possible solution.
        # Finally, return best frame found.
        pass



    def build_all_rotations(self,pieces):
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





        

