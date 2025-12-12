

import numpy as np

from typing import List, Tuple, Dict, Optional
from utils import rotate_piece, crop_background, build_all_rotations, max_rows_for_single_col, split_image_bisect_until_max
from solver import Solver

import cv2

class PuzzleSolver:
    def __init__(self, pieces, grid_rows, grid_cols, is_rotated=False):
        
        self.pieces = pieces               # dict[(piece_idx, rot_idx)] -> PieceRot
        self.num_pieces = len(pieces)
    
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.is_rotated = is_rotated
        # self.positions = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]

        self.best_cost = float("inf")
        self.best_layout = None  # 2D: (piece_idx, rot_idx)

        # self.current_layout = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.used_piece = [False] * self.num_pieces
        self.current_cost = 0.0

        self.solutions_found = 0

        self.sim = []
        self.candidates = []


    
        # Grouping
        self.height_to_indices: Dict[int, List[int]] = {}
        self.width_to_indices: Dict[int, List[int]] = {}

        self.used_index = set()
        self.new_to_old = []

    
    def solve_packing_2(self):
        # get a union set with same height or width
        # [H/W] -> set(piece index)
        # Every piece may be original w or h, need to be recorded in as well
        # make sure each piece only appear once
        # get one dict: [H/W] -> set(piece index)

        # First, we have original pieces.
        # Do find_same_HW, build_solution_same_HW until no more same H/W groups can be found.
        # Then, do frame() for all remaining pieces if there is any.

        while True:
            print(f"length: {len(self.pieces)}")
            if len(self.pieces) <= 1:
                break
            prev_num_pieces = self.num_pieces
           
            self.height_to_indices= {}
            self.width_to_indices = {}
            self.find_same_HW()
            self.build_solution_same_HW()
            if self.num_pieces == prev_num_pieces:
                break
        return self.pieces[0]
       
        

    def find_same_HW(self):
    # Build height_to_indices and width_to_indices
        for idx, prot in enumerate(self.pieces):
            ph, pw = prot.shape[:2]
            self.height_to_indices.setdefault(ph, []).append(idx)
            self.width_to_indices.setdefault(pw, []).append(idx)

        # Merge height and weight dicts, make sure indices are not duplicated
        # 1) 合并 height_to_indices 和 width_to_indices
        for h, indices in self.height_to_indices.items():
            if h in self.width_to_indices:
                # merge and deduplicate
                merged = list(set(self.width_to_indices[h]) | set(indices))
                self.width_to_indices[h] = merged
            else:
                self.width_to_indices[h] = list(set(indices))

        # 合并只差1的组并裁剪pieces（放在排序前）
        merged = self.merge_and_crop_groups(self.width_to_indices, tol=1)
        self.width_to_indices = merged

        # 2) 按 value(list) 的长度降序排序（越多越靠前）
        # sorted_heights = sorted(self.height_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True)
        sorted_widths  = sorted(self.width_to_indices.items(),  key=lambda kv: len(kv[1]), reverse=True)

        # 3) 如果你想把它们变回 dict（保持这个顺序，Python 3.7+ dict 保序）
        # self.height_to_indices = dict(sorted_heights)
        self.width_to_indices  = dict(sorted_widths)
        print(self.width_to_indices)

    
    def merge_and_crop_groups(self, groups: dict, tol: int = 1):
        """
        合并key只差tol的组，并将组内所有图片裁剪为组内最小尺寸。
        groups: dict[key, List[idx]]
        images: List[np.ndarray]
        返回: new_groups, new_images
        """
        # 1. 合并key只差tol的组
        keys = sorted(groups.keys())
        merged = {}
        used = set()
        # 第一步：先合并395-400为一个组
        merge_keys = [k for k in keys if 395 <= k <= 400]
        # 判断分组是按宽度还是高度：如果所有组内图片的宽度都等于key，则按宽度分组，否则按高度分组
        axis = 1  # 默认按宽度
        if merge_keys:
            # 检查第一个key的所有图片shape
            sample_idx = groups[merge_keys[0]][0]
            sample_shape = self.pieces[sample_idx].shape
            if all(self.pieces[idx].shape[0] == k for k in merge_keys for idx in groups[k]):
                axis = 0  # 按高度分组
            merged_group = []
            if axis == 1:
                min_dim = min(self.pieces[idx].shape[1] for k in merge_keys for idx in groups[k])
            else:
                min_dim = min(self.pieces[idx].shape[0] for k in merge_keys for idx in groups[k])
            for k in merge_keys:
                for idx in groups[k]:
                    img = self.pieces[idx]
                    h, w = img.shape[:2]
                    if axis == 1:
                        if w > min_dim:
                            self.pieces[idx] = img[:h, :min_dim].copy()
                        else:
                            self.pieces[idx] = img
                    else:
                        if h > min_dim:
                            self.pieces[idx] = img[:min_dim, :w].copy()
                        else:
                            self.pieces[idx] = img
                    merged_group.append(idx)
                used.add(k)
            merged[min(merge_keys)] = sorted(list(set(merged_group)))

        # 第二步：对剩余key做只差tol的合并
        for i, k in enumerate(keys):
            if k in used:
                continue
            group = set(groups[k])
            merged_group = [*group]
            for j in range(i+1, len(keys)):
                k2 = keys[j]
                if k2 in used:
                    continue
                if abs(k2 - k) == tol:
                    group2 = set(groups[k2])
                    merged_group += list(group2)
                    used.add(k2)
                    # 判断分组是按宽度还是高度
                    axis = 1
                    if all(self.pieces[idx].shape[0] == k or self.pieces[idx].shape[1] == k for idx in list(group) + list(group2)):
                        # 如果所有图片的高度等于key，则按高度分组
                        if all(self.pieces[idx].shape[0] == k for idx in list(group) + list(group2)):
                            axis = 0
                    if axis == 1:
                        min_dim = min(self.pieces[idx].shape[1] for idx in list(group) + list(group2))
                    else:
                        min_dim = min(self.pieces[idx].shape[0] for idx in list(group) + list(group2))
                    for idx in list(group) + list(group2):
                        img = self.pieces[idx]
                        h, w = img.shape[:2]
                        # 对每个piece分别判断key是高还是宽
                        if h == k or h == k2:
                            # 按高度分组
                            if h > min_dim:
                                self.pieces[idx] = img[:min_dim, :w].copy()
                            else:
                                self.pieces[idx] = img
                        elif w == k or w == k2:
                            # 按宽度分组
                            if w > min_dim:
                                self.pieces[idx] = img[:h, :min_dim].copy()
                            else:
                                self.pieces[idx] = img
                        else:
                            # 不裁剪
                            self.pieces[idx] = img
            merged[k] = sorted(list(set(merged_group)))
        return merged


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
        
        
        final_pieces = []
        self.used_index = set()
        
        #TODO
        # all_lists = [self.height_to_indices, self.width_to_indices]
        all_lists = [self.width_to_indices]
        print(all_lists)
        for lst in all_lists:
            for HW, same_lists in lst.items():
                # delete used from sameH_lists
                same_lists = [idx for idx in same_lists if idx not in self.used_index]
                print(f"[INFO] Processing H/W={HW} group with pieces: {same_lists}")

                if len(same_lists) == 0:
                    continue
                if len(same_lists) == 1:
                #     final_pieces.append(self.pieces[same_lists[0]])
                    continue
               
                #TODO: change grid rows grid cols
                # picked, (H, W) = max_rows_for_single_col(self.pieces, same_lists, max_H=400, max_W=400)

                # self.grid_rows = len(picked)
                print("[DEBUG] Rows picked: {self.grid_rows}")
                self.grid_rows = len(same_lists)
                self.grid_cols = 1
                crt_pieces = []
                new_to_old = []
                for idx in range(self.num_pieces):
                    if idx in same_lists:
                        print(idx)
                        crt_pieces.append(self.pieces[idx])
                        new_to_old.append(idx)
                self.new_to_old = new_to_old

                all_rots = build_all_rotations(crt_pieces, self.is_rotated)
    
                solver = Solver(all_rots, self.grid_rows, self.grid_cols)
                self.best_layout, best_cost = solver.solve(len(same_lists)) # the number of grid_rows * grid_cols should be larger than N

                # Update variables
                # self.initialize_DFS_variables(same_lists)
                
                print(f"[INFO] Best cost for H/W={HW} group: {best_cost:.4f}")
                print(f"[INFO] Best layout for H/W={HW} group: {self.best_layout}")
                
                img, pieces_in_rect = self.group_pieces()
                output_image_path = f"{HW}.png"
                cv2.imwrite(output_image_path, img)

                self.used_index.update(pieces_in_rect)
                


                pieces = split_image_bisect_until_max(img, 400, 0)
                print(f"[DEBUG] split_image_bisect_until_max: {[p.shape for p in pieces]}")
                final_pieces.extend(pieces)
                
                # print(len(pieces), [p.shape[:2] for p in pieces])



        for idx in range(self.num_pieces):
            if idx not in self.used_index:
                final_pieces.append(self.pieces[idx])
                
        self.pieces = final_pieces
        self.num_pieces = len(self.pieces)


        # split. if the size is larger than 400, clip into pieces <= 400

    

            

    def group_pieces(self):
        result_rect, sublayout, pieces_in_rect, _ = self.find_largest_filled_rectangle()

        # sublayout 可能是 ragged（每行列数不同），先安全取 row/col 数
        R = len(sublayout)
        C = max(len(row) for row in sublayout) if R > 0 else 0
        if R == 0 or C == 0:
            return None, pieces_in_rect

        # 1) 预先计算“每个格子旋转后的尺寸”
        cell_hw = [[None] * len(sublayout[r]) for r in range(R)]
        for r in range(R):
            for c in range(len(sublayout[r])):
                piece_idx, rot_idx = sublayout[r][c]
                img = rotate_piece(self.pieces[piece_idx], rot_idx)
                print(f"[DEBUG] group_pieces: piece_idx={piece_idx}, rot_idx={rot_idx}, img.shape={img.shape}")
                ph, pw = img.shape[:2]
                cell_hw[r][c] = (ph, pw)

        # 2) 计算 row_heights：每行取最大高度（同一行理论上应相等；取 max 更稳）
        row_heights = []
        for r in range(R):
            row_heights.append(max(cell_hw[r][c][0] for c in range(len(sublayout[r]))))

        # 3) 计算 col_widths：每列取最大宽度（同一列理论上应相等；取 max 更稳）
        col_widths = [0] * C
        for r in range(R):
            for c in range(len(sublayout[r])):
                col_widths[c] = max(col_widths[c], cell_hw[r][c][1])

        # 4) prefix sums -> 每格起点
        y_offsets = [0]
        for h in row_heights:
            y_offsets.append(y_offsets[-1] + h)

        x_offsets = [0]
        for w in col_widths:
            x_offsets.append(x_offsets[-1] + w)

        H = y_offsets[-1]
        W = x_offsets[-1]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        print(f"[INFO] group canvas size = {H} x {W}")

        # 5) paste：注意 numpy 是 [y, x]
        for r in range(R):
            for c in range(len(sublayout[r])):
                piece_idx, rot_idx = sublayout[r][c]
                img = rotate_piece(self.pieces[piece_idx], rot_idx)
                ph, pw = img.shape[:2]

                y0 = y_offsets[r]
                x0 = x_offsets[c]

                canvas[y0:y0+ph, x0:x0+pw, :] = img

        canvas = crop_background(canvas, bg_color=(0, 0, 0), tol=0)
        return canvas, pieces_in_rect
        


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
                #new to old
                real_idx = self.new_to_old[cell[0]]
                cell = (real_idx, cell[1])
                row_list.append(cell)
                #TODO: rotate
                
                if cell is not None:
                    pieces_in_rect.add(cell[0])
            sublayout.append(row_list)
            
            print(f"[DEBUG] pCol after row {rr}: {pCol}")
        print(f"[INFO] Sublayout contents: {sublayout}")
        print(f"[INFO] Sublayout contents: {self.best_layout}")

       
        
        return (result_rect, sublayout, pieces_in_rect, (pRow, pCol))
    
        
    
    

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









        

