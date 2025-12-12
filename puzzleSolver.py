
import numpy as np

from typing import List, Tuple, Dict, Optional
from utils import rotate_piece, crop_background, build_all_rotations
from solver import Solver

import cv2

class PuzzleSolver:
    def __init__(self, pieces, grid_rows, grid_cols):
        
        self.pieces = pieces               # dict[(piece_idx, rot_idx)] -> PieceRot
        self.num_pieces = len(pieces)
    
        # DFS Usage
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
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
        # 2) 按 value(list) 的长度降序排序（越多越靠前）
        sorted_heights = sorted(self.height_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True)
        sorted_widths  = sorted(self.width_to_indices.items(),  key=lambda kv: len(kv[1]), reverse=True)

        # 3) 如果你想把它们变回 dict（保持这个顺序，Python 3.7+ dict 保序）
        self.height_to_indices = dict(sorted_heights)
        self.width_to_indices  = dict(sorted_widths)
        print(self.height_to_indices)
        print(self.width_to_indices)


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
        all_lists = [self.height_to_indices, self.width_to_indices]
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

                all_rots = build_all_rotations(crt_pieces)
    
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
                final_pieces.append(img)
        for idx in range(self.num_pieces):
            if idx not in self.used_index:
                final_pieces.append(self.pieces[idx])
                
        self.pieces = final_pieces
        self.num_pieces = len(self.pieces)
        

        


    def group_pieces(self):
        # 构建 sublayout 和 pieces 集合
        result_rect, sublayout, pieces_in_rect, (pRow, pCol) = self.find_largest_filled_rectangle()
        top_row, left_col, h_val, w_val = result_rect

        canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        # canvas = []

        print(f"[INFO] Grouping pieces into one rectangle of size {h_val} x {w_val}...")
        print(f"[INFO] Sublayout: {sublayout}")
        x0_prev = 0
        for r in range(len(sublayout)):
            y0_prev = 0
            for c in range(len(sublayout[0])):
                print(f"[DEBUG] Placing piece: {sublayout[r][c]}")
                piece_idx, rot_idx = sublayout[r][c]
                piece = self.pieces[piece_idx]
                img = rotate_piece(piece, rot_idx)
                print(f"[DEBUG] Placing piece shape: {img.shape}")
                # if rotate, shape will rotate as well
                ph, pw = img.shape[:2]
                print(f"[DEBUG] Piece shape: {ph} x {pw}")

                
                # canvas = canvas.append(np.zeros((ph, pw, 3), dtype=np.uint8))
                canvas[x0_prev:x0_prev+ph, y0_prev:y0_prev+pw, :] = img
                x0_prev = x0_prev+ph
                y0_prev = y0_prev+pw
                print(f"[DEBUG]: CRT {y0_prev}, {x0_prev}")
        canvas = crop_background(canvas, bg_color=(0,0,0), tol=0)

        return (canvas, pieces_in_rect)


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
    
        
    
    # def initialize_DFS_variables(self, sameH_lists):
    #     # TODO: change back
    #     self.grid_rows = 
    #     self.grid_cols = 0
    #     print(f"[INFO] Initialized DFS grid size: {self.grid_rows} x {self.grid_cols}")
    #     self.positions = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]

    #     self.best_cost = float("inf")
    #     self.best_layout = None  # 2D: (piece_idx, rot_idx)

    #     # 当前状态
    #     self.current_layout = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
    #     self.used_piece = [False] * self.num_pieces
    #     self.current_cost = 0.0

    #     self.solutions_found = 0
    
    

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









        

