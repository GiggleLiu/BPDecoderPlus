import numpy as np
from typing import Dict, List, Tuple

class OSDDecoder:
    def __init__(self, H: np.ndarray):
        """
        Args:
            H: Parity Check Matrix (m rows, n columns)
        """
        self.H = H.astype(np.int8)
        self.num_checks, self.num_errors = H.shape

    def solve(self, syndrome: np.ndarray, marginals: Dict[int, float], error_var_start: int, osd_order: int = 10, random_seed: int = None) -> np.ndarray:
        """
        Args:
            syndrome: the observed syndrome
            marginals: the marginal probabilities calculated by BP
            error_var_start: the starting index of the error variables
            osd_order: the search depth (lambda).
                       0 = OSD-0 (no search, fast but low accuracy)
                       >0 = OSD-E (search in the most suspicious osd_order free variables)
            random_seed: optional random seed for deterministic behavior
        """
        # 1. Extract the probabilities and add a small perturbation (Break ties)
        probs = np.zeros(self.num_errors)
        for i in range(self.num_errors):
            var_idx = error_var_start + i
            if var_idx in marginals:
                probs[i] = marginals[var_idx][1].item()

        # Add a small perturbation to prevent sorting uncertainty
        if random_seed is not None:
            np.random.seed(random_seed)
        probs += np.random.uniform(0, 1e-6, size=self.num_errors)

        # 2. Sort (Soft Decision)
        # Sort the errors by probability from largest to smallest
        sorted_indices = np.argsort(probs)[::-1]
        
        # 3. Build the augmented matrix [H_sorted | s]
        H_sorted = self.H[:, sorted_indices]
        augmented = np.hstack([H_sorted, syndrome.reshape(-1, 1)]).astype(np.int8)
        
        # 4. Execute full RREF elimination
        # This step will turn the matrix into the form of [I  M | s'] (logically)
        # pivot_cols is the column corresponding to I, free_cols is the column corresponding to M
        pivot_cols = self._compute_rref(augmented)
        
        # Determine the column indices of the free variables (in the sorted space)
        all_cols = set(range(self.num_errors))
        free_cols = sorted(list(all_cols - set(pivot_cols)))

        # Debug: Check distribution of pivot vs free variables
        # print(f"Number of pivots: {len(pivot_cols)}, Number of free: {len(free_cols)}")
        # if len(free_cols) > 0:
        #     free_probs = [probs[sorted_indices[col]] for col in free_cols[:min(10, len(free_cols))]]
        #     pivot_probs = [probs[sorted_indices[col]] for col in pivot_cols[:min(10, len(pivot_cols))]]
        #     print(f"Top 10 free variable probs: {free_probs}")
        #     print(f"Top 10 pivot variable probs: {pivot_probs}")

        # 5. OSD post-processing (Searching)
        # We not only need to find a solution, but also need to find the *lightest* solution.

        # Basis solution (OSD-0 Solution): Assume all free variables are 0
        # e_S = s' (for the pivot columns)
        solution_base = np.zeros(self.num_errors, dtype=np.int8)

        # Build the row mapping: the pivot column c corresponds to which row r
        pivot_row_map = {}
        for r in range(augmented.shape[0]):
            # Find the pivot column in this row
            # Note: Since it is RREF, the first non-zero element in this row is the pivot
            row_pivots = np.where(augmented[r, :self.num_errors] == 1)[0]
            if len(row_pivots) > 0:
                col = row_pivots[0] # The first one is the pivot
                if col in pivot_cols: # 再次确认
                    pivot_row_map[col] = r
                    # OSD-0 assignment
                    solution_base[col] = augmented[r, -1]

        # If osd_order == 0, return the OSD-0 result directly
        if osd_order == 0:
            final_solution_sorted = solution_base
        else:
            # --- OSD-E (Exhaustive Search) Core logic ---
            # We only search in the "most suspicious" (highest probability) osd_order free variables.

            # Select free variables with highest probability
            # Map free columns in sorted space back to their original probabilities
            free_cols_with_probs = [(col, probs[sorted_indices[col]]) for col in free_cols]
            # Sort by probability (descending)
            free_cols_with_probs.sort(key=lambda x: x[1], reverse=True)
            # Select top osd_order free variables
            search_cols = [col for col, _ in free_cols_with_probs[:osd_order]]
            min_weight = self.num_errors + 1 # Initialize to a large number
            best_solution_sorted = None
            
            # Iterate through 2^k possibilities (k = len(search_cols))
            # For d=3, order=10 ~ 15 is very fast
            num_candidates = 1 << len(search_cols)
            
            for i in range(num_candidates):
                # Set e_T (free variables assignment for this candidate)
                e_T_search = np.array([(i >> j) & 1 for j in range(len(search_cols))], dtype=np.int8)

                # Initialize candidate solution
                cand_sol = np.zeros(self.num_errors, dtype=np.int8)
                cand_sol[search_cols] = e_T_search

                # Compute target syndrome: s' + M @ e_T (mod 2)
                M_subset = augmented[:, search_cols]
                target_syndrome = (augmented[:, -1] + M_subset @ e_T_search) % 2

                # Set pivot variables based on target syndrome
                for r in range(augmented.shape[0]):
                    pivots_in_row = np.where(augmented[r, :self.num_errors] == 1)[0]
                    if len(pivots_in_row) > 0:
                        pivot_c = pivots_in_row[0]
                        cand_sol[pivot_c] = target_syndrome[r]

                # Calculate Hamming weight and track best solution
                w = np.sum(cand_sol)
                if w < min_weight:
                    min_weight = w
                    best_solution_sorted = cand_sol.copy()
            
            final_solution_sorted = best_solution_sorted

        # 6. Inverse mapping (Unsort)
        estimated_errors = np.zeros(self.num_errors, dtype=int)
        estimated_errors[sorted_indices] = final_solution_sorted
        
        return estimated_errors

    def _compute_rref(self, M: np.ndarray) -> List[int]:
        """
        Compute the reduced row echelon form (RREF) of matrix M - In-place modification
        """
        m, n = M.shape
        num_cols_to_scan = n - 1 
        
        pivot_row = 0
        pivot_cols = []
        
        for col in range(num_cols_to_scan):
            if pivot_row >= m:
                break
            
            candidates = np.where(M[pivot_row:, col] == 1)[0]
            
            if len(candidates) == 0:
                continue 
            
            swap_r = candidates[0] + pivot_row
            if swap_r != pivot_row:
                M[[pivot_row, swap_r]] = M[[swap_r, pivot_row]]
            
            pivot_cols.append(col)
            
            rows_to_xor = np.where(M[:, col] == 1)[0]
            rows_to_xor = rows_to_xor[rows_to_xor != pivot_row]
            
            if len(rows_to_xor) > 0:
                M[rows_to_xor, :] ^= M[pivot_row, :]
            
            pivot_row += 1
            
        return pivot_cols