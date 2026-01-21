import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class OSDDecoder:
    def __init__(self, H: np.ndarray):
        """
        Args:
            H: Parity Check Matrix (m rows, n columns)
        """
        self.H = H.astype(np.int8)
        self.num_checks, self.num_errors = H.shape

    def _compute_soft_weight(self, solution: np.ndarray, llrs: np.ndarray) -> float:
        """
        Compute soft-weighted cost based on BP log-likelihood ratios (LLRs).
        
        According to BP+OSD paper (arXiv:2005.07016), the cost should be:
        cost = sum_i [ e_i * |LLR_i| ] for errors that disagree with BP hard decision
        
        LLR > 0 means BP thinks no error (p < 0.5)
        LLR < 0 means BP thinks error (p > 0.5)
        
        Args:
            solution: binary error pattern (0 or 1 for each variable)
            llrs: log-likelihood ratios from BP (LLR = log((1-p)/p))
        
        Returns:
            Soft-weighted cost (lower is better)
        """
        # Cost is sum of |LLR| for positions where solution disagrees with BP hard decision
        # BP hard decision: error if LLR < 0, no error if LLR > 0
        bp_hard_decision = (llrs < 0).astype(int)
        disagreement = (solution != bp_hard_decision).astype(float)
        cost = np.sum(disagreement * np.abs(llrs))
        return cost

    def solve(self, syndrome: np.ndarray, error_probs: np.ndarray, osd_order: int = 10, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Solve for the most likely error pattern using OSD post-processing.
        
        Args:
            syndrome: the observed syndrome (binary array of length m)
            error_probs: error probabilities from BP for each variable (array of length n)
                        Values should be in range [0, 1], representing P(error=1)
            osd_order: the search depth (lambda).
                       0 = OSD-0 (no search, fast but low accuracy)
                       >0 = OSD-E (search in the most suspicious osd_order free variables)
            random_seed: optional random seed for deterministic behavior
            
        Returns:
            Estimated error pattern (binary array of length n)
        """
        # 1. Convert probabilities to numpy array and clip to valid range
        probs = np.asarray(error_probs, dtype=np.float64).flatten()
        if len(probs) != self.num_errors:
            raise ValueError(f"error_probs length {len(probs)} doesn't match number of errors {self.num_errors}")
        
        # Clip probabilities to avoid numerical issues
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Compute LLRs for soft-weighted cost function: LLR = log((1-p)/p)
        llrs = np.log((1 - probs) / probs)

        # Add a small perturbation to prevent sorting uncertainty
        if random_seed is not None:
            np.random.seed(random_seed)
        probs += np.random.uniform(0, 1e-6, size=self.num_errors)

        # 2. Sort (Soft Decision)
        # Sort by reliability: |p - 0.5| descending (most reliable first)
        reliability = np.abs(probs - 0.5)
        sorted_indices = np.argsort(reliability)[::-1]
        
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
            # Search over the least reliable (most uncertain) free variables

            # Select free variables with lowest reliability (closest to 0.5)
            free_cols_with_reliability = [(col, reliability[sorted_indices[col]]) for col in free_cols]
            # Sort by reliability (ascending - least reliable first)
            free_cols_with_reliability.sort(key=lambda x: x[1])
            # Select least reliable osd_order free variables
            search_cols = [col for col, _ in free_cols_with_reliability[:osd_order]]
            min_cost = float('inf')  # Initialize to infinity for soft-weighted cost
            best_solution_sorted = None
            
            # Precompute sorted LLRs for cost function
            llrs_sorted = llrs[sorted_indices]
            
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

                # Calculate soft-weighted cost and track best solution
                # Use sorted LLRs since cand_sol is in sorted order
                cost = self._compute_soft_weight(cand_sol, llrs_sorted)
                if cost < min_cost:
                    min_cost = cost
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