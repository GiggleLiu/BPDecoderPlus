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

        # Cache for RREF computation
        self._cached_rref = None
        self._cached_pivot_cols = None
        self._cached_column_order = None

    def _compute_soft_weight(self, solution: np.ndarray, probs: np.ndarray) -> float:
        """
        Compute soft-weighted cost using standard log-probability weight.

        Cost = sum of -log(p_i) for positions where solution has an error (solution[i] = 1).
        This is the standard minimum weight decoding approach.

        Args:
            solution: binary error pattern (0 or 1 for each variable)
            probs: error probabilities from BP (p_i = P(error=1))

        Returns:
            Soft-weighted cost (lower is better)
        """
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        cost = np.sum(solution * (-np.log(probs_clipped)))
        return cost

    def _generate_osd_cs_candidates(self, k: int, osd_order: int) -> List[np.ndarray]:
        """
        Generate OSD-CS (Combination Sweep) candidate strings.

        OSD-CS searches over single-bit and two-bit flips instead of exhaustive search.
        This is much faster than exhaustive OSD-E while maintaining good performance.

        Args:
            k: Number of free variables
            osd_order: Maximum number of positions to consider for flips

        Returns:
            List of candidate bit patterns
        """
        candidates = []

        # Zero vector (all free variables = 0)
        candidates.append(np.zeros(k, dtype=np.int8))

        # Single-bit flips
        for i in range(k):
            candidate = np.zeros(k, dtype=np.int8)
            candidate[i] = 1
            candidates.append(candidate)

        # Two-bit flips (within osd_order)
        for i in range(min(osd_order, k)):
            for j in range(i + 1, min(osd_order, k)):
                candidate = np.zeros(k, dtype=np.int8)
                candidate[i] = 1
                candidate[j] = 1
                candidates.append(candidate)

        return candidates

    def solve(self, syndrome: np.ndarray, error_probs: np.ndarray, osd_order: int = 10, osd_method: str = 'exhaustive', random_seed: Optional[int] = None) -> np.ndarray:
        """
        Solve for the most likely error pattern using OSD post-processing.

        Args:
            syndrome: the observed syndrome (binary array of length m)
            error_probs: error probabilities from BP for each variable (array of length n)
                        Values should be in range [0, 1], representing P(error=1)
            osd_order: the search depth (lambda).
                       0 = OSD-0 (no search, fast but low accuracy)
                       >0 = OSD-E (search in the most suspicious osd_order free variables)
            osd_method: 'exhaustive' (default) or 'combination_sweep' (OSD-CS)
                       'exhaustive': search all 2^k combinations (k = min(osd_order, num_free))
                       'combination_sweep': search single and double bit flips only
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

        # Add a small perturbation to prevent sorting uncertainty
        if random_seed is not None:
            np.random.seed(random_seed)
        probs += np.random.uniform(0, 1e-6, size=self.num_errors)

        # 2. Sort (Soft Decision)
        # Sort by reliability: |p - 0.5| descending (most reliable first)
        reliability = np.abs(probs - 0.5)
        sorted_indices = np.argsort(reliability)[::-1]
        
        # 3. Build the augmented matrix [H_sorted | s] and compute RREF
        # Use caching to avoid recomputing RREF for same column order
        augmented, pivot_cols = self._get_rref_cached(sorted_indices, syndrome)
        
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

            # Precompute sorted probabilities for cost function
            probs_sorted = probs[sorted_indices]

            # Generate candidates based on method
            if osd_method == 'combination_sweep':
                # OSD-CS: Generate single and double bit flip candidates
                candidates = self._generate_osd_cs_candidates(len(search_cols), osd_order)
            else:
                # Exhaustive: Generate all 2^k combinations
                num_candidates = 1 << len(search_cols)
                candidates = [np.array([(i >> j) & 1 for j in range(len(search_cols))], dtype=np.int8)
                             for i in range(num_candidates)]

            # Search through candidates
            for e_T_search in candidates:

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
                # Use sorted probabilities since cand_sol is in sorted order
                cost = self._compute_soft_weight(cand_sol, probs_sorted)
                if cost < min_cost:
                    min_cost = cost
                    best_solution_sorted = cand_sol.copy()
            
            final_solution_sorted = best_solution_sorted

        # 6. Inverse mapping (Unsort)
        estimated_errors = np.zeros(self.num_errors, dtype=int)
        estimated_errors[sorted_indices] = final_solution_sorted

        return estimated_errors

    def _get_rref_cached(self, sorted_indices: np.ndarray, syndrome: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Get RREF with caching. Cache is valid when column order is unchanged.

        Args:
            sorted_indices: Column permutation order
            syndrome: Syndrome vector

        Returns:
            Tuple of (augmented matrix in RREF, pivot columns)
        """
        # IMPORTANT: Caching disabled due to bug where syndrome column wasn't being
        # transformed by the same row operations as the cached RREF matrix.
        # This caused invalid solutions that didn't satisfy the syndrome.
        # TODO: Implement proper caching by storing row operations and applying them to new syndromes

        # Always compute fresh RREF
        H_sorted = self.H[:, sorted_indices]
        augmented = np.hstack([H_sorted, syndrome.reshape(-1, 1)]).astype(np.int8)
        pivot_cols = self._compute_rref(augmented)

        return augmented, pivot_cols

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