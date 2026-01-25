import torch
import numpy as np
from typing import Tuple

class BatchBPDecoder:
    """Batch Belief Propagation decoder for parallel syndrome processing on GPU."""

    def __init__(self, H: np.ndarray, channel_probs: np.ndarray, device='cuda', ms_scaling_factor: float = 1.0):
        """
        Args:
            H: Parity check matrix (num_checks, num_errors)
            channel_probs: Physical error rates per errors (num_errors,)
            device: torch device
            ms_scaling_factor: Scaling factor for min-sum algorithm (default 1.0 to match ldpc library)
        """
        self.device = torch.device(device)
        self.H = torch.from_numpy(H).to(dtype=torch.float32, device=self.device)
        self.num_checks, self.num_qubits = H.shape

        # Store channel probs
        self.channel_probs = torch.from_numpy(channel_probs).to(dtype=torch.float32, device=self.device)
        
        # Min-sum scaling factor (default 1.0 to match ldpc library)
        self.ms_scaling_factor = ms_scaling_factor

        # Build edge lists for message passing
        #Example: If edge #5 connects Check 2 and Qubit 7, then:
        # self.check_edges[5] will be 2
        # self.qubit_edges[5] will be 7
        self.check_edges, self.qubit_edges = torch.where(self.H)
        self.num_edges = len(self.check_edges)

        # Precompute adjacency structures for fast message passing
        self._precompute_adjacency()

    def _precompute_adjacency(self):
        """Precompute adjacency structures for vectorized message passing."""
        # For each check, store which edges belong to it
        self.check_to_edges = []
        self.max_check_degree = 0
        for c in range(self.num_checks):
            edges = (self.check_edges == c).nonzero(as_tuple=True)[0]
            self.check_to_edges.append(edges)
            self.max_check_degree = max(self.max_check_degree, len(edges))

        # For each qubit, store which edges belong to it
        self.qubit_to_edges = []
        self.max_qubit_degree = 0
        for q in range(self.num_qubits):
            edges = (self.qubit_edges == q).nonzero(as_tuple=True)[0]
            self.qubit_to_edges.append(edges)
            self.max_qubit_degree = max(self.max_qubit_degree, len(edges))

        # Create padded tensors for vectorized operations
        # Check adjacency: (num_checks, max_check_degree)
        self.check_edge_indices = torch.full((self.num_checks, self.max_check_degree), -1,
                                              dtype=torch.long, device=self.device)
        self.check_edge_mask = torch.zeros((self.num_checks, self.max_check_degree),
                                            dtype=torch.bool, device=self.device)
        for c, edges in enumerate(self.check_to_edges):
            self.check_edge_indices[c, :len(edges)] = edges
            self.check_edge_mask[c, :len(edges)] = True


        # Qubit adjacency: (num_qubits, max_qubit_degree)
        self.qubit_edge_indices = torch.full((self.num_qubits, self.max_qubit_degree), -1,
                                              dtype=torch.long, device=self.device)
        self.qubit_edge_mask = torch.zeros((self.num_qubits, self.max_qubit_degree),
                                            dtype=torch.bool, device=self.device)
        for q, edges in enumerate(self.qubit_to_edges):
            self.qubit_edge_indices[q, :len(edges)] = edges
            self.qubit_edge_mask[q, :len(edges)] = True

    def decode(self, syndromes: torch.Tensor, max_iter: int = 20, damping: float = 0.2, method: str = 'min-sum') -> torch.Tensor:
        """
        Decode batch of syndromes in parallel using BP.

        Args:
            syndromes: (batch_size, num_checks)
            max_iter: Maximum BP iterations
            damping: Damping factor
            method: 'sum-product' or 'min-sum' message passing

        Returns:
            marginals: (batch_size, num_qubits) - posterior error probabilities
        """
        batch_size = syndromes.shape[0]
        syndromes = syndromes.to(self.device)
        
        # Use LLR (log-likelihood ratio) representation for numerical stability
        # LLR = log(P(0)/P(1)), positive means more likely 0
        # Initialize channel LLRs
        channel_llr = torch.log((1 - self.channel_probs + 1e-10) / (self.channel_probs + 1e-10))

        # Messages: (batch_size, num_edges) in LLR form
        # msg_q2c[b, e] = LLR from qubit to check
        # msg_c2q[b, e] = LLR from check to qubit
        msg_q2c = channel_llr[self.qubit_edges].unsqueeze(0).expand(batch_size, -1).clone()
        msg_c2q = torch.zeros(batch_size, self.num_edges, device=self.device)

        for _ in range(max_iter):
            # Check-to-qubit messages
            if method == 'min-sum':
                msg_c2q_new = self._check_to_qubit_minsum(msg_q2c, syndromes)
            elif method == 'sum-product':
                msg_c2q_new = self._check_to_qubit_sumproduct(msg_q2c, syndromes)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'min-sum' or 'sum-product'.")

            # Damping
            msg_c2q = damping * msg_c2q + (1 - damping) * msg_c2q_new

            # Qubit-to-check messages
            msg_q2c = self._qubit_to_check(msg_c2q, channel_llr)

        # Compute marginals
        marginals = self._compute_marginals(msg_c2q, channel_llr)

        return marginals

    def _check_to_qubit_minsum(self, msg_q2c: torch.Tensor, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Compute check-to-qubit messages using min-sum algorithm.

        Args:
            msg_q2c: (batch_size, num_edges) - qubit-to-check LLRs
            syndromes: (batch_size, num_checks) - syndrome values

        Returns:
            msg_c2q: (batch_size, num_edges) - check-to-qubit LLRs
        """
        batch_size = msg_q2c.shape[0]
        msg_c2q = torch.zeros(batch_size, self.num_edges, device=self.device)

        # Process each check
        for c in range(self.num_checks):
            edges = self.check_to_edges[c]
            if len(edges) == 0:
                continue

            # Get incoming messages for this check: (batch, degree)
            incoming = msg_q2c[:, edges]

            # For each outgoing edge, compute product of signs and min of magnitudes
            # excluding that edge
            signs = torch.sign(incoming)  # (batch, degree)
            mags = torch.abs(incoming)    # (batch, degree)

            # Product of all signs
            total_sign = torch.prod(signs, dim=1, keepdim=True)  # (batch, 1)

            # Apply syndrome: flip sign if syndrome is 1
            syndrome_sign = 1 - 2 * syndromes[:, c:c+1]  # (batch, 1)
            total_sign = total_sign * syndrome_sign

            # For each edge, divide out its sign to get product of others
            outgoing_signs = total_sign / (signs + 1e-10)  # (batch, degree)

            # Min magnitude excluding each edge
            # Use second minimum trick
            sorted_mags, _ = torch.sort(mags, dim=1)
            min_mag = sorted_mags[:, 0:1]      # (batch, 1)
            second_min = sorted_mags[:, 1:2] if sorted_mags.shape[1] > 1 else min_mag

            # For each edge: if it has the min, use second_min; else use min
            is_min = (mags == min_mag)
            outgoing_mags = torch.where(is_min, second_min, min_mag)  # (batch, degree)

            # Apply min-sum scaling factor
            msg_c2q[:, edges] = self.ms_scaling_factor * outgoing_signs * outgoing_mags

        return msg_c2q

    def _check_to_qubit_sumproduct(self, msg_q2c: torch.Tensor, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Compute check-to-qubit messages using sum-product algorithm.

        Uses the formula: LLR_c2q = 2 * atanh(prod_{q' != q} tanh(LLR_q2c / 2))

        Args:
            msg_q2c: (batch_size, num_edges) - qubit-to-check LLRs
            syndromes: (batch_size, num_checks) - syndrome values

        Returns:
            msg_c2q: (batch_size, num_edges) - check-to-qubit LLRs
        """
        batch_size = msg_q2c.shape[0]
        msg_c2q = torch.zeros(batch_size, self.num_edges, device=self.device)

        # Process each check
        for c in range(self.num_checks):
            edges = self.check_to_edges[c]
            if len(edges) == 0:
                continue

            # Get incoming messages for this check: (batch, degree)
            incoming = msg_q2c[:, edges]

            # Compute tanh(LLR/2) for sum-product
            # Clamp to avoid numerical issues with tanh
            half_llr = torch.clamp(incoming / 2, min=-20, max=20)
            tanh_vals = torch.tanh(half_llr)  # (batch, degree)

            # Product of all tanh values
            total_prod = torch.prod(tanh_vals, dim=1, keepdim=True)  # (batch, 1)

            # Apply syndrome: flip sign if syndrome is 1
            syndrome_sign = 1 - 2 * syndromes[:, c:c+1]  # (batch, 1)
            total_prod = total_prod * syndrome_sign

            # For each edge, divide out its tanh to get product of others
            # Add small epsilon to avoid division by zero
            outgoing_prod = total_prod / (tanh_vals + 1e-10)  # (batch, degree)

            # Clamp to valid range for atanh (-1, 1)
            outgoing_prod = torch.clamp(outgoing_prod, min=-1 + 1e-7, max=1 - 1e-7)

            # Convert back: 2 * atanh(prod)
            msg_c2q[:, edges] = 2 * torch.atanh(outgoing_prod)

        return msg_c2q

    def _qubit_to_check(self, msg_c2q: torch.Tensor, channel_llr: torch.Tensor) -> torch.Tensor:
        """
        Compute qubit-to-check messages.

        Args:
            msg_c2q: (batch_size, num_edges) - check-to-qubit LLRs
            channel_llr: (num_qubits,) - channel LLRs

        Returns:
            msg_q2c: (batch_size, num_edges) - qubit-to-check LLRs
        """
        batch_size = msg_c2q.shape[0]
        msg_q2c = torch.zeros(batch_size, self.num_edges, device=self.device)

        # Process each qubit
        for q in range(self.num_qubits):
            edges = self.qubit_to_edges[q]
            if len(edges) == 0:
                continue

            # Get incoming messages for this qubit: (batch, degree)
            incoming = msg_c2q[:, edges]

            # Sum of all incoming messages plus channel
            total_sum = incoming.sum(dim=1, keepdim=True) + channel_llr[q]  # (batch, 1)

            # For each edge, subtract its contribution
            msg_q2c[:, edges] = total_sum - incoming

        return msg_q2c

    def _compute_marginals(self, msg_c2q: torch.Tensor, channel_llr: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior marginals from messages.

        Args:
            msg_c2q: (batch_size, num_edges) - check-to-qubit LLRs
            channel_llr: (num_qubits,) - channel LLRs

        Returns:
            marginals: (batch_size, num_qubits) - P(error=1)
        """
        batch_size = msg_c2q.shape[0]
        marginals = torch.zeros(batch_size, self.num_qubits, device=self.device)

        for q in range(self.num_qubits):
            edges = self.qubit_to_edges[q]

            # Total LLR = channel + sum of all incoming
            if len(edges) > 0:
                total_llr = channel_llr[q] + msg_c2q[:, edges].sum(dim=1)
            else:
                total_llr = channel_llr[q].expand(batch_size)

            # Convert LLR to probability: P(1) = 1 / (1 + exp(LLR))
            marginals[:, q] = torch.sigmoid(-total_llr)

        return marginals

