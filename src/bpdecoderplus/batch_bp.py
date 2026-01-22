import torch
import numpy as np
from typing import Tuple

class BatchBPDecoder:
    """Batch Belief Propagation decoder for parallel syndrome processing on GPU."""

    def __init__(self, H: np.ndarray, channel_probs: np.ndarray, device='cuda'):
        """
        Args:
            H: Parity check matrix (num_checks, num_qubits)
            channel_probs: Physical error rates per qubit (num_qubits,)
            device: torch device
        """
        self.device = torch.device(device)
        self.H = torch.from_numpy(H).to(dtype=torch.float32, device=self.device)
        self.num_checks, self.num_qubits = H.shape

        # Store channel probs
        self.channel_probs = torch.from_numpy(channel_probs).to(dtype=torch.float32, device=self.device)

        # Build edge lists for message passing
        self.check_edges, self.qubit_edges = torch.where(self.H)
        self.num_edges = len(self.check_edges)

    def decode(self, syndromes: torch.Tensor, max_iter: int = 20, damping: float = 0.2, method: str = 'sum-product') -> torch.Tensor:
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

        # Initialize messages as probabilities: (batch_size, num_edges, 2)
        # msg[b, e, 0] = P(qubit=0), msg[b, e, 1] = P(qubit=1)
        msg_c2q = torch.ones(batch_size, self.num_edges, 2, device=self.device) * 0.5
        msg_q2c = torch.ones(batch_size, self.num_edges, 2, device=self.device) * 0.5

        # Initialize qubit-to-check with priors
        for e in range(self.num_edges):
            q = self.qubit_edges[e]
            p = self.channel_probs[q]
            msg_q2c[:, e, 0] = 1 - p
            msg_q2c[:, e, 1] = p

        for _ in range(max_iter):
            msg_c2q_new = torch.zeros_like(msg_c2q)

            # Check to qubit messages (sum-product with parity constraint)
            for c in range(self.num_checks):
                edge_mask = (self.check_edges == c)
                edges_in_check = torch.where(edge_mask)[0]

                if len(edges_in_check) == 0:
                    continue

                # For each edge in this check
                for i, edge_idx in enumerate(edges_in_check):
                    # Get other edges
                    other_edges = [edges_in_check[j] for j in range(len(edges_in_check)) if j != i]

                    if len(other_edges) == 0:
                        # Single variable check - just pass syndrome
                        msg_c2q_new[:, edge_idx, 0] = 1 - syndromes[:, c]
                        msg_c2q_new[:, edge_idx, 1] = syndromes[:, c]
                    else:
                        # Compute marginal over other variables satisfying parity
                        other_msgs = msg_q2c[:, other_edges, :]  # (batch, len(other_edges), 2)

                        if method == 'min-sum':
                            # Use minimum-sum approximation
                            msg = self._compute_minsum_check_to_qubit(other_msgs, syndromes[:, c])
                            msg_c2q_new[:, edge_idx, :] = msg
                        else:
                            # Use sum-product (default)
                            # For x_i=0: sum over configs with parity = syndrome
                            # For x_i=1: sum over configs with parity = 1-syndrome
                            p0 = self._compute_parity_marginal(other_msgs, syndromes[:, c], target_parity=0)
                            p1 = self._compute_parity_marginal(other_msgs, syndromes[:, c], target_parity=1)

                            # Normalize
                            total = p0 + p1 + 1e-10
                            msg_c2q_new[:, edge_idx, 0] = p0 / total
                            msg_c2q_new[:, edge_idx, 1] = p1 / total

            # Damping
            msg_c2q = damping * msg_c2q + (1 - damping) * msg_c2q_new

            # Qubit to check messages
            msg_q2c_new = torch.zeros_like(msg_q2c)
            for q in range(self.num_qubits):
                edge_mask = (self.qubit_edges == q)
                edges_in_qubit = torch.where(edge_mask)[0]

                if len(edges_in_qubit) == 0:
                    continue

                prior = self.channel_probs[q]

                for i, edge_idx in enumerate(edges_in_qubit):
                    # Product of prior and all other incoming messages
                    p0 = torch.ones(batch_size, device=self.device) * (1 - prior)
                    p1 = torch.ones(batch_size, device=self.device) * prior

                    for j, other_edge in enumerate(edges_in_qubit):
                        if j != i:
                            p0 = p0 * msg_c2q[:, other_edge, 0]
                            p1 = p1 * msg_c2q[:, other_edge, 1]

                    # Normalize
                    total = p0 + p1 + 1e-10
                    msg_q2c_new[:, edge_idx, 0] = p0 / total
                    msg_q2c_new[:, edge_idx, 1] = p1 / total

            msg_q2c = msg_q2c_new

            # Check convergence by syndrome satisfaction
            # Compute hard decision from current marginals
            temp_marginals = torch.zeros(batch_size, self.num_qubits, device=self.device)
            for q in range(self.num_qubits):
                edge_mask = (self.qubit_edges == q)
                edges_in_qubit = torch.where(edge_mask)[0]
                prior = self.channel_probs[q]
                p0 = torch.ones(batch_size, device=self.device) * (1 - prior)
                p1 = torch.ones(batch_size, device=self.device) * prior
                for edge_idx in edges_in_qubit:
                    p0 = p0 * msg_c2q[:, edge_idx, 0]
                    p1 = p1 * msg_c2q[:, edge_idx, 1]
                total = p0 + p1 + 1e-10
                temp_marginals[:, q] = p1 / total

            decoding = (temp_marginals > 0.5).float()
            computed_syndrome = (self.H @ decoding.T).T % 2
            converged = (computed_syndrome == syndromes).all(dim=1)
            if converged.all():
                break

        # Compute marginals
        marginals = torch.zeros(batch_size, self.num_qubits, device=self.device)
        for q in range(self.num_qubits):
            edge_mask = (self.qubit_edges == q)
            edges_in_qubit = torch.where(edge_mask)[0]

            prior = self.channel_probs[q]
            p0 = torch.ones(batch_size, device=self.device) * (1 - prior)
            p1 = torch.ones(batch_size, device=self.device) * prior

            for edge_idx in edges_in_qubit:
                p0 = p0 * msg_c2q[:, edge_idx, 0]
                p1 = p1 * msg_c2q[:, edge_idx, 1]

            # Normalize and return P(error=1)
            total = p0 + p1 + 1e-10
            marginals[:, q] = p1 / total

        return marginals

    def _compute_parity_marginal(self, msgs: torch.Tensor, syndrome: torch.Tensor, target_parity: int) -> torch.Tensor:
        """
        Compute marginal probability that variables have given parity.

        Args:
            msgs: (batch, num_vars, 2) - probability messages
            syndrome: (batch,) - observed syndrome
            target_parity: 0 or 1

        Returns:
            prob: (batch,) - probability of target parity
        """
        batch_size, num_vars, _ = msgs.shape

        if num_vars == 0:
            return torch.ones(batch_size, device=self.device)

        # Use dynamic programming to compute parity distribution
        # dp[b, v, p] = probability that first v variables have parity p
        dp = torch.zeros(batch_size, num_vars + 1, 2, device=self.device)
        dp[:, 0, 0] = 1.0  # Base case: 0 variables have parity 0

        for v in range(num_vars):
            for p in range(2):
                # If variable v is 0, parity stays same
                dp[:, v + 1, p] += dp[:, v, p] * msgs[:, v, 0]
                # If variable v is 1, parity flips
                dp[:, v + 1, 1 - p] += dp[:, v, p] * msgs[:, v, 1]

        # Return probability of desired parity given syndrome
        desired_parity = (syndrome.long() + target_parity) % 2
        result = torch.zeros(batch_size, device=self.device)
        for b in range(batch_size):
            result[b] = dp[b, num_vars, desired_parity[b]]

        return result

    def _compute_minsum_check_to_qubit(self, other_msgs: torch.Tensor, syndrome: torch.Tensor, scaling_factor: float = 0.625) -> torch.Tensor:
        """
        Compute check-to-qubit message using minimum-sum approximation.

        Args:
            other_msgs: (batch, num_other_qubits, 2) - messages from other qubits
            syndrome: (batch,) - syndrome value for this check
            scaling_factor: Scaling factor for min-sum (default 0.625)

        Returns:
            msg: (batch, 2) - message to target qubit [P(0), P(1)]
        """
        batch_size = other_msgs.shape[0]
        num_other = other_msgs.shape[1]

        if num_other == 0:
            # No other qubits - just return syndrome
            msg = torch.zeros(batch_size, 2, device=self.device)
            msg[:, 0] = 1 - syndrome
            msg[:, 1] = syndrome
            return msg

        # Convert probabilities to LLRs: LLR = log(P(0)/P(1))
        llrs = torch.log((other_msgs[:, :, 0] + 1e-10) / (other_msgs[:, :, 1] + 1e-10))

        # Compute sign product and minimum magnitude
        sign_product = torch.prod(torch.sign(llrs), dim=1)  # (batch,)
        min_magnitude = torch.min(torch.abs(llrs), dim=1)[0]  # (batch,)

        # Apply syndrome: if syndrome=1, flip sign
        sign_product = sign_product * (1 - 2 * syndrome)

        # Output LLR with scaling
        output_llr = scaling_factor * sign_product * min_magnitude

        # Convert back to probabilities
        prob_0 = torch.sigmoid(output_llr)
        prob_1 = 1 - prob_0

        return torch.stack([prob_0, prob_1], dim=1)

