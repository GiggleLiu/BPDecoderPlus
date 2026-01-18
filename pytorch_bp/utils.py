"""
Utility functions for Belief Propagation.
"""

from typing import List
import torch


def deep_copy_messages(messages: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
    """
    Deep copy message structure.
    
    Args:
        messages: Nested list of message tensors
    
    Returns:
        Deep copy of messages
    """
    return [[msg.clone() for msg in var_msgs] for var_msgs in messages]
