import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        