import torch
from torch import nn

class FastSpeechV2(nn.Module):
    def __init__(self, n_layers: int, n_heads: int) -> None:
        super().__init__()