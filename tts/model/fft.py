import torch
from torch import nn

from tts.model.utils import Conv1D

class PreNormFFTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int, n_channels: int, dropout: float = 0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.lay_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.lay_norm2 = nn.LayerNorm(embed_dim)
        self.conv = nn.Sequential(
            Conv1D(in_channels=embed_dim, out_channels=n_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            Conv1D(in_channels=n_channels, out_channels=embed_dim, kernel_size=kernel_size, padding='same')
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, non_pad_mask=None, attn_mask=None):
        output = self.lay_norm1(x)
        q = self.q_proj(output)
        k = self.k_proj(output)
        v = self.v_proj(output)
        output, _ = self.multihead_attention(q, k, v, attn_mask=attn_mask)
        output = self.dropout(output)
        x = x + output

        output = self.lay_norm2(x)
        output = self.conv(output)
        output = self.dropout(output)
        x = x + output

        return x

class FFTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int, n_channels: int, dropout: float = 0) -> None:
        super().__init__()
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.lay_norm1 = nn.LayerNorm(embed_dim)
        self.conv = nn.Sequential(
            Conv1D(in_channels=embed_dim, out_channels=n_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            Conv1D(in_channels=n_channels, out_channels=embed_dim, kernel_size=kernel_size, padding='same')
        )
        self.lay_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, non_pad_mask = None, attn_mask = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        output, _ = self.multihead_attention(q, k, v, attn_mask=attn_mask)
        output = self.dropout(output)
        output = x + output
        x = self.lay_norm1(output)

        if non_pad_mask is not None:
            x = x * non_pad_mask

        output = self.conv(x)
        output = self.dropout(output)
        output = x + output
        x = self.lay_norm2(output)

        if non_pad_mask is not None:
            x = x * non_pad_mask

        return x