import torch
from torch import nn

from tts.model.utils import Conv1D, create_alignment
import torch.nn.functional as F

class FeaturePredictor(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.conv1 = Conv1D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding='same')
        self.lay_norm1 = nn.LayerNorm(embed_dim)
        self.activasion = nn.ReLU()
        self.conv2 = Conv1D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding='same')
        self.lay_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lay_norm1(x)
        x = self.activasion(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.lay_norm2(x)
        x = self.activasion(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = x.squeeze(-1)
        x = F.relu(x)
        return x


class LengthRegulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden_phonems, durations):
        # hidden_fonems: tensor of size [B x SEQ_LEN X HIDDEN_SIZE]
        # durations: tensor of size [B x SEQ_LEN]
        # print(durations.size())
        # print(hidden_phonems.size())
        expand_max_len = torch.max(
            torch.sum(durations, -1), -1)[0]
        alignment = torch.zeros(durations.size(0),
                                expand_max_len,
                                durations.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     durations.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(hidden_phonems.device)

        output = alignment @ hidden_phonems
        return output