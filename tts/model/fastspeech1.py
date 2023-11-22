import torch
from torch import nn
import torch.nn.functional as F

from tts.model.utils import PositionalEncoding, Conv1D, create_alignment
from tts import waveglow
from tts import text
from tts import audio
from tts import vocode_utils
from tts import waveglow

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

class DurationPredictor(nn.Module):
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

class FastSpeechV1(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, pad_idx: int, n_blocks: int, n_heads: int, fft_kernel: int, lr_kernel: int, embed_dim: int, n_mels: int, conv_channels: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        self.phoneme_blocks = nn.ModuleList([
            FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
        ])
        self.duration_predictor = DurationPredictor(embed_dim=embed_dim, kernel_size=lr_kernel, dropout=dropout)
        self.LR = LengthRegulator()
        self.mel_blocks = nn.ModuleList([
            FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
        ])
        self.mel_linear = nn.Linear(embed_dim, n_mels)
        self.vocoder = self._load_vocoder()

    def _load_vocoder(self):
        vocoder = vocode_utils.get_WaveGlow()
        vocoder = vocoder.cuda()
        return vocoder

    
    def forward(self, text, duration, **batch):
        x = self.embedding_layer(text)
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.phoneme_blocks[i](x)
        pred_durations = self.duration_predictor(x)
        x = self.LR(x, duration)
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.mel_blocks[i](x)
        predicted_mel = self.mel_linear(x)
        predicted_mel = torch.permute(predicted_mel, (0, 2, 1))
        pred_durations = F.relu(pred_durations)
        return {"pred_mel": predicted_mel, "pred_duration": pred_durations}
    
    @torch.inference_mode()
    def text2voice(self, text: str, dataset):
        x = dataset.text2tokens(text)
        x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        x = self.embedding_layer(x)
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.phoneme_blocks[i](x)
        pred_durations = self.duration_predictor(x)
        pred_durations = torch.maximum(pred_durations, torch.ones_like(pred_durations))
        pred_durations = pred_durations.int()
        x = self.LR(x, pred_durations)
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.mel_blocks[i](x)
        predicted_mel = self.mel_linear(x)
        predicted_mel = torch.permute(predicted_mel, (0, 2, 1))
        audio = waveglow.inference.inference_audio(predicted_mel, self.vocoder)
        return audio


