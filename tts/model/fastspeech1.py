import torch
from torch import nn

from tts.model.utils import PositionalEncoding, Conv1D
from tts import waveglow
from tts import text
from tts import audio
from tts import vocode_utils
from tts import waveglow

class FFTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int, dropout: float = 0) -> None:
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
            Conv1D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            Conv1D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding='same')
        )
        self.lay_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        output, _ = self.multihead_attention(q, k, v)
        output = self.dropout(output)
        output = x + output
        x = self.lay_norm1(output)

        output = self.conv(x)
        output = self.dropout(output)
        output = x + output
        x = self.lay_norm2(output)

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
        max_seq_len = 0
        hidden_mels = []
        for batch_ix in range(hidden_phonems.size()[0]):
            indexes = torch.arange(start=0, end=durations.size()[-1]).to(hidden_phonems.device)
            repeated_indexes = torch.repeat_interleave(indexes, durations[batch_ix])
            hidden_mel = hidden_phonems[batch_ix, repeated_indexes, :]
            hidden_mels.append(hidden_mel)
            max_seq_len = max(max_seq_len, hidden_mel.size()[0])
        
        output_mels = []
        for hidden_mel in hidden_mels:
            output_mels.append(
                torch.nn.functional.pad(
                    input=hidden_mel.t(), pad=(0, max_seq_len - hidden_mel.size()[0]),
                    mode='constant', value=0
                ).t()
            )
        output_mels = torch.stack(output_mels).to(hidden_phonems.device)
        return output_mels

class FastSpeechV1(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, pad_idx: int, n_blocks: int, n_heads: int, fft_kernel: int, lr_kernel: int, embed_dim: int, n_mels: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        self.phoneme_blocks = nn.Sequential(*[
            FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout) for _ in range(n_blocks)
        ])
        # TODO: padding attention masks
        self.duration_predictor = DurationPredictor(embed_dim=embed_dim, kernel_size=lr_kernel, dropout=dropout)
        self.LR = LengthRegulator()
        self.mel_blocks = nn.Sequential(*[
            FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout) for _ in range(n_blocks)
        ])
        self.mel_linear = nn.Linear(embed_dim, n_mels)
        self.vocoder = self._load_vocoder()

    def _load_vocoder(glow_state_dict):
        vocoder = vocode_utils.get_WaveGlow()
        return vocoder

    
    def forward(self, text, duration, **batch):
        x = self.embedding_layer(text)
        x = self.pos_enc(x)
        x = self.phoneme_blocks(x)
        pred_durations = self.duration_predictor(x)
        x = self.LR(x, duration)
        x = self.pos_enc(x)
        x = self.mel_blocks(x)
        predicted_mel = self.mel_linear(x)
        pred_durations = torch.maximum(pred_durations, torch.ones_like(pred_durations))
        # TODO: make sure output predicted mel the same as target mel
        return {"pred_mel": predicted_mel, "pred_duration": pred_durations}
    
    @torch.inference_mode()
    def text2voice(self, text: str, dataset):
        x = dataset.text2tokens(text)
        x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        x = self.embedding_layer(x)
        x = self.pos_enc(x)
        x = self.phoneme_blocks(x)
        pred_durations = self.duration_predictor(x)
        pred_durations = torch.maximum(pred_durations, torch.ones_like(pred_durations))
        pred_durations = pred_durations.int()
        x = self.LR(x, pred_durations)
        x = self.pos_enc(x)
        x = self.mel_blocks(x)
        predicted_mel = self.mel_linear(x)
        predicted_mel = torch.permute(predicted_mel, (0, 2, 1))
        audio = waveglow.inference.inference_audio(predicted_mel, self.vocoder)
        return audio


