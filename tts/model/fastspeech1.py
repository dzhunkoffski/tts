import torch
from torch import nn
import torch.nn.functional as F

from tts.model.utils import PositionalEncoding, Conv1D, create_alignment
from tts.model.feature_predictor import LengthRegulator, FeaturePredictor
from tts.model.fft import FFTBlock

from tts import waveglow
from tts import text
from tts import audio
from tts import vocode_utils
from tts import waveglow
    
class FastSpeechV1(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, pad_idx: int, n_blocks: int, n_heads: int, fft_kernel: int, lr_kernel: int, embed_dim: int, n_mels: int, conv_channels: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        self.phoneme_blocks = nn.ModuleList([
            FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
        ])
        self.duration_predictor = FeaturePredictor(embed_dim=embed_dim, kernel_size=lr_kernel, dropout=dropout)
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


