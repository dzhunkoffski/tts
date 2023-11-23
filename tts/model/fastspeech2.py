import torch
from torch import nn
import torch.nn.functional as F

from tts import vocode_utils
from tts import waveglow

from tts.model.utils import PositionalEncoding
from tts.model.fft import PreNormFFTBlock, FFTBlock
from tts.model.feature_predictor import FeaturePredictor, LengthRegulator

class VarAdaptor(nn.Module):
    def __init__(self, embed_dim: int, feature_kernel: int, dropout: float, min_pitch: float, max_pitch: float, min_energy: float, max_energy: float, codebook_size: int) -> None:
        super().__init__()
        self.duration_predictor = FeaturePredictor(embed_dim=embed_dim, kernel_size=feature_kernel, dropout=dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = FeaturePredictor(embed_dim=embed_dim, kernel_size=feature_kernel, dropout=dropout)
        self.energy_predictor = FeaturePredictor(embed_dim=embed_dim, kernel_size=feature_kernel, dropout=dropout)
        
        self.pitch_embedding = nn.Embedding(num_embeddings=codebook_size, embedding_dim=embed_dim)
        self.energy_embedding = nn.Embedding(num_embeddings=codebook_size, embedding_dim=embed_dim)

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.codebook_size = codebook_size
    
    def forward(self, x, duration=None, energy=None, pitch=None, duration_coeff: float = 1.0, pitch_coeff: float = 1.0, energy_coeff: float = 1.0, inference: bool = False):
        pred_duration = self.duration_predictor(x)
        if inference:
            pred_duration = pred_duration * duration_coeff
            pred_duration = torch.maximum(pred_duration, torch.ones_like(pred_duration)).int()
            x = self.length_regulator(x, pred_duration)
        else:
            x = self.length_regulator(x, duration)
        
        pred_pitch = self.pitch_predictor(x)
        pred_energy = self.energy_predictor(x)
        if inference:
            pred_pitch = pred_pitch * pitch_coeff
            pred_energy = pred_energy * energy_coeff

            pitch_boundaries = torch.logspace(start=self.min_pitch, end=self.max_pitch, steps=self.codebook_size, device=x.device)[1:-1]
            pitch = torch.bucketize(pred_pitch, boundaries=pitch_boundaries).to(pitch_boundaries.device)
            pitch = self.pitch_embedding(pitch)
            energy_boundaries = torch.linspace(start=self.min_energy, end=self.max_energy, steps=self.codebook_size, device=x.device)[1:-1]
            energy = torch.bucketize(pred_energy, boundaries=energy_boundaries).to(energy_boundaries.device)
            energy = self.energy_embedding(energy)
        else:
            pitch_boundaries = torch.logspace(start=self.min_pitch, end=self.max_pitch, steps=self.codebook_size, device=x.device)[1:-1]
            pitch = torch.bucketize(pitch, boundaries=pitch_boundaries).to(pitch_boundaries.device)
            pitch = self.pitch_embedding(pitch)

            energy_boundaries = torch.linspace(start=self.min_energy, end=self.max_energy, steps=self.codebook_size, device=x.device)[1:-1]
            energy = torch.bucketize(energy, boundaries=energy_boundaries).to(energy_boundaries.device)
            energy = self.energy_embedding(energy)

        x = x + pitch
        x = x + 0.000001 * energy
        return x, pred_duration, pred_pitch, pred_energy
            
            
        
class FastSpeechV2(nn.Module):
    def __init__(
            self, max_len: int, vocab_size: int, pad_idx: int, n_blocks: int, 
            n_heads: int, fft_kernel: int, feature_kernel: int, embed_dim: int, n_mels: int, 
            conv_channels: int, min_pitch: float, max_pitch: float, codebook_size: int,
            min_energy: float, max_energy: float, prenorm: bool, dropout: float = 0.0
        ) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.phoneme_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        if prenorm:
            self.encoder = nn.ModuleList([
                PreNormFFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
            ])
            self.decoder = nn.ModuleList([
                PreNormFFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
            ])
        else:
            self.encoder = nn.ModuleList([
                FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
            ])
            self.decoder = nn.ModuleList([
                FFTBlock(embed_dim=embed_dim, num_heads=n_heads, kernel_size=fft_kernel, dropout=dropout, n_channels=conv_channels) for _ in range(n_blocks)
            ])
        self.variance_adaptor = VarAdaptor(
            embed_dim=embed_dim, feature_kernel=feature_kernel, dropout=dropout,
            min_pitch=min_pitch, max_pitch=max_pitch, min_energy=min_energy, 
            max_energy=max_energy, codebook_size=codebook_size
        )
        self.mel_linear = nn.Linear(embed_dim, n_mels)
        self.vocoder = self._load_vocoder()
    
    def _load_vocoder(self):
        vocoder = vocode_utils.get_WaveGlow()
        vocoder = vocoder.cuda()
        return vocoder
    
    def forward(self, text, duration, energy, pitch, **batch):
        x = self.phoneme_embedding_layer(text)
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.encoder[i](x)
        x, pred_duration, pred_pitch, pred_energy = self.variance_adaptor(
            x, duration=duration, energy=energy, pitch=pitch, inference=False
        )
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.decoder[i](x)
        predicted_mel = self.mel_linear(x)
        predicted_mel = torch.permute(predicted_mel, (0, 2, 1))
        pred_duration = F.relu(pred_duration)
        return {"pred_mel": predicted_mel, "pred_duration": pred_duration, "pred_pitch": pred_pitch, "pred_energy": pred_energy}

    @torch.inference_mode()
    def text2voice(self, text: str, dataset, duration_coeff=1.0, pitch_coeff=1.0, energy_coeff=1.0):
        x = dataset.text2tokens(text)
        x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)

        x = self.phoneme_embedding_layer(x)
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.encoder[i](x)
        x, _, _, _ = self.variance_adaptor(
            x, inference=True, duration_coeff=duration_coeff, pitch_coeff=pitch_coeff, energy_coeff=energy_coeff
        )
        x = self.pos_enc(x)
        for i in range(self.n_blocks):
            x = self.decoder[i](x)
        predicted_mel = self.mel_linear(x)
        predicted_mel = torch.permute(predicted_mel, (0,2,1))
        audio = waveglow.inference.inference_audio(predicted_mel, self.vocoder)
        return {"audio": audio, "mel_spec": predicted_mel}
