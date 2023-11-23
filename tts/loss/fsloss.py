import torch
from torch import nn

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, input, target):
        return self.mse(torch.log(input + 1), torch.log(target + 1))


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec_loss = nn.MSELoss()
        self.duration_loss = MSLELoss()
    
    def forward(self, mel_target, duration, pred_mel, pred_duration, **batch):
        return self.mel_spec_loss(pred_mel, mel_target) + self.duration_loss(pred_duration, duration)
    
class FastSpeech2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.mel_spec_loss = nn.MSELoss()
        self.duration_loss = MSLELoss()
        self.pitch_loss = MSLELoss()
        self.energy_loss = MSLELoss()
    
    def forward(self, mel_target, duration, pitch, energy, pred_mel, pred_duration, pred_pitch, pred_energy, **batch):
        return self.mel_spec_loss(pred_mel, mel_target) + (self.duration_loss(pred_duration, duration) + self.pitch_loss(pred_pitch, pitch) + self.energy_loss(pred_energy, energy)) / 3