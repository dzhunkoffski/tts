import torch
from torch import nn

from tts.base.base_metric import BaseMetric

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, input, target):
        return self.mse(torch.log(input + 1), torch.log(target + 1))

class DurationLossMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MSLELoss()

    def __call__(self, duration, pred_duration, **batch):
        return self.loss(pred_duration, duration)

class PitchLossMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MSLELoss()
    
    def __call__(self, pitch, pred_pitch, **batch):
        return self.loss(pred_pitch, pitch)

class EnergyLossMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MSLELoss()
    
    def __call__(self, energy, pred_energy, **batch):
        return self.loss(pred_energy, energy)

class MelLossMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.MSELoss()

    def __call__(self, mel_target, pred_mel, **batch):
        return self.loss(pred_mel, mel_target)