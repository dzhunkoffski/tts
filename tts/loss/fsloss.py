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
        target_mel_seqlen = mel_target.size()[-1]
        pred_mel = torch.permute(pred_mel, (0, 2, 1))
        pred_mel = pred_mel[:, :, :target_mel_seqlen]
        return self.mel_spec_loss(pred_mel, mel_target) + self.duration_loss(pred_duration, duration)
