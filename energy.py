import pyworld as pw
import glob
from scipy.io import wavfile
from tqdm import tqdm
from tts.audio import hparams_audio
import os
import numpy as np
import torch

wav_list = glob.glob('data/LJSpeech-1.1/wavs/**.wav')
wav_list.sort()
os.mkdir('energy')
for i, wav_path in tqdm(list(enumerate(wav_list))):
    sr, data = wavfile.read(wav_path)
    data = data.astype(float)

    e = torch.stft(
        torch.tensor(data), 
        n_fft=hparams_audio.filter_length,
        win_length=hparams_audio.win_length,
        hop_length=hparams_audio.hop_length,
        return_complex=True
    )
    e = e.abs()
    e = torch.norm(e, 2, dim=0)
    e = e.numpy()
    with open(f'energy/{i}.npy', 'wb') as fd:
        np.save(fd, e)