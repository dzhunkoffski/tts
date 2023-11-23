import pyworld as pw
import glob
from scipy.io import wavfile
from tqdm import tqdm
from tts.audio import hparams_audio
import os
import numpy as np

wav_list = glob.glob('data/LJSpeech-1.1/wavs/**.wav')
wav_list.sort()
os.mkdir('pitch_contour')
for i, wav_path in tqdm(list(enumerate(wav_list))):
    sr, data = wavfile.read(wav_path)
    data = data.astype(float)
    _f0, t = pw.dio(data, sr, frame_period=hparams_audio.hop_length * 1000 / hparams_audio.sampling_rate)
    f0 = pw.stonemask(data, _f0, t, sr)
    with open(f'pitch_contour/{i}.npy', 'wb') as fd:
        np.save(fd, f0)