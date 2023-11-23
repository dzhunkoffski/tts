import glob
from typing import List
import os

import pandas as pd
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset

from tts.text import text_to_sequence

from sklearn.model_selection import train_test_split

class LJSpeechDataset(Dataset):
    def __init__(
            self, dataset_path: str, 
            is_train: bool, train_size: float, 
            text_cleaners: List[str], mel_spec_path: str, energy_path: str, pitch_contour_path: str,
            alignment_path: str, sr: int, backend: str = 'soundfile', dataset_size: int = -1, *args, **kwargs):
        self.wavs = glob.glob(f'{dataset_path}/wavs/*.wav')
        self.wavs.sort()
        self.texts = pd.read_csv(f'{dataset_path}/metadata.csv', sep='|', header=None)[1]
        self.texts = self.texts.values.tolist()
        # with open(f'{dataset_path}/train.txt', 'r') as fd:
        #     self.texts = fd.readlines()
        self.text_cleaners = text_cleaners
        self.mel_spec_path = mel_spec_path
        self.alignment_path = alignment_path
        self.pitch_contour_path = pitch_contour_path
        self.energy_path = energy_path
        self.sample_rate = sr
        self.mel_specs = glob.glob(f'{mel_spec_path}/*.npy')
        self.mel_specs.sort()
        self.backend = backend
        train_indexes, val_indexes = train_test_split(list(range(len(self.texts))), train_size=train_size, random_state=42)
        if is_train:
            self.ixs = train_indexes
        else:
            self.ixs = val_indexes

        broken_indexes = []
        for ix in self.ixs:
            text = text_to_sequence(self.texts[ix], self.text_cleaners)
            duration = np.load(os.path.join(self.alignment_path, str(ix) + ".npy"))
            if len(text) != len(duration):
                broken_indexes.append(ix)
        print('BROKEN IXS:', len(broken_indexes))
        for broken_ix in broken_indexes:
            self.ixs.remove(broken_ix)
        
        self.ixs = self.ixs[:dataset_size]
        
        self.pitch_min, self.pitch_max = self._get_min_max_pitch()
        self.energy_min, self.energy_max = self._get_min_max_energy()
        
    def __len__(self):
        return len(self.ixs)
    
    def _get_min_max_pitch(self):
        pitch_min = 10000000
        pitch_max = 0
        for i in self.ixs:
            pitch_contour = np.load(os.path.join(self.pitch_contour_path, str(i) + ".npy"))
            pitch_min = min(pitch_min, pitch_contour.min())
            pitch_max = max(pitch_max, pitch_contour.max())
        return pitch_min, pitch_max
    
    def _get_min_max_energy(self):
        energy_min = 10000000
        energy_max = 0
        for i in self.ixs:
            energy = np.load(os.path.join(self.energy_path, str(i) + ".npy"))
            energy_min = min(energy_min, energy.min())
            energy_max = max(energy_max, energy.max())
        return energy_min, energy_max
    
    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0, :]
        if sr != self.sample_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.sample_rate)
        return audio_tensor
    
    def text2tokens(self, text):
        text = text_to_sequence(text, self.text_cleaners)
        text = torch.LongTensor(text)
        return text
    
    def __getitem__(self, item):
        item = self.ixs[item]
        character = self.text2tokens(self.texts[item])

        # audio_wav = self.load_audio(self.wavs[item])
        audio_tensor_spec = np.load(self.mel_specs[item])
        audio_tensor_spec = torch.tensor(audio_tensor_spec).t()

        duration = np.load(os.path.join(self.alignment_path, str(item) + ".npy"))
        duration = torch.tensor(duration, dtype=torch.float)

        energy = np.load(os.path.join(self.energy_path, str(item) + ".npy"))
        energy = torch.tensor(energy, dtype=torch.float)

        pitch = np.load(os.path.join(self.pitch_contour_path, str(item) + ".npy"))
        pitch = torch.tensor(pitch, dtype=torch.float)

        return {
            "raw_text": self.texts[item],
            "text": character, 
            "duration": duration, 
            "pitch": pitch,
            "energy": energy,
            "mel_target": audio_tensor_spec,
            "mel_spec_path": self.mel_specs[item],
            "alignment_path": os.path.join(self.alignment_path, str(item) + ".npy")
        }
        
