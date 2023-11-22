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
            text_cleaners: List[str], mel_spec_path: str,
            alignment_path: str, sr: int, backend: str = 'soundfile', dataset_size: int = -1, *args, **kwargs):
        self.wavs = glob.glob(f'{dataset_path}/wavs/*.wav')
        self.wavs.sort()
        self.texts = pd.read_csv(f'{dataset_path}/metadata.csv', sep='|', header=None)[1]
        self.texts = self.texts.values.tolist()
        self.text_cleaners = text_cleaners
        self.mel_spec_path = mel_spec_path
        self.alignment_path = alignment_path
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
        
    def __len__(self):
        return len(self.ixs)
    
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
        duration = torch.tensor(duration)

        return {
            "raw_text": self.texts[item],
            "text": character, 
            "duration": duration, 
            "mel_target": audio_tensor_spec,
            "mel_spec_path": self.mel_specs[item],
            "alignment_path": os.path.join(self.alignment_path, str(item) + ".npy")
        }
        
