"""
Name: data.py
Function: define data loaders for object detection
"""

import os
import numpy as numpy
import torch
import librosa
from torch import nn, Tensor

class audio_data(torch.utils.data.Dataset):
    def __init__(self, data_dir, sr=16000):
        self.data_dir = data_dir
        self.sr = sr

        #file directories (keep the root file)
        self.files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    
    def __getitem__(self, idx):
        #load audio at 16000 sample rate
        audio, _ = librosa.load(self.files[idx], sr=self.sr, mono=True)

        #to tensor
        audio = torch.from_numpy(audio)
        audio = audio

        return torch.unsqueeze(audio,0)

    def __len__(self):
        return len(self.files)
