from torch.utils.data import Dataset
import torch
import numpy as np
import os
from scipy.io import wavfile
import sounddevice as sd

class VoiceData(Dataset):

    def __init__(self):

        self.data_dir = './recordings/'
        self.file_paths = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self,idx):
        fp = self.file_paths[idx]

        # get y

        y = int(fp[0])

        #  
        self.fs, wv = wavfile.read(self.data_dir+fp)
        assert self.fs==8000
        wv = wv[:12000]
        x=np.zeros((12000), dtype=np.float32)
        x[:len(wv)] = wv
        x = x/np.max(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        return (x,y)
        
        
    def play(self, idx):
        x, y = self.__getitem__(idx)
        sd.play(x, self.fs, blocking=True)
        print(y)
    

