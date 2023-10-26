import pandas as pd
import os
import torch
import torchaudio
import random
from utils import convert_audio

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        if mode == 'train':
            file_dir = config.datasets.train_csv_path
        elif mode == 'test':
            file_dir = config.datasets.test_csv_path
        file_dir = os.path.normpath(file_dir)
        assert len(config.common.main_dir) > 0
        assert os.path.exists(file_dir), "Given: "+file_dir
        self.audio_files = pd.read_csv(file_dir, sep="/n", on_bad_lines='skip')
        self.transform = transform
        # Num of samples
        self.fixed_length = config.datasets.fixed_length
        # Cut the length of audio
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)  

    def __getitem__(self, idx):
        # Get path and load, waveform: tensor
        waveform, sample_rate = torchaudio.load(self.audio_files.iloc[idx, :].values[0])
        # Preprocess the waveform's sample rate
        if sample_rate != self.sample_rate:
            waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        if self.transform:
            waveform = self.transform(waveform)

        # Cut the length of audio
        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                # random start point
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
                # cut tensor
                waveform = waveform[:, start:start+self.tensor_cut]
                return waveform, self.sample_rate
            else:
                return waveform, self.sample_rate
        

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch

#
def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors