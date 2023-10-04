import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn


class Audio2Mel(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            sampling_rate=22050,
            n_mel_channels=80,
            mel_fmin=0.0,
            mel_fmax=None,
    ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ##############################################
        # FFT Parameters
        ##############################################
        window = torch.hann_window(win_length, device=self.device).float()
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                                   n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).to(self.device).float()
        #  Note: parameters in your model, which should be saved and restored in the state_dict,
        #   but not trained by the optimizer, you should register them as Buffers
        #   Allows to save and load these tensors when you save and load the entire model state
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio_in):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio_in, (p, p), "reflect").squeeze(1)
        # Short time Fourier transform
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1]))
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec
