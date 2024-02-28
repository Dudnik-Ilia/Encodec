import hydra
import torch
from torch import nn

from model import EncodecModel
from msstftd import DiscriminatorSTFT, MultiScaleSTFTDiscriminator

CHECKPOINT_OLD = "/home/woody/iwi1/iwi1010h/checkpoints/680242/bs3_cut36000_length0ep12_lr0.0003.pt"


class AudioDiscriminator(nn.Module):
    def __init__(self):
        super(AudioDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1)
        x = self.fc_layers(x)
        return x

class MAudioDiscriminator(nn.Module):
    def __init__(self, num_classifiers=3):
        super(MAudioDiscriminator, self).__init__()
        self.num_classifiers = num_classifiers
        self.classifiers = nn.ModuleList([AudioDiscriminator() for i in range(num_classifiers)])

    def forward(self, logits: list):
        pred = 0.
        for i in range(self.num_classifiers):
            pred += self.classifiers[i](logits[i])
        return pred / self.num_classifiers

@hydra.main(config_path='config', config_name='config')
def main(config):
    #model = EncodecModel.my_encodec_model(checkpoint=CHECKPOINT_OLD)

    disc_model = MultiScaleSTFTDiscriminator(filters=config.model.filters,
                                             hop_lengths=config.model.disc_hop_lengths,
                                             win_lengths=config.model.disc_win_lengths,
                                             n_ffts=config.model.disc_n_ffts)

    classifier = MAudioDiscriminator()

    x = torch.rand(size=(1, 1, 48000))
    logits, _ = disc_model(x)

    pred = classifier(logits)

    return pred


if __name__ == '__main__':
    main()