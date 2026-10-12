import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # TODO: add the remaining PatchGAN layers
        self.output = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, grayscale, color):
        x = torch.cat((grayscale, color), dim=1)
        # TODO: run the image pair through the finished network
        raise NotImplementedError("Discriminator layers still need finishing")

