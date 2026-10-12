import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # TODO: add the rest of the encoder and decoder, with skip connections
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x is the grayscale image
        raise NotImplementedError("Generator forward pass still needs the decoder")

