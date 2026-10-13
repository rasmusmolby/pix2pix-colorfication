import torch
import torch.nn as nn


class Generator(nn.Module):
    # TODO Add a class comment explaining abit about this module. Maybe abit about pix2pix?
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        # Shrink the image while learning features from its shapes and edges.
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def down_block(in_features, out_features):
            # ENCODER BLOCK
            return nn.Sequential(
                nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def up_block(in_features, out_features, dropout=False):
            # DECODER BLOCK
            layers = [
                nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                # DROPOUT
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        # Compress images to smaller dimensions with encoder
        self.down8 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up1 = up_block(512, 512, dropout=True)
        self.up2 = up_block(1024, 512, dropout=True)
        self.up3 = up_block(1024, 512, dropout=True)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        # Restore the full image size and keep color values between -1 and 1 (TODO double check when changing color library?)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
      # ENCODER
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        down8 = self.down8(down7)
        # DECODER
        up1 = torch.cat((self.up1(down8), down7), dim=1)
        up2 = torch.cat((self.up2(up1), down6), dim=1)
        up3 = torch.cat((self.up3(up2), down5), dim=1)
        up4 = torch.cat((self.up4(up3), down4), dim=1)
        up5 = torch.cat((self.up5(up4), down3), dim=1)
        up6 = torch.cat((self.up6(up5), down2), dim=1)
        up7 = torch.cat((self.up7(up6), down1), dim=1)
        return self.output(up7)

# Smoketest
if __name__ == "__main__":
    generator = Generator().eval()
    with torch.inference_mode():
        output = generator(torch.randn(1, 1, 256, 256))
    assert output.shape == (1, 3, 256, 256), output.shape
    assert torch.isfinite(output).all(), "No finite values :("
    assert (output >= -1).all() and (output <= 1).all(), "Output is outside color range of -1 to 1 doofus"
    print(f"Gen smoke test : {tuple(output.shape)}")
