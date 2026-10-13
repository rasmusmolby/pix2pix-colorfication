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


# Smoketest
if __name__ == "__main__":
    model = Discriminator().eval()
    grayscale = torch.randn(1, 1, 64, 64)
    color = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        predictions = model(grayscale, color)
    assert predictions.ndim == 4 and predictions.shape[:2] == (1, 1)
    assert predictions.shape[2] > 0 and predictions.shape[3] > 0
    assert torch.isfinite(predictions).all()
    print(f"Disc smoke test : {tuple(predictions.shape)}")

