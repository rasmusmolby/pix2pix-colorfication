import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn


class Discriminator(nn.Module):
        # TODO Add a class comment explaining abit about this module?
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
@hydra.main(config_path="../configs", config_name="model")
def main(cfg: DictConfig):
    model = Discriminator(cfg.in_channels, cfg.out_channels).eval()
    grayscale = torch.randn(
        cfg.batch_size, cfg.in_channels, cfg.image_size, cfg.image_size
    )
    color = torch.randn(
        cfg.batch_size, cfg.out_channels, cfg.image_size, cfg.image_size
    )
    with torch.no_grad():
        predictions = model(grayscale, color)
    assert predictions.ndim == 4 and predictions.shape[:2] == (cfg.batch_size, 1)
    assert predictions.shape[2] > 0 and predictions.shape[3] > 0
    assert torch.isfinite(predictions).all()
    print(f"Disc smoke test : {tuple(predictions.shape)}")


if __name__ == "__main__":
    main()

