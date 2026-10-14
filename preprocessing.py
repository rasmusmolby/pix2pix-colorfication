from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ColorizationDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_size = image_size
        self.image_paths = sorted(
            path
            for path in Path(image_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            color = image.convert("RGB")

        color = color.resize((self.image_size, self.image_size))
        grayscale = color.convert("L")

        # TODO: convert both images to tensors and normalize to [-1, 1]. Check arcitecture
        # grayscale should be [1, H, W], color should be [3, H, W]. Depends on color library?
        raise NotImplementedError("Image pair conversion still needs finishing")


def get_dataloader(image_dir, batch_size=16, image_size=256, shuffle=True):
    dataset = ColorizationDataset(image_dir, image_size)
    # TODO: split the images into training and validation sets
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@hydra.main(config_path="configs", config_name="preprocessing")
def main(cfg: DictConfig):
    loader = get_dataloader(
        cfg.image_dir, cfg.batch_size, cfg.image_size, cfg.shuffle
    )
    print(f"Found {len(loader.dataset)} images")
    print(f"Image size: {cfg.image_size}, batch size: {cfg.batch_size}")


if __name__ == "__main__":
    main()
