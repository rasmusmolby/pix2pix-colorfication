# pix2pix-colorfication
Project with focus on developing an AI model capable of colorizing grayscale images with a Generative Adversarial Network (GAN) approach.

Hyperparameters are in `configs/preprocessing.yaml` and `configs/model.yaml`.
Hydra is pinned to 1.4s release candidate for the projects Python 3.14 environment.
Install the dependencies with `pip install -r requirements.txt`, then run from the project folder:

```bash
python preprocessing.py
python models/generator.py
```

Values can also be changed for one run without editing the files:

```bash
python preprocessing.py batch_size=8 image_dir=data
python models/generator.py batch_size=2
python models/discriminator.py --cfg job
```

The generator uses 256x256 images (larger multiples of 256 also work).
Put images in `data/` before running preprocessing (or set `shuffle=false` for an empty folder).
Preprocessing currently just reports the dataset size; image conversion and the
discriminator forward pass still need finishing. `--cfg job` shows the settings
without running the script. Hydra saves run logs and settings in `outputs/`.
