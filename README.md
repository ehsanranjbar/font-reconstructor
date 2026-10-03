# Font Reconstructor

A convolutional autoencoder that looks at an image of a short text and reconstructs the
*fingerprint* of the font it was written in: one small image per glyph of the font's charset.
The latent vector of the encoder doubles as a font embedding, which is scored by how often the
true font is among the k nearest font centroids (top-k accuracy by cosine similarity).

Training data is synthetic. Random texts are rendered with the fonts you provide, so no labelled
images are needed.

## Setup

The project uses a plain virtual environment. Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

`pip` installs the PyTorch build that matches the machine: the MPS-enabled build on Apple silicon
and the default CUDA build on Linux. For a specific CUDA version, install `torch` and `torchvision`
from the matching index first, see <https://pytorch.org/get-started/locally/>.

## Data

Put the fonts and an annotation file under `data/` (ignored by git):

```
data/
  fonts/            TrueType or OpenType font files
  fonts.csv         one row per font
  cache/            rendered images and fingerprints, created on first use
```

`fonts.csv` needs three columns:

| column | meaning |
|--------|---------|
| `font` | display name of the font, used as its label |
| `file` | path of the font file, relative to `data/fonts` |
| `supported_charset` | the characters of the font. Random texts are drawn from them, and each one except the space gets a channel of the fingerprint. |

Every font should list its characters in the same order. A null character (`\0`) keeps the slot of
a character that a font does not support empty, so that glyph positions stay aligned between fonts.
Fonts that fail to open, or that miss a glyph of their charset, are skipped with a message.

The number of fingerprint channels is the longest charset in the file. The model reads it from the
data, so it does not have to be configured.

## Training

```bash
python train.py -c config.json
python train.py -c config.json --lr 0.0005 --bs 256    # override single settings
python train.py -r saved/models/<name>/<run>/checkpoint-epoch10.pth    # resume
```

Checkpoints go to `saved/models/<name>/<run>/` and logs to `saved/log/<name>/<run>/`. Follow a
run with `tensorboard --logdir saved/log`.

The first run renders all text images and font fingerprints into `data/cache/`. Later runs with
the same settings map these files into memory, so they start fast and data loader workers share
one copy.

## Testing a checkpoint

```bash
python test.py -r saved/models/<name>/<run>/model_best.pth
```

The config next to the checkpoint is used. The script reports the loss, the configured metrics and
the top-k accuracies on the validation split, without augmentations.

## Devices

`n_gpu` and `device` in the config select where the model runs.

| `n_gpu` | `device` | result |
|---------|----------|--------|
| `0` | any | CPU |
| `1` or more | `"auto"` (default) | CUDA if available, else Apple's MPS backend, else CPU |
| `1` or more | `"cuda"`, `"mps"` or `"cpu"` | that backend, CPU if it is not available |

More than one GPU is only used with CUDA (`DataParallel`). The `-d` option limits the visible CUDA
devices. If an operation is not implemented for MPS in your PyTorch version, run with
`PYTORCH_ENABLE_MPS_FALLBACK=1` to let it fall back to the CPU.

## Configuration

`config.json` is organised in these blocks:

| block | content |
|-------|---------|
| `arch` | model class and its arguments. Input size, output size and channels follow the dataset. |
| `dataset` | fonts, font size, text length and image sizes shared by all data loaders |
| `data_loader` | size, seed, validation split, augmentations and batching of the training data |
| `clustering_data_loader` | `samples_per_font` clean images per font, used to estimate the font centroids before each validation. Remove the block to skip top-k accuracy. |
| `optimizer`, `lr_scheduler` | any class of `torch.optim` and `torch.optim.lr_scheduler` |
| `loss`, `metrics` | names of functions in `font_reconstructor/model/loss.py` and `metric.py` |
| `trainer` | epochs, checkpointing (`save_period`, `keep_last_checkpoints`), monitoring, early stopping, `topk` values, tensorboard |

Random augmentations only apply to the training split.

## Project layout

```
train.py, test.py           command line entry points
config.json                 default configuration
font_reconstructor/
  config.py                 ConfigParser: config file, command line overrides, run directories
  factory.py                builds fonts, loaders and the model from the config
  evaluation.py             font centroids and the evaluation loop shared by validation and test.py
  utils.py                  device selection, seeding, metric tracking
  dataset/                  fonts, rendering, caching, datasets, transforms, loaders
  model/                    AutoEncoder, losses, metrics
  trainer/                  BaseTrainer (epochs, checkpoints, monitoring) and Trainer
  logger/                   logging setup, tensorboard writer, figures
tests/                      pytest suite, uses the fonts that ship with matplotlib
```

## Development

```bash
pytest
flake8
```

## License

MIT, see [LICENSE](LICENSE).
