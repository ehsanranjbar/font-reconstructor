# Font Reconstructor

A small convolutional autoencoder that looks at an image of a short text and reconstructs the
*fingerprint* of the font it was written in: one small image per glyph of the font's charset.

The latent vector of the encoder is also a font embedding. It is trained with a contrastive loss, so
that images of the same font lie close together. A font is identified by comparing an image's
embedding with the average embedding of each known font. A new font only needs a few reference images
for that, no retraining. Validation measures exactly this: the validation fonts are never trained on.

Training data is synthetic. Real Persian text is rendered with the fonts you provide, so no labelled
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

### Text shaping

Persian and Arabic letters change shape and join inside a word, and are written right to left. Pillow
only does this with its Raqm layout engine, which needs the FriBiDi library at runtime:

```bash
brew install fribidi            # macOS
sudo apt install libfribidi0    # Debian, Ubuntu

python -c "from PIL import features; print(features.check('raqm'))"    # has to print True
```

On Apple silicon Homebrew installs to `/opt/homebrew/lib`, which is not on the default library search
path. If the check still prints `False`, run with `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`.

`config.json` sets `"layout_engine": "raqm"`, so training stops with an explanation if shaping is not
available. Without shaping every letter would be drawn in its isolated form, in reversed order.

## Data

Put the fonts, an annotation file and the text corpus under `data/` (ignored by git):

```text
data/
  fonts/            TrueType or OpenType font files
  fonts.csv         one row per font
  corpus/           text files to render, see below
  cache/            rendered images and fingerprints, created on first use
```

`fonts.csv` has these columns:

| column | meaning |
|--------|---------|
| `font` | display name of the font, used as its label |
| `file` | path of the font file, relative to `data/fonts` |
| `supported_charset` | the characters of the font. Texts only use these characters, and each one except the space gets a channel of the fingerprint. |
| `family` | optional. Groups the weights and styles of one typeface. Without it the family name stored in the font file is used. |
| `style` | optional. The style of the font, which the style head of the model learns to predict. Any set of labels works, for example calligraphic styles such as naskh or kufi. Without it the style is derived from the names of the font: `outline` or `shadow` for decorated fonts, otherwise `light`, `regular` or `bold`, followed by `italic` for slanted ones. |

Every font should list its characters in the same order. A null character (`\0`) keeps the slot of
a character that a font does not support empty, so that glyph positions stay aligned between fonts.
Fonts that fail to open, or that miss a glyph of their charset, are skipped with a message.

The number of fingerprint channels is the longest charset in the file. The model reads it from the
data, so it does not have to be configured.

### Duplicate fonts

Font collections often contain the same typeface under several names. To the model these are different
classes that look the same, which hurts training and makes top-1 accuracy unreachable for them.

```bash
python scripts/dedupe_fonts.py            # dry run: prints the duplicates and writes a report
python scripts/dedupe_fonts.py --apply    # moves duplicates to data/fonts_duplicates/ and rewrites the csv files
```

The script renders every glyph of every font, compares all pairs of fonts, and calls two fonts
duplicates if they agree on every glyph (`--threshold` for the average difference, `--glyph-threshold`
for the worst glyph). Weights and styles of a typeface are kept. The dry run writes
`data/dedupe_report/` with the list of duplicates and a picture of sample pairs for judging the
thresholds. `--apply` keeps backups of the csv files. Changing the font list invalidates the caches
and the font numbering of earlier runs, so apply it before a training run, not during one.

### Synthetic fonts

Collections have many regular fonts and few outlined or slanted ones. `synthetic_variants` in the `dataset`
block fills those styles up: it takes a number of extra fonts per style, and each is a real font drawn
outlined, slanted or heavier. A variant is a font of its own, with its own text images, fingerprint and
style label.

```json
"synthetic_variants": {"outline": 200, "italic": 120, "bold italic": 100}
```

| style | drawn as | made from |
|-------|----------|-----------|
| `outline` | hollow, with a line around the glyphs | regular, bold and light fonts |
| `italic` | slanted, top leaning left as Persian slanted styles do | regular fonts |
| `bold italic` | slanted | bold fonts |
| `bold` | with a heavier stroke | regular fonts |

Variants are in the family of their font, so the variants of a held out font are not trained on.
Validation uses real fonts only. All fonts, real and synthetic, are among those a font is identified from.

### Text corpus

Texts are drawn from the files listed under `corpus_files`. Each line of a file is a phrase, or a
single word. A text is a run of consecutive words of one line that fits `text_length`. Words with
characters outside a font's charset are skipped. Without `corpus_files`, random characters of the
charset are rendered.

```bash
python scripts/download_corpus.py
```

This downloads the two sources of the default config. The choice follows the Persis font recognition
paper (Mohammadian et al., 2023), which renders the Shahnameh and dictionary words. Persis does not
publish its text files, so public equivalents are used:

| file | content | source |
|------|---------|--------|
| `data/corpus/shahnameh.txt` | the Shahnameh of Ferdowsi, one half-verse per line | [Persian poems corpus](https://github.com/amnghd/Persian_poems_corpus), public domain text |
| `data/corpus/words.txt` | the 50,000 most frequent words of the Persian Wikipedia | [Persian Words Frequency](https://github.com/behnam/persian-words-frequency), CC BY-SA 3.0 |

Any other utf-8 text file works as well. Arabic letter forms and digits are converted to their Persian
forms, and diacritics are removed.

### Images like real ones

A text image that reaches the model in practice was printed or shown on a screen, photographed, and
then cleaned up to a tight crop of white text on black. Training images go through a simulation of
that history (`CaptureSimulation` in `font_reconstructor/dataset/capture.py`):

1. slight rotation and perspective that deskewing did not remove
2. ink spread or loss, lens blur and camera motion
3. a dark-on-light photo with uneven lighting, sensor noise, limited resolution and JPEG compression
4. the cleanup: background removal, contrast stretch, and for half of the images a threshold. The paper
   level is fitted as a plane, so the inside of heavy strokes stays ink.
5. a crop to the text with a random margin and resizing to the model input. The crop never cuts into the
   text by default, because the dots and tails of letters at the edge tell fonts apart.

Every result is compared with the clean text it was made from. If the text was lost, broken up or drowned
in noise, the simulation runs again, and after four such results the clean rendering is used instead. An
image like that tells little about the font, and nobody would use it to identify one. `min_agreement` sets
how much damage is accepted: it is the lowest correlation with the clean text, 0.9 by default.

The simulation also avoids damage that a real capture does not do. The photo is exposed for its brightest
paper, so paper is never blown out and taken for ink, and strokes are only thickened or thinned by a pixel
where that does not wipe out hairlines or close up outlined letters.

Steps 2 and 4 make strokes thicker or thinner, which is what a real threshold does. Texts are rendered
at `render_scale` times the model input, so these changes are finer than one pixel of the input.

| setting | where | meaning |
|---------|-------|---------|
| `render_scale` | `dataset` | size of the cached renderings relative to the model input, 2 by default. `font_size` should be about `render_scale` times the input height. |
| `number_ratio` | `dataset` | share of the texts that get a number of one to four digits added |
| `capture` | `dataset` | arguments of `CaptureSimulation`, for example `{"binarize_prob": 1.0}` if your preprocessing always binarizes |
| `random_augmentations` | `data_loader` | run the training images through the simulation |
| `validation_augmentations` | `data_loader` | run the validation images through it as well. Every image gets the same distortion each time. |
| `random_augmentations` | `clustering_data_loader` | run the reference images of each font through it too |

Set the `capture` options to match your own preprocessing. The closer the simulated cleanup is to the
real one, the better the model carries over to real images. A set of real, labelled crops is the only
way to measure that.

## Training

```bash
python train.py -c config.json
python train.py -c config.json --lr 0.0005              # override single settings
python train.py -r saved/models/<name>/<run>/checkpoint-epoch10.pth    # resume
```

Checkpoints go to `saved/models/<name>/<run>/` and logs to `saved/log/<name>/<run>/`. Follow a
run with `tensorboard --logdir saved/log`.

The first run renders all text images and font fingerprints into `data/cache/`. Later runs with
the same settings map these files into memory, so they start fast and data loader workers share
one copy.

### Learning rate

The `lr_scheduler` block takes any class of `torch.optim.lr_scheduler`. Its `interval` says when the
scheduler is stepped: after every `"epoch"`, or after every `"batch"`. An epoch has thousands of batches,
so schedules with a warmup need `"batch"`.

The default is `OneCycleLR`: the rate rises from `max_lr / div_factor` to `max_lr` during the first
`pct_start` of all steps, then falls along a cosine to almost zero. The length of the schedule is the
`epochs` of the trainer, so `epochs` is a budget here: lower it to finish sooner, and expect the run to
use all of it unless early stopping ends it. The `lr` of the optimizer and the `--lr` option have no
effect with this scheduler, `max_lr` sets the rate. The position in the schedule is stored in
checkpoints, so a resumed run continues where it stopped.

To find `max_lr` for a config, run a range test. It trains for a few hundred steps while the rate rises
and writes a plot of the loss against the rate:

```bash
python scripts/find_lr.py -c config.json
```

Pick the rate where the loss stops falling, a few times below the rate where it starts to rise. Run the
test again after changing the batch size or the loss weights, they move the usable rates.

### What is trained and how it is measured

- **Held out fonts.** `validation_split` holds out whole font families. Validation and `test.py`
  only use those fonts. Each of them is identified among all fonts, the training fonts included.
- **Loss.** The reconstruction loss (`loss`) plus `contrastive_loss.weight` times a supervised
  contrastive loss on the latent vector. The default reconstruction loss is `multiscale_l1_loss`: the
  mean absolute error of the fingerprint and of its averages over blocks of 2, 4 and 8 pixels. With a
  plain pixel loss (`l1_loss`), a thin stroke drawn one pixel off costs twice as much as drawing
  nothing, which taught the model to leave outline fonts blank. `val_recon_skill` is always measured
  with the plain pixel error, so it stays comparable between losses.
- **Glyphs of the text only, optional.** With a `reconstruction` block of `{"glyphs": "text", "glyphs_per_sample": 8}` the
  reconstruction is trained on glyphs that occur in the text of each image instead of the whole fingerprint:
  eight of them per image, picked at random, some twice if the text shows fewer. The model is then not asked to
  guess glyphs it has not seen. It also makes the conditioned decoder affordable, which only draws those
  glyphs. Validation still draws the whole fingerprint, so `val_loss` stays comparable between runs, and
  `val_seen_glyph_loss` and `val_unseen_glyph_loss` show how the glyphs it was not trained to draw come out.
  The training `loss` covers the picked glyphs only. It does not work together with the adversarial term.
  Set `glyphs` to `"all"` to train on the whole fingerprint again. Both settings can also be given on the
  command line, which overrides the config: `python train.py -c config.json --glyphs all`, and
  `--decoder joint` or `--decoder conditioned` for the kind of decoder.
- **Batches.** The contrastive loss needs several images of a font in one batch. Training batches
  hold `batch_fonts` fonts with `batch_samples_per_font` images each.
- **Style head.** With a `style_head` block, a single linear layer on the latent vector predicts the
  style of the font, trained with cross-entropy scaled by `weight`. It tells how much of the style can
  be read from the latent vector directly. With `"detach": true` it only measures that and does not
  shape the encoder. Training logs `style_loss` and `style_acc`. Validation logs `style_acc` and
  `style_balanced_acc`, the accuracy averaged over the styles. Most fonts are regular, so a model that
  always answers "regular" already gets a high `style_acc`, but only chance level on the balanced one.
- **Adversarial term, optional.** With an `adversarial_loss` block, a discriminator judges patches of
  the drawn fingerprints against real ones, and the decoder is also trained to pass as real. This
  follows MC-GAN (Azadi et al., 2018) and sharpens strokes that a pixel loss leaves blurry.
- **Logged values.** `loss` is the reconstruction loss for training and validation. Training also
  logs the other terms and their weighted sum `total_loss`. Validation logs `top{k}_acc` and the
  values below.

### What validation reports

Besides the losses and accuracies, every validation logs these scalars:

| scalar | meaning |
|--------|---------|
| `val_mrr` | mean reciprocal rank of the true font. 1 if it is always the best match. It reacts to every change in rank, not only to the top-k cut-offs. |
| `val_recon_skill` | share of the baseline's reconstruction error that the model removes. The baseline is the median fingerprint of the training fonts, the best answer without looking at the input. 0 is no better than that, 1 is perfect. |
| `val_seen_glyph_loss`, `val_unseen_glyph_loss` | reconstruction error of the glyphs that occur in the input text, and of those the model had to infer |

It also draws the figures below, every `figure_period` epochs. They go to tensorboard, and as image
files to the run's log directory, which is the easier place to look at them:

```text
saved/log/<name>/<run>/figures/
  latest/                 the newest version of every figure, the files to keep open
    samples.png
    worst_cases.png
    ...
  samples/                every epoch of one figure side by side
    epoch_001.png
    epoch_002.png
  worst_cases/
    ...
```

The files are written at `figure_dpi` dots per inch (200 by default), twice the resolution shown in
tensorboard, so they stay sharp when zoomed. Set `save_figures` to false to write none.


| tag | shows |
|-----|-------|
| `samples` | a fixed panel of held out fonts that cover the styles, the same every epoch: the clean rendering, the image given to the model, the rank of the true font with the three best matches, the predicted style, and target, output and error of every glyph. A blue bar marks the glyphs that occur in the input text. |
| `worst_cases` | the samples whose true font ranked lowest, next to the fingerprint of the font they were taken for. If the two look alike, the fonts are near-duplicates. |
| `glyph_error` | reconstruction error of each glyph, in the input text and not |
| `identification/by_rank` | share of samples whose true font is within the k best matches, for every k |
| `identification/by_style`, `identification/by_text_length` | top-1 accuracy by the style of the font and by the length of the text |
| `style_predictions` | which styles the style head answers for each true style |
| `latent_space` | what each latent dimension contributes: its share of all variation, split into the part between fonts, which tells them apart, and the part inside a font, which follows the text. Below it, the top-1 and top-5 accuracy with only the n most useful dimensions, which shows how many dimensions the model needs, and the lengths of the latent vectors. |
| `latent_comparison` | the latent vectors of the panel fonts side by side, three texts of each. Every latent feature is a column, coloured by how far it lies above or below the validation average. A column that keeps its colour within a font and changes between fonts tells fonts apart; one that changes within a font reacts to the text. Below the rows of each font, a dot marks the features whose values lie within half a spread of each other over its texts, which describe the font, and a cross marks those that lie one and a half spreads or more apart, which follow the text. The counts of both stand beside the font. Bars above give each feature's share of variation that lies between fonts, and a matrix beside it gives the cosine similarity of every pair of rows. |

The embedding projector gets the font, its family and its style as metadata, so the points can be
coloured by any of them.

## Testing a checkpoint

```bash
python test.py -r saved/models/<name>/<run>/model_best.pth
```

The config next to the checkpoint is used. The script reports the loss, the configured metrics and
the top-k accuracies on the held out fonts, without augmentations.

## Model

The model is `CompactAutoEncoder`. Its encoder is a stack of residual blocks followed by global
average pooling, so texts of any width map to a latent vector of the same size. Its decoder upsamples
by resizing followed by a convolution, which avoids the checkerboard artifacts of transposed
convolutions.

Only the encoder is needed to identify a font. The decoder is a training aid, so the two are sized
separately:

| argument | part | effect |
|----------|------|--------|
| `base_channels`, `latent_dim` | encoder | size and speed of the model in use. 79 thousand parameters at the defaults of 8 and 32, 312 thousand at the 16 and 32 of `config.json`. |
| `decoder_channels`, `decoder_blocks` | decoder | training cost only. `decoder_blocks` is the number of convolution blocks at each resolution. |
| `decoder_type` | decoder | `"joint"` draws all glyphs at once, one output channel per glyph. `"conditioned"` draws one glyph at a time from the latent vector and a learned character embedding, sharing its weights between all glyphs. It runs once per glyph and trains several times slower. |

| decoder | arguments | decoder parameters | training speed |
|---------|-----------|--------------------|----------------|
| joint, as small as the encoder | none | 42 thousand | fastest |
| joint, wide | `decoder_channels` 32, `decoder_blocks` 2 | 1.0 million | about half as fast |
| conditioned | `decoder_type` "conditioned", `decoder_channels` 16 | 137 thousand | about 15 times slower |
| conditioned, on glyphs of the text | the same with `reconstruction.glyphs` "text" | the same | about twice as slow as the joint decoder at a width of 8 |

In a short comparison of 3,000 training steps each, the wide joint decoder reconstructed held out fonts
best and identified them best. The conditioned decoder and the adversarial term did not pay for their
cost in that budget. They are options for longer experiments, not defaults.

The table is for fingerprints of 32 by 32 pixels. `config.json` draws them at 64 by 64 with a conditioned decoder of
`decoder_channels` 8, which has 79 thousand parameters, trained on eight glyphs of each text.

With a `style_head` block the model also gets a linear layer on the latent vector that predicts the
style of the font, see the training section.

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
| `dataset` | fonts, font size, layout engine, text corpus, text length and image sizes shared by all data loaders |
| `data_loader` | size, seed, held out share of fonts, augmentations and batch composition of the training data |
| `clustering_data_loader` | `samples_per_font` clean images per font, used to estimate the font centroids before each validation. Remove the block to skip top-k accuracy. |
| `optimizer`, `lr_scheduler` | any class of `torch.optim` and `torch.optim.lr_scheduler` |
| `loss`, `metrics` | names of functions in `font_reconstructor/model/loss.py` and `metric.py` |
| `contrastive_loss` | `weight` and `temperature` of the contrastive loss. Remove the block to train on reconstruction only. |
| `style_head` | `weight` of the style loss (0 turns the head off) and `detach` |
| `adversarial_loss` | `weight` of the adversarial term (0 turns it off), `start_epoch`, and `lr` and `discriminator_channels` of the discriminator |
| `trainer` | epochs, checkpointing (`save_period`, `keep_last_checkpoints`), monitoring, early stopping, `topk` values, tensorboard, `figure_period`, `save_figures`, `figure_dpi` |

Random augmentations only apply to the training data.

## Project layout

```text
train.py, test.py           command line entry points
config.json                 default configuration
scripts/download_corpus.py  downloads the text corpus
scripts/dedupe_fonts.py     finds and removes fonts that draw the same glyphs
scripts/find_lr.py          learning rate range test
font_reconstructor/
  config.py                 ConfigParser: config file, command line overrides, run directories
  factory.py                builds fonts, corpus, loaders, model and losses from the config
  evaluation.py             font centroids and the evaluation loop shared by validation and test.py
  reporting.py              per-sample validation results, the extra scalars and the figure panel
  utils.py                  device selection, seeding, metric tracking
  dataset/                  fonts, corpus, rendering, capture simulation, caching, datasets, samplers, loaders
  model/                    CompactAutoEncoder, GlyphDiscriminator, losses, metrics
  trainer/                  BaseTrainer (epochs, checkpoints, monitoring) and Trainer
  logger/                   logging setup, tensorboard writer, the figures written to tensorboard
tests/                      pytest suite, uses the fonts that ship with matplotlib
```

## Development

```bash
pytest
flake8
```

## License

MIT, see [LICENSE](LICENSE).
