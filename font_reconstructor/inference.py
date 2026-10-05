"""
Using a trained model: from an image to the fonts it most likely shows.

A trained model is exported to a single file that holds everything needed to use it without the training
configuration and without the fonts it was trained on: the weights, the arguments of the architecture, and
how its training images were rendered, which new fonts have to be rendered like.

`FontIdentifier` ties the pieces together: it cleans and segments an image (`font_reconstructor.preprocessing`),
encodes the pieces with the model, and looks the result up in a `FontIndex`. It also renders sample images of
a font file and encodes them, which is how fonts get into the index, new ones included: the model does not
have to be trained on a font to tell it from the others.
"""
import csv
import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFont

import font_reconstructor.model.model as module_arch
from font_reconstructor.dataset import FontSet, RandomTextImageDataset, TextCorpus, TransformedSubset, image_transform
from font_reconstructor.dataset.fonts import _LAYOUT_ENGINES, derive_style, family_name, unsupported_glyphs
from font_reconstructor.index import FontIndex
from font_reconstructor.preprocessing import Piece, prepare
from font_reconstructor.utils import prepare_device

EXPORT_FORMAT = 1
# settings of the training data that decide what an image of a font looks like to the model
_DATASET_KEYS = ('font_size', 'layout_engine', 'text_length', 'text_image_dims', 'font_fingerprint_dims',
                 'number_ratio', 'render_scale', 'corpus_files', 'capture')
# a font needs most of the glyphs of the fingerprint, or the texts of the training data can not be set in it
_MIN_GLYPH_SHARE = 2 / 3


def model_id(state_dict) -> str:
    """
    a short identifier of the weights of a model. Latent vectors are only comparable if it is the same.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode('utf-8'))
        digest.update(state_dict[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def export_model(checkpoint_path, out_path, fonts: FontSet, notes: Optional[dict] = None) -> dict:
    """
    Write the model of a training checkpoint to a file that `load_model` reads.

    :param fonts: the font set of the run. It gives the glyphs of the fingerprint and the names of the styles.
    :param notes: anything worth keeping with the model, for example its validation results
    :return: the exported dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    dataset = {key: config['dataset'][key] for key in _DATASET_KEYS if key in config.get('dataset', {})}
    text_width, text_height = dataset['text_image_dims']
    glyph_width, glyph_height = dataset['font_fingerprint_dims']
    has_style_head = 'style_head.weight' in state_dict

    exported = {
        'format': EXPORT_FORMAT,
        'arch': {
            'type': config['arch']['type'],
            'args': {
                **config['arch']['args'],
                'decoder_output_channels': fonts.num_glyphs,
                'input_dims': [text_height, text_width],
                'output_dims': [glyph_height, glyph_width],
                'style_classes': len(fonts.style_names) if has_style_head else 0,
            },
        },
        'state_dict': state_dict,
        'model_id': model_id(state_dict),
        'glyphs': max((fonts.glyphs(index) for index in range(len(fonts))), key=len),
        'style_names': list(fonts.style_names) if has_style_head else [],
        'dataset': dataset,
        'epoch': int(checkpoint['epoch']),
        'notes': dict(notes or {}),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, out_path)
    return exported


def load_model(path, device='cpu'):
    """
    :return: (model in evaluation mode on `device`, the exported dict without its weights)
    """
    exported = torch.load(path, map_location='cpu', weights_only=True)
    if exported.get('format') != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a model exported by export_model, or of another format version.")
    model = getattr(module_arch, exported['arch']['type'])(**exported['arch']['args'])
    model.load_state_dict(exported.pop('state_dict'))
    return model.to(device).eval(), exported


@dataclass
class Identification:
    """
    :param matches: the best matching fonts for all the text of the image together, best first, see
        FontIndex.search
    :param pieces: the pieces of text that were found in the image and encoded
    :param lines: the same for every line of text on its own, as dicts of `line` (its number from the top),
        `pieces` (how many it was cut into) and `matches`. An image can show more than one font.
    :param style: the style the model reads from the image and its probability, if the model predicts styles
    """
    matches: List[dict]
    pieces: List[Piece]
    lines: List[dict] = field(default_factory=list)
    style: Optional[tuple] = None


class FontIdentifier:
    """
    :param model_path: a model exported by `export_model`
    :param index_path: the SQLite file of the font index. It is created if it does not exist.
    :param device: 'auto', 'cuda', 'mps' or 'cpu'
    """

    def __init__(self, model_path, index_path, device: str = 'auto'):
        self.device, _ = prepare_device(1, device)
        self.model, self.info = load_model(model_path, self.device)
        self.dims = tuple(self.info['dataset']['text_image_dims'])
        self.index = FontIndex(index_path, dim=self.model.latent_dim, model_id=self.info['model_id'])

    # images

    @torch.no_grad()
    def encode(self, images: Sequence[np.ndarray], batch_size: int = 256) -> np.ndarray:
        """
        :param images: uint8 arrays (height, width) of model inputs, white text on black
        :return: array (images, latent_dim)
        """
        latents = []
        for start in range(0, len(images), batch_size):
            batch = np.stack(images[start:start + batch_size]).astype(np.float32) / 127.5 - 1.0
            latents.append(self.model.encode(torch.from_numpy(batch).unsqueeze(1).to(self.device)).float().cpu())
        return torch.cat(latents).numpy()

    @torch.no_grad()
    def identify(self, image, k: int = 5) -> Identification:
        """
        The fonts an image of text most likely shows.

        :param image: a path or a PIL image of text on a plain background
        :return: an Identification. Its matches are empty if no text was found or the index has no fonts.
        """
        pieces = prepare(image, self.dims)
        if not pieces:
            return Identification([], [])
        latents = self.encode([piece.image for piece in pieces])
        style = None
        if self.info['style_names']:
            logits = self.model.predict_style(torch.from_numpy(latents).to(self.device))
            probabilities = F.softmax(logits, dim=1).mean(dim=0).cpu()
            best = int(probabilities.argmax())
            style = (self.info['style_names'][best], float(probabilities[best]))
        lines = []
        for number in sorted({piece.line for piece in pieces}):
            members = [position for position, piece in enumerate(pieces) if piece.line == number]
            lines.append({'line': number, 'pieces': len(members), 'matches': self.index.search(latents[members], k)})
        return Identification(self.index.search(latents, k), pieces, lines, style)

    # fonts

    def describe_font(self, font_path) -> dict:
        """
        What a font file is and which glyphs of the fingerprint it has.

        :return: dict with name, family, style and charset. The charset has a null character for every glyph
                 the font lacks, which keeps the glyph positions of the fingerprint.
        """
        dataset = self.info['dataset']
        ttf = ImageFont.truetype(str(font_path), dataset['font_size'], encoding='unic',
                                 layout_engine=_LAYOUT_ENGINES['basic'])
        glyphs = self.info['glyphs']
        missing = set(unsupported_glyphs(ttf, glyphs))
        if len(glyphs) - len(missing) < _MIN_GLYPH_SHARE * len(glyphs):
            raise ValueError(f"{font_path} has only {len(glyphs) - len(missing)} of the {len(glyphs)} characters the "
                             f"model was trained on ({glyphs}), too few to set its texts in this font.")
        family, style_name = ttf.getname()
        name = ' '.join(part for part in (family, style_name) if part) or Path(font_path).stem
        return {
            'name': name,
            'family': family_name(ttf) or family or name,
            'style': derive_style(name, style_name or ''),
            'charset': ''.join("\0" if glyph in missing else glyph for glyph in glyphs),
        }

    def render_font(self, font_path, samples: int = 100, seed: int = 0, augment: bool = True) -> List[np.ndarray]:
        """
        Sample images of a font, made the way the training images are: short Persian texts, put through the
        simulated capture with a camera.

        :param seed: the texts and their distortions follow from it, the same seed gives the same images
        :param augment: False for clean renderings
        :return: uint8 arrays (height, width) of model inputs
        """
        dataset = self.info['dataset']
        description = self.describe_font(font_path)
        font_path = Path(font_path)
        annotation = io.StringIO()
        rows = csv.writer(annotation)
        rows.writerow(['font', 'file', 'supported_charset', 'family'])
        rows.writerow([description['name'], font_path.name, description['charset'], description['family']])
        annotation.seek(0)
        fonts = FontSet(str(font_path.parent), annotation, font_size=dataset['font_size'],
                        layout_engine=dataset.get('layout_engine', 'auto'))
        if len(fonts) != 1:
            raise ValueError(f"{font_path} could not be loaded as a font.")

        corpus_files = [file for file in dataset.get('corpus_files') or [] if os.path.exists(file)]
        corpus = TextCorpus(corpus_files) if corpus_files else None
        texts = RandomTextImageDataset(
            fonts, corpus=corpus, random_seed=seed, total_samples=samples, text_length=dataset['text_length'],
            text_image_dims=self.dims, number_ratio=dataset.get('number_ratio', 0.0),
            render_scale=dataset.get('render_scale', 1), group_by_font=True, return_target=False, cache_images=False)
        transform = image_transform(self.dims, augment=augment, **(dataset.get('capture') or {}))
        subset = TransformedSubset(texts, transform=transform, deterministic=True)
        return [np.round((subset[index]['image'][0].numpy() + 1.0) * 127.5).astype(np.uint8) for index in range(samples)]

    def add_font(self, font_path, name: Optional[str] = None, samples: int = 100, source: str = 'added by user',
                 replace: bool = False, seed: int = 0) -> dict:
        """
        Put a font file into the index: render sample images of it, encode them and store their vectors.

        :param name: name of the font in the index, read from the font file by default
        :return: the description of the font, see `describe_font`
        """
        description = self.describe_font(font_path)
        name = name or description['name']
        if name in self.index and not replace:
            raise ValueError(f"The index already has a font named '{name}'. Replace it or give this one another name.")
        vectors = self.encode(self.render_font(font_path, samples, seed))
        self.index.add(name, vectors, family=description['family'], style=description['style'],
                       file=str(font_path), source=source, replace=replace)
        return {**description, 'name': name, 'samples': len(vectors)}
