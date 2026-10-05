"""
Identify the font of a text image, and manage the index of fonts it is looked up in.

  python identify.py query photo.jpg              the fonts the image most likely shows
  python identify.py add MyFont.ttf               put a font into the index, also one the model never saw
  python identify.py remove "My Font Regular"     take a font out
  python identify.py list                         the fonts of the index
  python identify.py build -c config.json         build the index from the fonts of a training configuration

All commands take --model, the exported model (scripts/export_model.py), and --index, the SQLite file of the
font index. See font_reconstructor/preprocessing.py for what a query image should look like.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from font_reconstructor.inference import FontIdentifier
from font_reconstructor.utils import raise_open_file_limit, read_json

DEFAULT_MODEL = 'pretrained/font_encoder.pth'
DEFAULT_INDEX = 'pretrained/fonts.sqlite'
# Below this similarity the best font is not a match. Text in a font of the index scores 0.85 and more,
# other things that were taken for text, and fonts that are not in the index, mostly less.
WEAK_MATCH = 0.8


def query(identifier, args):
    result = identifier.identify(args.image, k=args.top)
    if args.debug_dir:
        out = Path(args.debug_dir)
        out.mkdir(parents=True, exist_ok=True)
        for number, piece in enumerate(result.pieces):
            Image.fromarray(piece.image).save(out / f'piece_{number:02d}_line{piece.line}.png')
    if args.json:
        print(json.dumps({'pieces': len(result.pieces), 'style': result.style, 'matches': result.matches,
                          'lines': result.lines}, ensure_ascii=False, indent=2))
        return 0 if result.matches else 1

    if not result.pieces:
        print("No text was found in the image. It should show dark text on a light background or the reverse, "
              "in roughly horizontal lines.")
        return 1
    if not result.matches:
        print("The font index is empty. Add fonts with 'identify.py add' or build it with 'identify.py build'.")
        return 1
    lines = len({piece.line for piece in result.pieces})
    print(f"{len(result.pieces)} piece{'s' if len(result.pieces) != 1 else ''} of text in {lines} "
          f"line{'s' if lines != 1 else ''}, compared with {len(identifier.index):,} fonts")
    if result.style is not None:
        print(f"style read from the image: {result.style[0]} ({result.style[1]:.0%})")
    if result.matches[0]['similarity'] < WEAK_MATCH:
        print(f"No font of the index is close to this text (similarity below {WEAK_MATCH}). The font may not be in "
              f"the index, or the image is not Persian text on a plain background.")
    width = max(len(match['name']) for match in result.matches)
    print(f"\nall text together\n{'':>4} {'font':<{width}}  {'similarity':>10}  {'style':<12} family")
    for match in result.matches:
        print(f"{match['rank']:>3}. {match['name']:<{width}}  {match['similarity']:>10.3f}  "
              f"{match['style'] or '':<12} {match['family'] or ''}")

    if lines > 1:
        # an image can show several fonts, which the result for all text together would blur into one
        print("\nline by line, the best matches of each")
        for line in result.lines:
            best = ',  '.join(f"{match['name']} ({match['similarity']:.2f})" for match in line['matches'][:args.per_line])
            weak = '   [no close font]' if line['matches'][0]['similarity'] < WEAK_MATCH else ''
            print(f"{line['line'] + 1:>3}. {best}{weak}")
        if len({line['matches'][0]['name'] for line in result.lines}) > 1:
            print("The lines point to different fonts. If the image shows several fonts, read the lines, not the "
                  "result for all text together.")
    return 0


def add(identifier, args):
    for font in args.fonts:
        if args.name and len(args.fonts) > 1:
            raise SystemExit("--name only works with a single font file.")
        try:
            added = identifier.add_font(font, name=args.name, samples=args.samples, replace=args.replace)
        except ValueError as error:
            print(f"skipped {font}: {error}")
            continue
        print(f"added '{added['name']}' ({added['style']}, family {added['family']}) with {added['samples']} samples")
    print(f"The index now has {len(identifier.index):,} fonts.")
    return 0


def remove(identifier, args):
    status = 0
    for name in args.names:
        if identifier.index.remove(name):
            print(f"removed '{name}'")
        else:
            print(f"There is no font named '{name}' in the index.")
            status = 1
    return status


def list_fonts(identifier, args):
    fonts = identifier.index.fonts()
    if args.filter:
        fonts = [font for font in fonts if args.filter.lower() in font['name'].lower()]
    if args.json:
        print(json.dumps(fonts, ensure_ascii=False, indent=2))
        return 0
    width = max((len(font['name']) for font in fonts), default=4)
    print(f"{'font':<{width}}  {'style':<12} {'samples':>7}  source")
    for font in fonts:
        print(f"{font['name']:<{width}}  {font['style'] or '':<12} {font['samples']:>7}  {font['source'] or ''}")
    print(f"{len(fonts):,} fonts")
    return 0


@torch.no_grad()
def build(identifier, args):
    """
    fill the index with the fonts of a training configuration, from the reference images that validation uses
    """
    from font_reconstructor import factory
    from font_reconstructor.dataset import split_fonts

    raise_open_file_limit()
    config = read_json(args.config)
    if args.samples:
        config['clustering_data_loader']['args']['samples_per_font'] = args.samples
    fonts = factory.build_fontset(config)
    corpus = factory.build_corpus(config)
    loader = factory.build_clustering_loader(config, fonts, identifier.device, corpus)
    _, held_out = split_fonts(fonts, config['data_loader']['args'].get('validation_split', 0.0))
    held_out = set() if held_out is None else set(held_out.tolist())

    vectors = [[] for _ in range(len(fonts))]
    for batch in loader:
        latents = identifier.model.encode(batch['image'].to(identifier.device)).float().cpu().numpy()
        for font_index, latent in zip(batch['font_index'].tolist(), latents):
            vectors[font_index].append(latent)

    added, names = 0, set()
    for font_index, samples in enumerate(vectors):
        if not samples or (fonts.is_synthetic(font_index) and not args.synthetic):
            continue
        # two font files can carry the same name. The file tells them apart then.
        name = fonts.names[font_index]
        if name in names:
            name = f"{name} ({Path(fonts.files[font_index]).stem})"
        names.add(name)
        source = 'held out from training' if font_index in held_out else 'training set'
        identifier.index.add(name, np.stack(samples), family=fonts.families[font_index],
                             style=fonts.styles[font_index], file=fonts.files[font_index], source=source, replace=True)
        added += 1
    print(f"Stored {added:,} fonts. The index now has {len(identifier.index):,} fonts.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Identify the font of a text image')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'exported model (default: {DEFAULT_MODEL})')
    parser.add_argument('--index', default=DEFAULT_INDEX, help=f'font index (default: {DEFAULT_INDEX})')
    parser.add_argument('--device', default='auto', help="'auto', 'cuda', 'mps' or 'cpu' (default: auto)")
    commands = parser.add_subparsers(dest='command', required=True)

    command = commands.add_parser('query', help='find the fonts an image shows')
    command.add_argument('image', help='image of text on a plain background')
    command.add_argument('-k', '--top', default=5, type=int, help='number of fonts to show (default: 5)')
    command.add_argument('--per-line', default=3, type=int, metavar='N',
                         help='number of fonts to show for each line of text (default: 3)')
    command.add_argument('--json', action='store_true', help='print the result as json')
    command.add_argument('--debug-dir', help='write the pieces of text that were found to this directory')
    command.set_defaults(run=query)

    command = commands.add_parser('add', help='put fonts into the index')
    command.add_argument('fonts', nargs='+', help='font files (.ttf, .otf)')
    command.add_argument('--name', help='name of the font in the index, read from the file by default')
    command.add_argument('--samples', default=100, type=int, help='sample images per font (default: 100)')
    command.add_argument('--replace', action='store_true', help='overwrite a font of the same name')
    command.set_defaults(run=add)

    command = commands.add_parser('remove', help='take fonts out of the index')
    command.add_argument('names', nargs='+', help='names of the fonts, as `list` shows them')
    command.set_defaults(run=remove)

    command = commands.add_parser('list', help='show the fonts of the index')
    command.add_argument('--filter', help='only fonts whose name contains this text')
    command.add_argument('--json', action='store_true', help='print the list as json')
    command.set_defaults(run=list_fonts)

    command = commands.add_parser('build', help='build the index from the fonts of a training configuration')
    command.add_argument('-c', '--config', required=True, help='training configuration with the fonts')
    command.add_argument('--samples', type=int, help='sample images per font (default: as in the configuration)')
    command.add_argument('--synthetic', action='store_true', help='also store the synthetic variants of fonts')
    command.set_defaults(run=build)

    args = parser.parse_args(argv)
    if not Path(args.model).exists():
        raise SystemExit(f"There is no model at {args.model}. Export one with scripts/export_model.py.")
    identifier = FontIdentifier(args.model, args.index, device=args.device)
    return args.run(identifier, args)


if __name__ == '__main__':
    sys.exit(main())
