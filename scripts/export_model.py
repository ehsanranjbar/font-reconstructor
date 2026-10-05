"""
Export the model of a training checkpoint to a single file for identify.py.

The exported file holds the weights, the arguments of the architecture and how the training images were
rendered. It needs neither the training configuration nor the fonts to be used.

Usage: python scripts/export_model.py -r saved/models/<name>/<run>/model_best.pth [-o pretrained/font_encoder.pth]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_reconstructor import factory  # noqa: E402
from font_reconstructor.inference import export_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Export a trained model for identify.py')
    parser.add_argument('-r', '--resume', required=True, help='path of the checkpoint')
    parser.add_argument('-o', '--out', default='pretrained/font_encoder.pth', help='file to write')
    args = parser.parse_args()

    checkpoint = torch.load(args.resume, map_location='cpu')
    fonts = factory.build_fontset(checkpoint['config'])
    notes = {'checkpoint': str(args.resume), 'monitor_best': float(checkpoint.get('monitor_best', float('nan')))}
    exported = export_model(args.resume, args.out, fonts, notes)
    parameters = sum(value.numel() for value in exported['state_dict'].values())
    print(f"Wrote {args.out}: model {exported['model_id']} of epoch {exported['epoch']}, "
          f"{parameters:,} values, {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
