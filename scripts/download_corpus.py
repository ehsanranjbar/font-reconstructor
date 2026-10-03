"""
Download the Persian text the training images are rendered from, into data/corpus/.

The choice of text follows the Persis font recognition paper (Mohammadian et al., 2023), which renders lines
of the Shahnameh and words of a Persian dictionary. Persis does not publish its text files, so equivalent
public sources are used:

  shahnameh.txt  The Shahnameh of Ferdowsi, one half-verse per line. Public domain text, taken from
                 https://github.com/amnghd/Persian_poems_corpus (collected from ganjoor.net).
  words.txt      The most frequent words of the Persian Wikipedia, one per line, taken from
                 https://github.com/behnam/persian-words-frequency by Behnam Esfahbod, published under the
                 Creative Commons Attribution-ShareAlike 3.0 license.

Usage: python scripts/download_corpus.py [--out data/corpus] [--words 50000]
"""
import argparse
import urllib.request
from pathlib import Path

SHAHNAMEH_URL = 'https://raw.githubusercontent.com/amnghd/Persian_poems_corpus/master/original/ferdousi.txt'
WORDS_URL = 'https://raw.githubusercontent.com/behnam/persian-words-frequency/master/persian-wikipedia.txt'


def download(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')


def shahnameh_lines(text):
    """
    the verses, without the two header lines of the source file (file name and number of verses)
    """
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines[2:] if line]


def frequent_words(text, limit):
    """
    the first `limit` words of a frequency list with one 'word<whitespace>count' entry per line, most
    frequent first. Lines starting with '#' are comments.
    """
    words = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or line.startswith('#'):
            continue
        words.append(fields[0])
        if len(words) >= limit:
            break
    return words


def main():
    parser = argparse.ArgumentParser(description='Download the Persian text corpus')
    parser.add_argument('--out', default='data/corpus', help='output directory (default: data/corpus)')
    parser.add_argument('--words', default=50_000, type=int, help='number of frequent words to keep (default: 50000)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    verses = shahnameh_lines(download(SHAHNAMEH_URL))
    (out_dir / 'shahnameh.txt').write_text('\n'.join(verses) + '\n', encoding='utf-8')
    print(f"Wrote {len(verses)} lines to {out_dir / 'shahnameh.txt'}")

    words = frequent_words(download(WORDS_URL), args.words)
    (out_dir / 'words.txt').write_text('\n'.join(words) + '\n', encoding='utf-8')
    print(f"Wrote {len(words)} words to {out_dir / 'words.txt'}")


if __name__ == '__main__':
    main()
