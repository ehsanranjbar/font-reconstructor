import hashlib
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# characters that are written differently in Arabic and Persian text, mapped to their Persian form
_CHAR_MAP = {
    'ي': 'ی',  # arabic yeh -> persian yeh
    'ى': 'ی',  # alef maksura -> persian yeh
    'ك': 'ک',  # arabic kaf -> persian kaf
    '‌': ' ',       # zero width non-joiner separates the parts of a word
    'ـ': '',        # tatweel
}
# arabic-indic and ascii digits are written with persian digits
_CHAR_MAP.update({chr(0x0660 + i): chr(0x06f0 + i) for i in range(10)})
_CHAR_MAP.update({chr(ord('0') + i): chr(0x06f0 + i) for i in range(10)})
_TRANSLATION = str.maketrans(_CHAR_MAP)


def normalize_text(text: str) -> str:
    """
    Reduce a line of text to words of letters and digits separated by single spaces.

    Persian forms replace their Arabic look-alikes, diacritics are removed and everything that is not a
    letter or a digit becomes a space.
    """
    text = text.translate(_TRANSLATION)
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in ('L', 'N'):
            chars.append(char)
        elif category[0] != 'M':  # marks (diacritics) are dropped, the rest separates words
            chars.append(' ')
    return ' '.join(''.join(chars).split())


class TextCorpus:
    """
    Real-world text to render, read from plain utf-8 text files.

    Each line of a file is a phrase, or a single word for word lists. Texts are sampled as runs of consecutive
    words of one line, so they keep the letter combinations and word lengths of the language.

    Only the paths and a content hash are pickled. Data loader workers read the files again on first use.
    """

    def __init__(self, files: Sequence[str]):
        if isinstance(files, (str, Path)):
            files = [files]
        self.files = [str(file) for file in files]
        if not self.files:
            raise ValueError("A text corpus needs at least one file.")

        digest = hashlib.md5()
        for file in self.files:
            digest.update(Path(file).read_bytes())
        self._content_hash = digest.hexdigest()

        self._lines = None
        self._lines_by_charset = {}

    def signature(self):
        """
        everything that determines the sampled texts, used to key caches
        """
        return ['corpus', self._content_hash]

    def _load(self) -> List[List[str]]:
        if self._lines is None:
            lines = []
            for file in self.files:
                with open(file, 'rt', encoding='utf-8') as handle:
                    for line in handle:
                        words = normalize_text(line).split()
                        if words:
                            lines.append(words)
            self._lines = lines
        return self._lines

    def lines_for(self, charset: str) -> List[List[str]]:
        """
        the lines of the corpus restricted to the words that can be written with the characters of `charset`
        """
        if charset not in self._lines_by_charset:
            allowed = set(charset)
            lines = []
            for words in self._load():
                words = [word for word in words if allowed.issuperset(word)]
                if words:
                    lines.append(words)
            self._lines_by_charset[charset] = lines
        return self._lines_by_charset[charset]

    def sample(self, rand, length_range: Tuple[int, int], charset: str, tries: int = 20) -> Optional[str]:
        """
        Draw a text of consecutive words with a length in [length_range[0], length_range[1]).

        :param rand: numpy RandomState, the only source of randomness
        :param charset: characters the text may consist of, besides spaces between words
        :return: the text, or None if the corpus has no fitting text for this charset
        """
        lines = self.lines_for(charset)
        if not lines:
            return None

        min_length, max_length = length_range
        for _ in range(tries):
            words = lines[rand.randint(0, len(lines))]
            start = rand.randint(0, len(words))
            target = rand.randint(min_length, max_length)

            text = words[start]
            if len(text) >= max_length:
                continue
            for word in words[start + 1:]:
                candidate = f"{text} {word}"
                if len(candidate) >= max_length:
                    break
                # grow to the target length, and past it only while the text is still too short
                if len(candidate) > target and len(text) >= min_length:
                    break
                text = candidate

            if len(text) >= min_length:
                return text

        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lines'] = None
        state['_lines_by_charset'] = {}
        return state
