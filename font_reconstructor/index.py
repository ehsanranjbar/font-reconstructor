"""
A small vector database of fonts: for every font the latent vectors of some sample images of it.

A font is identified by comparing the latent vector of a query image with the centre of each font, the mean
of its sample vectors, by cosine similarity. That is the measure the validation of a training run reports.
The samples themselves are kept, not only their mean, so that a font can be given more samples later and the
centres can be computed again.

Everything is one SQLite file. With a few thousand fonts an exact search is a single matrix product that
takes less than a millisecond, so there is no approximate index to build or to keep in step.
"""
import datetime
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fonts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    family TEXT,
    style TEXT,
    file TEXT,
    source TEXT,
    added TEXT NOT NULL,
    samples INTEGER NOT NULL,
    vectors BLOB NOT NULL
);
"""
# sample vectors are stored as 16 bit floats. Rounding them to 8 bits already changes no identification.
_STORED = np.float16


class FontIndex:
    """
    :param path: the SQLite file. It is created if it does not exist.
    :param dim: size of the latent vectors. Needed for a new file, checked against an existing one.
    :param model_id: identifier of the model the vectors come from, see `font_reconstructor.inference`.
        Vectors of different models can not be compared, so an existing file of another model is refused.
    """

    def __init__(self, path, dim: Optional[int] = None, model_id: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.executescript(_SCHEMA)
        self._cache = None

        stored_dim, stored_model = self.meta('dim'), self.meta('model_id')
        if stored_dim is None:
            if dim is None:
                raise ValueError(f"{self.path} is a new font index, it needs the size of its vectors.")
            self._set_meta('dim', str(int(dim)))
        elif dim is not None and int(stored_dim) != int(dim):
            raise ValueError(f"{self.path} holds vectors of size {stored_dim}, not {dim}.")
        if stored_model is None:
            if model_id is not None:
                self._set_meta('model_id', model_id)
        elif model_id is not None and stored_model != model_id:
            raise ValueError(f"{self.path} was built with another model ({stored_model}, this one is {model_id}). "
                             f"Build a new index for this model.")
        self.dim = int(self.meta('dim'))

    def meta(self, key: str) -> Optional[str]:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def _set_meta(self, key: str, value: str):
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def close(self):
        self._db.close()

    def __len__(self):
        return self._db.execute("SELECT COUNT(*) FROM fonts").fetchone()[0]

    def __contains__(self, name: str):
        return self._db.execute("SELECT 1 FROM fonts WHERE name = ?", (name,)).fetchone() is not None

    # changing the index

    def add(self, name: str, vectors: np.ndarray, family: Optional[str] = None, style: Optional[str] = None,
            file: Optional[str] = None, source: Optional[str] = None, replace: bool = False, extend: bool = False):
        """
        Store a font with the latent vectors of its sample images.

        :param vectors: array (samples, dim), as the encoder gives them, not scaled to unit length
        :param source: where the font comes from, for example 'training set' or 'added by user'
        :param replace: overwrite a font of this name
        :param extend: add the vectors to those a font of this name already has
        """
        vectors = self._check(vectors)
        if name in self:
            if extend:
                vectors = np.concatenate([self.vectors(name), vectors])
            elif not replace:
                raise ValueError(f"The index already has a font named '{name}'. Replace it, extend it or pick another name.")
        stored = np.ascontiguousarray(vectors, dtype=_STORED)
        added = datetime.datetime.now().isoformat(timespec='seconds')
        with self._db:
            if name in self:
                self._db.execute(
                    "UPDATE fonts SET family = COALESCE(?, family), style = COALESCE(?, style), file = COALESCE(?, file), "
                    "source = COALESCE(?, source), added = ?, samples = ?, vectors = ? WHERE name = ?",
                    (family, style, file, source, added, len(stored), stored.tobytes(), name))
            else:
                self._db.execute(
                    "INSERT INTO fonts (name, family, style, file, source, added, samples, vectors) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (name, family, style, file, source, added, len(stored), stored.tobytes()))
        self._cache = None

    def remove(self, name: str) -> bool:
        """
        :return: whether there was a font of this name
        """
        with self._db:
            removed = self._db.execute("DELETE FROM fonts WHERE name = ?", (name,)).rowcount > 0
        self._cache = None
        return removed

    def _check(self, vectors) -> np.ndarray:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) == 0:
            raise ValueError(f"Expected vectors of shape (samples, {self.dim}), got {vectors.shape}.")
        if not np.isfinite(vectors).all():
            raise ValueError("The vectors contain values that are not finite.")
        return vectors

    # reading the index

    def fonts(self) -> List[dict]:
        """
        :return: the fonts in the order they were added, as dicts of name, family, style, file, source, added
                 and samples
        """
        columns = ('name', 'family', 'style', 'file', 'source', 'added', 'samples')
        rows = self._db.execute(f"SELECT {', '.join(columns)} FROM fonts ORDER BY id").fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def vectors(self, name: str) -> np.ndarray:
        """
        :return: array (samples, dim) of the sample vectors of a font
        """
        row = self._db.execute("SELECT vectors FROM fonts WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        return np.frombuffer(row[0], dtype=_STORED).reshape(-1, self.dim).astype(np.float32)

    def centroids(self):
        """
        :return: (fonts, centres). fonts is the list of `fonts()`, centres an array (fonts, dim) of the mean
                 sample vector of each font, scaled to unit length.
        """
        if self._cache is None:
            fonts = self.fonts()
            rows = self._db.execute("SELECT vectors FROM fonts ORDER BY id").fetchall()
            centres = np.zeros((len(rows), self.dim), dtype=np.float32)
            for position, (blob,) in enumerate(rows):
                centres[position] = np.frombuffer(blob, dtype=_STORED).reshape(-1, self.dim).astype(np.float32).mean(axis=0)
            centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
            self._cache = (fonts, centres)
        return self._cache

    def search(self, vectors: Sequence, k: int = 5) -> List[dict]:
        """
        The fonts whose centre is most similar to the query.

        :param vectors: one latent vector (dim,), or several (n, dim) of images that show the same font, for
            example the pieces of one photo. Several are scaled to unit length and averaged, which identifies
            a font much better than any one of them.
        :return: the `k` best fonts, best first, as the dicts of `fonts()` with `similarity`, the cosine
                 similarity to the query, and `rank`
        """
        fonts, centres = self.centroids()
        if not fonts:
            return []
        vectors = self._check(vectors)
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        query = unit.mean(axis=0)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        similarity = centres @ query
        best = np.argsort(-similarity)[:k]
        return [{**fonts[position], 'similarity': float(similarity[position]), 'rank': rank}
                for rank, position in enumerate(best, start=1)]
