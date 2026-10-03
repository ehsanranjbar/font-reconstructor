"""
Collects what a validation pass reveals beyond its average loss, for the scalars and figures of a run.

`ValidationReport` is fed every validation batch by `evaluate`. It keeps per-sample results (the rank of the
true font, the predicted style, the error of every glyph) and reduces them to a few extra scalars and to the
arrays the figures in `font_reconstructor.logger.figures` are drawn from.
"""
import heapq
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from font_reconstructor.dataset.capture import clean_text_image
from font_reconstructor.logger import figures


class ValidationReport:
    """
    :param fonts: the FontSet of the run. With it the glyphs of each sample's text are known, so the error of
        glyphs that the model saw in the text is separated from the error of glyphs it had to infer.
    :param baseline: tensor (glyphs, height, width), a fingerprint that ignores the input, for example the
        median fingerprint of the training fonts. The reconstruction skill is measured against it.
    :param worst_cases: number of samples with the worst rank to keep for the figure of worst cases
    """

    def __init__(self, fonts=None, baseline: Optional[torch.Tensor] = None, worst_cases: int = 6):
        self.fonts = fonts
        self.baseline = baseline
        self.worst_cases = worst_cases

        self.samples = 0
        self.ranks = []
        self.font_index = []
        self.style_index = []
        self.style_prediction = []
        self.text_length = []
        self.latent_norms = []
        self._latent_sum = None
        self._latent_squares = None
        self._font_latent_sum = None  # sums of the latent vectors of each font, to tell fonts from texts
        self._font_counts = None
        self._glyph_error = None  # sums of the error per glyph: [seen, unseen]
        self._glyph_count = None
        self._model_error = 0.0
        self._baseline_error = 0.0
        self._worst = []  # heap of (rank, -similarity of the true font, dataset index, predicted fonts, similarities)
        self.num_fonts = None

    @torch.no_grad()
    def update(self, batch, target, latent, output, topk_acc=None, style_logits=None):
        """
        :param batch: the sample dict of the batch, with `font_index` and `text`
        :param target: tensor (batch, glyphs, height, width) on the device of the model
        :param latent: tensor (batch, latent_dim)
        :param output: tensor like target
        :param topk_acc: TopKCosimAccuracy with the font centroids, for the retrieval results
        :param style_logits: tensor (batch, styles) of the style head, if the model has one
        """
        n = target.shape[0]
        font_index = batch['font_index']
        self.font_index.append(np.asarray(font_index))
        self.text_length.append(np.array([len(text) for text in batch['text']]))
        if 'style_index' in batch:
            self.style_index.append(np.asarray(batch['style_index']))
        if style_logits is not None:
            self.style_prediction.append(style_logits.argmax(dim=1).cpu().numpy())

        # latent statistics
        latent = latent.float()
        self.latent_norms.append(latent.norm(dim=1).cpu().numpy())
        if self._latent_sum is None:
            self._latent_sum = torch.zeros(latent.shape[1], dtype=torch.float64)
            self._latent_squares = torch.zeros(latent.shape[1], dtype=torch.float64)
        # moved to the cpu first: not every device has 64 bit floats, Apple's MPS backend does not
        latent_cpu = latent.cpu().double()
        self._latent_sum += latent_cpu.sum(dim=0)
        self._latent_squares += (latent_cpu ** 2).sum(dim=0)
        fonts_needed = int(font_index.max()) + 1
        if self._font_latent_sum is None or fonts_needed > self._font_latent_sum.shape[0]:
            grown = torch.zeros(fonts_needed, latent.shape[1], dtype=torch.float64)
            grown_counts = torch.zeros(fonts_needed, dtype=torch.float64)
            if self._font_latent_sum is not None:
                grown[:self._font_latent_sum.shape[0]] = self._font_latent_sum
                grown_counts[:self._font_counts.shape[0]] = self._font_counts
            self._font_latent_sum, self._font_counts = grown, grown_counts
        self._font_latent_sum.index_add_(0, font_index, latent_cpu)
        self._font_counts.index_add_(0, font_index, torch.ones(n, dtype=torch.float64))

        # reconstruction error of every glyph
        glyph_error = (output - target).abs().mean(dim=(2, 3)).cpu()
        self._model_error += glyph_error.mean(dim=1).sum().item()
        if self.baseline is not None:
            baseline = self.baseline.to(target.device).unsqueeze(0)
            self._baseline_error += (baseline - target).abs().mean(dim=(1, 2, 3)).sum().item()
        if self.fonts is not None:
            seen, valid = glyph_masks(self.fonts, batch['text'], batch['font_index'], glyph_error.shape[1])
            if self._glyph_error is None:
                self._glyph_error = torch.zeros(2, glyph_error.shape[1], dtype=torch.float64)
                self._glyph_count = torch.zeros(2, glyph_error.shape[1], dtype=torch.float64)
            for row, mask in enumerate((seen & valid, ~seen & valid)):
                self._glyph_error[row] += (glyph_error * mask).sum(dim=0).double()
                self._glyph_count[row] += mask.sum(dim=0).double()

        # retrieval
        if topk_acc is not None:
            self.num_fonts = topk_acc.num_fonts
            rank, top_fonts, top_similarities = topk_acc.ranks(latent, font_index)
            true_similarity = topk_acc.similarities(latent).gather(1, font_index.to(latent.device).unsqueeze(1))
            rank, top_fonts = rank.cpu().numpy(), top_fonts.cpu().numpy()
            top_similarities = top_similarities.cpu().numpy()
            true_similarity = true_similarity.squeeze(1).cpu().numpy()
            self.ranks.append(rank)
            for i in range(n):
                entry = (int(rank[i]), -float(true_similarity[i]), self.samples + i,
                         top_fonts[i].tolist(), top_similarities[i].tolist())
                if len(self._worst) < self.worst_cases:
                    heapq.heappush(self._worst, entry)
                elif entry[:3] > self._worst[0][:3]:
                    heapq.heapreplace(self._worst, entry)

        self.samples += n

    # results

    def _concatenated(self, parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    @property
    def all_ranks(self):
        return self._concatenated(self.ranks)

    def scalars(self):
        """
        :return: dict of further validation metrics.
            mrr                 mean reciprocal rank of the true font, 1 if it is always the best match
            recon_skill         share of the baseline's reconstruction error that the model removes, 0 is no
                                better than ignoring the input, 1 is a perfect reconstruction
            seen_glyph_loss     reconstruction error of the glyphs that occur in the input text
            unseen_glyph_loss   reconstruction error of the glyphs the model had to infer
        """
        scalars = {}
        ranks = self.all_ranks
        if len(ranks):
            scalars['mrr'] = float(np.mean(1.0 / ranks))
        if self.baseline is not None and self._baseline_error > 0:
            scalars['recon_skill'] = 1.0 - self._model_error / self._baseline_error
        if self._glyph_error is not None:
            totals, counts = self._glyph_error.sum(dim=1), self._glyph_count.sum(dim=1)
            if counts[0] > 0:
                scalars['seen_glyph_loss'] = float(totals[0] / counts[0])
            if counts[1] > 0:
                scalars['unseen_glyph_loss'] = float(totals[1] / counts[1])
        return scalars

    def accuracy_by_rank(self, max_rank: int = 100):
        """
        :return: (ranks 1..max_rank, share of the samples whose true font is within each rank)
        """
        ranks = self.all_ranks
        ks = np.arange(1, min(max_rank, self.num_fonts or max_rank) + 1)
        return ks, np.array([(ranks <= k).mean() for k in ks])

    def top1_by_group(self, groups):
        """
        :param groups: array with a group id per sample
        :return: (group ids, top-1 accuracy of each group, number of samples of each group)
        """
        ranks, groups = self.all_ranks, np.asarray(groups)
        ids = np.unique(groups)
        accuracy = np.array([(ranks[groups == group] == 1).mean() for group in ids])
        counts = np.array([(groups == group).sum() for group in ids])
        return ids, accuracy, counts

    def style_confusion(self, num_styles: int):
        """
        :return: array (styles, styles) of sample counts, rows are the true style, columns the predicted one
        """
        matrix = np.zeros((num_styles, num_styles), dtype=np.int64)
        np.add.at(matrix, (self._concatenated(self.style_index).astype(int),
                           self._concatenated(self.style_prediction).astype(int)), 1)
        return matrix

    def glyph_errors(self):
        """
        :return: (seen, unseen), arrays (glyphs,) of the mean error of each glyph when it occurs in the input
                 text and when it does not. NaN where a glyph never occurred that way.
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            errors = (self._glyph_error / self._glyph_count).numpy()
        return errors[0], errors[1]

    def latent_mean(self):
        """
        :return: array (latent_dim,), the mean of each latent dimension over the samples
        """
        return (self._latent_sum / self.samples).numpy()

    def font_signal(self):
        """
        How much of the variation of each latent dimension lies between fonts.

        :return: array (latent_dim,) in [0, 1]. 1 for a dimension that only depends on the font and is the same
                 for all texts of it, 0 for one that does not tell fonts apart at all.
        """
        mean = self._latent_sum / self.samples
        total = (self._latent_squares / self.samples - mean ** 2).clamp(min=1e-12)
        present = self._font_counts > 0
        font_means = self._font_latent_sum[present] / self._font_counts[present].unsqueeze(1)
        weights = (self._font_counts[present] / self.samples).unsqueeze(1)
        between = (weights * (font_means - mean) ** 2).sum(dim=0)
        return (between / total).clamp(0, 1).numpy()

    def latent_spread(self):
        """
        :return: array (latent_dim,), the standard deviation of each latent dimension over the samples
        """
        mean = self._latent_sum / self.samples
        variance = (self._latent_squares / self.samples - mean ** 2).clamp(min=0)
        return variance.sqrt().numpy()

    def worst(self):
        """
        :return: list of (rank, dataset index, predicted fonts, similarities) of the worst ranked samples,
                 worst first
        """
        return [(rank, index, fonts, similarities)
                for rank, _, index, fonts, similarities in sorted(self._worst, reverse=True)]


def glyph_masks(fonts, texts, font_indices, num_glyphs):
    """
    :return: (seen, valid), bool tensors (samples, glyphs). seen marks the glyphs that occur in the text of a
             sample. valid marks the glyph slots that the font of the sample fills.
    """
    seen = torch.zeros(len(texts), num_glyphs, dtype=torch.bool)
    valid = torch.zeros_like(seen)
    for row, (text, font_index) in enumerate(zip(texts, font_indices)):
        characters = set(text)
        for column, glyph in enumerate(fonts.glyphs(int(font_index))[:num_glyphs]):
            if glyph != "\0":
                valid[row, column] = True
                seen[row, column] = glyph in characters
    return seen, valid


def base_dataset(dataset):
    """
    the RandomTextImageDataset behind a loader's dataset, which may be wrapped in a TransformedSubset
    """
    return getattr(dataset, 'dataset', dataset)


def median_fingerprint(dataset, target_transform) -> Optional[torch.Tensor]:
    """
    The per-pixel median fingerprint of the fonts of a dataset, as a tensor like a transformed target.

    It is the best single answer for someone who never looks at the input, which makes it the reference for
    how much a model gains by looking.
    """
    base = base_dataset(dataset)
    if not hasattr(base, 'font_fingerprint'):
        return None
    fingerprints = np.stack([base.font_fingerprint(font_index) for font_index in base.font_indices])
    return target_transform(np.median(fingerprints, axis=0).astype(np.uint8))


def choose_panel(dataset, size: int = 8, seed: int = 0) -> Sequence[int]:
    """
    Pick the sample indices of a fixed panel: one sample each of `size` fonts that cover the styles.

    The fonts are taken style by style in turn, so that rare styles are shown too. The dataset has to be
    built with group_by_font, where the first samples are one per font.
    """
    base = base_dataset(dataset)
    fonts, font_indices = base.fonts, list(base.font_indices)
    order = np.random.default_rng(seed).permutation(len(font_indices))

    by_style = {}
    for position in order:
        by_style.setdefault(fonts.styles[font_indices[position]], []).append(int(position))

    panel = []
    queues = [by_style[style] for style in sorted(by_style)]
    while len(panel) < min(size, len(font_indices), len(dataset)):
        for queue in queues:
            if queue and len(panel) < size:
                position = queue.pop(0)
                if position < len(dataset):
                    panel.append(position)
        if not any(queues):
            break
    return panel


def clean_input(dataset, index, dims):
    """
    the clean, tightly cropped rendering of sample `index`, as a float array in [-1, 1]
    """
    raw = base_dataset(dataset)[index]['image']
    return clean_text_image(raw, dims).astype(np.float32) / 127.5 - 1.0


class ValidationReporter:
    """
    Builds the validation reports and figures of a run.

    It fixes, once, what every epoch is compared on: the panel of samples that is drawn each time, and the
    baseline fingerprint that the reconstruction is measured against.

    :param valid_dataset: the validation dataset, as given to its loader
    :param train_dataset: the training dataset. The baseline is the median fingerprint of its fonts.
    :param panel_size: number of samples of the fixed panel
    :param worst_cases: number of worst ranked samples shown each epoch
    :param texts_per_font: number of texts of each panel font in the comparison of latent vectors
    """

    def __init__(self, valid_dataset, train_dataset=None, panel_size: int = 8, worst_cases: int = 6,
                 texts_per_font: int = 3):
        self.dataset = valid_dataset
        self.texts_per_font = texts_per_font
        self.base = base_dataset(valid_dataset)
        self.fonts = getattr(self.base, 'fonts', None)
        self.worst_cases = worst_cases

        target_transform = getattr(valid_dataset, 'target_transform', None)
        self.target_transform = target_transform
        self.baseline = None
        if train_dataset is not None and target_transform is not None:
            self.baseline = median_fingerprint(train_dataset, target_transform)

        self.panel_indices = []
        if self.fonts is not None and getattr(self.base, 'group_by_font', False):
            self.panel_indices = list(choose_panel(valid_dataset, panel_size))
        self._panel = None
        self._comparison = None

    def new_report(self) -> ValidationReport:
        return ValidationReport(self.fonts, self.baseline, self.worst_cases)

    def _load_panel(self):
        """
        the samples of the panel. They are read once, because validation samples are the same on every read.
        """
        if self._panel is None:
            samples = [self.dataset[index] for index in self.panel_indices]
            dims = (samples[0]['image'].shape[-1], samples[0]['image'].shape[-2])
            self._panel = {
                'samples': samples,
                'images': torch.stack([sample['image'] for sample in samples]),
                'targets': torch.stack([sample['target'] for sample in samples]),
                'clean': [clean_input(self.dataset, index, dims) for index in self.panel_indices],
            }
        return self._panel

    @torch.no_grad()
    def panel_figure(self, model, device, topk_acc=None):
        panel = self._load_panel()
        samples = panel['samples']
        model.eval()
        latent = model.encode(panel['images'].to(device))
        output = model.decode(latent).cpu()
        targets = panel['targets']
        font_index = torch.tensor([sample['font_index'] for sample in samples])
        seen, _ = glyph_masks(self.fonts, [sample['text'] for sample in samples], font_index, targets.shape[1])

        errors = (output - targets).abs().mean(dim=(1, 2, 3))
        skills = None
        if self.baseline is not None:
            skills = 1.0 - errors / (self.baseline.unsqueeze(0) - targets).abs().mean(dim=(1, 2, 3))

        ranks = matches = None
        if topk_acc is not None:
            ranks, top_fonts, top_similarities = topk_acc.ranks(latent, font_index)
            ranks, top_fonts, top_similarities = ranks.cpu(), top_fonts.cpu(), top_similarities.cpu()
            matches = [[(self.fonts.names[int(font)], float(similarity)) for font, similarity in zip(*row)]
                       for row in zip(top_fonts, top_similarities)]

        styles = None
        if getattr(model, 'style_head', None) is not None:
            styles = F.softmax(model.predict_style(latent), dim=1).cpu()

        rows = []
        for i, sample in enumerate(samples):
            row = {
                'font': sample['font'], 'text': sample['text'],
                'clean': panel['clean'][i], 'input': sample['image'][0].numpy(),
                'target': targets[i].numpy(), 'output': output[i].numpy(), 'seen': seen[i].numpy(),
                'error': float(errors[i]),
                'skill': None if skills is None else float(skills[i]),
                'rank': None if ranks is None else int(ranks[i]),
                'matches': [] if matches is None else matches[i],
            }
            if styles is not None:
                prediction = int(styles[i].argmax())
                row.update(style=self.fonts.styles[sample['font_index']],
                           style_prediction=self.fonts.style_names[prediction],
                           style_confidence=float(styles[i, prediction]))
            rows.append(row)
        return figures.plot_panel(rows, num_fonts=None if topk_acc is None else topk_acc.num_fonts)

    @torch.no_grad()
    def latent_comparison_figure(self, model, device, report: ValidationReport):
        """
        The latent vectors of the panel fonts side by side, a few texts of each.

        The validation set cycles through its fonts, so the samples of a font are one round of fonts apart,
        and the samples of one round share their text. That gives every font the same texts to be compared on.
        """
        fonts_per_round = len(self.base.font_indices)
        if self._comparison is None:
            rows = []
            for group, position in enumerate(self.panel_indices):
                for text_number in range(self.texts_per_font):
                    index = position + text_number * fonts_per_round
                    if index < len(self.dataset):
                        sample = self.dataset[index]
                        rows.append((group, sample['font'], sample['text'], sample['image']))
            self._comparison = rows
        rows = self._comparison

        model.eval()
        latent = model.encode(torch.stack([image for _, _, _, image in rows]).to(device)).float().cpu().numpy()
        return figures.plot_latent_comparison(
            latent, [group for group, _, _, _ in rows], [font for _, font, _, _ in rows],
            [text for _, _, text, _ in rows], report.latent_mean(), report.latent_spread(), report.font_signal())

    def worst_cases_figure(self, report: ValidationReport):
        cases = []
        for rank, index, top_fonts, similarities in report.worst():
            sample = self.dataset[index]
            predicted = self.target_transform(self.base.font_fingerprint(top_fonts[0]))
            cases.append({
                'font': sample['font'], 'text': sample['text'], 'input': sample['image'][0].numpy(),
                'target': sample['target'].numpy(), 'predicted': predicted.numpy(),
                'predicted_font': self.fonts.names[top_fonts[0]], 'rank': rank, 'similarity': similarities[0],
            })
        return figures.plot_worst_cases(cases, num_fonts=report.num_fonts)

    def figures(self, model, device, report: ValidationReport, topk_acc=None) -> dict:
        """
        :return: dict of tensorboard tag to matplotlib figure, of everything this validation pass can show
        """
        result = {}
        if self.panel_indices:
            result['samples'] = self.panel_figure(model, device, topk_acc)

        if report.samples:
            norms = np.concatenate(report.latent_norms)
            result['latent_space'] = figures.plot_latent_health(report.latent_spread(), norms)
            if self.panel_indices:
                result['latent_comparison'] = self.latent_comparison_figure(model, device, report)

        if self.fonts is not None and report._glyph_error is not None:
            glyphs = max((self.fonts.glyphs(index) for index in range(len(self.fonts))), key=len)
            seen, unseen = report.glyph_errors()
            labels = [glyph if glyph != "\0" else '' for glyph in glyphs.ljust(len(seen))]
            result['glyph_error'] = figures.plot_glyph_error(labels, seen, unseen)

        if len(report.all_ranks):
            ranks, accuracy = report.accuracy_by_rank()
            result['identification/by_rank'] = figures.plot_accuracy_by_rank(
                ranks, accuracy, num_fonts=report.num_fonts)
            lengths, accuracy, counts = report.top1_by_group(np.concatenate(report.text_length))
            result['identification/by_text_length'] = figures.plot_bars(
                [f"{length} characters" for length in lengths], accuracy, counts,
                'Top-1 identification by text length', 'Held out fonts. Longer texts show more of a font.')
            if self.fonts is not None and report.style_index:
                styles, accuracy, counts = report.top1_by_group(np.concatenate(report.style_index))
                result['identification/by_style'] = figures.plot_bars(
                    [self.fonts.style_names[int(style)] for style in styles], accuracy, counts,
                    'Top-1 identification by style', 'Held out fonts, grouped by the style of the font.')
            if self.fonts is not None and self.target_transform is not None and report.worst():
                result['worst_cases'] = self.worst_cases_figure(report)

        if self.fonts is not None and report.style_prediction:
            names = self.fonts.style_names
            result['style_predictions'] = figures.plot_confusion(names, report.style_confusion(len(names)))
        return result
