import matplotlib
import numpy as np
import pytest
import torch

from font_reconstructor.evaluation import compute_font_centroids, evaluate
from font_reconstructor.model.loss import l1_loss
from font_reconstructor.logger import figures
from font_reconstructor.model.metric import style_accuracy
from font_reconstructor.reporting import ValidationReport
from font_reconstructor.model.loss import adversarial_loss, discriminator_loss, supervised_contrastive_loss
from font_reconstructor.model.metric import TopKCosimAccuracy
from font_reconstructor.utils import MetricTracker, prepare_device


class IdentityEncoder(torch.nn.Module):
    def encode(self, x):
        return x


def test_topk_accuracy():
    centroids = torch.eye(4)
    topk_acc = TopKCosimAccuracy(centroids)

    latent = torch.tensor([
        [1.0, 0.5, 0.2, 0.0],  # ranks fonts 0, 1, 2, 3
        [0.0, 0.2, 0.5, 1.0],  # ranks fonts 3, 2, 1, 0
    ])
    font_index = torch.tensor([1, 0])

    top1, top2, top4, top9 = topk_acc(latent, font_index, ks=(1, 2, 4, 9))
    assert top1 == 0.0
    assert top2 == 0.5
    assert top4 == 1.0
    assert top9 == 1.0


def test_topk_accuracy_ignores_fonts_without_centroid():
    valid = torch.tensor([True, False, True])
    topk_acc = TopKCosimAccuracy(torch.eye(3), valid)

    latent = torch.tensor([[0.0, 1.0, 0.0]])
    assert topk_acc(latent, torch.tensor([1]), ks=(3,)) == [0.0]
    assert topk_acc(latent, torch.tensor([0]), ks=(3,)) == [1.0]


def test_supervised_contrastive_loss_prefers_clusters_of_one_label():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = torch.tensor([[1.0, 0, 0], [1, 0.01, 0], [0, 1, 0], [0.01, 1, 0], [0, 0, 1], [0, 0.01, 1]])
    mismatched = clustered[[0, 2, 4, 1, 3, 5]]

    assert supervised_contrastive_loss(clustered, labels) < 0.01
    assert supervised_contrastive_loss(mismatched, labels) > 5
    # only the direction of the embeddings counts
    assert torch.isclose(supervised_contrastive_loss(clustered * 7, labels),
                         supervised_contrastive_loss(clustered, labels))


def test_supervised_contrastive_loss_matches_its_definition():
    torch.manual_seed(0)
    labels = [0, 0, 1, 1, 1, 2, 3]
    embeddings = torch.randn(len(labels), 5, requires_grad=True)
    loss = supervised_contrastive_loss(embeddings, torch.tensor(labels), temperature=0.2)

    z = torch.nn.functional.normalize(embeddings.detach(), dim=1)
    similarity = z @ z.T / 0.2
    per_sample = []
    for i, label in enumerate(labels):
        positives = [j for j, other in enumerate(labels) if other == label and j != i]
        if not positives:
            continue  # samples without a partner in the batch do not contribute
        others = [j for j in range(len(labels)) if j != i]
        log_denominator = torch.logsumexp(similarity[i, others], dim=0)
        per_sample.append(-sum(similarity[i, j] - log_denominator for j in positives) / len(positives))
    assert torch.isclose(loss, torch.stack(per_sample).mean(), atol=1e-6)

    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_supervised_contrastive_loss_without_partners_is_zero():
    embeddings = torch.randn(3, 4, requires_grad=True)
    loss = supervised_contrastive_loss(embeddings, torch.tensor([0, 1, 2]))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_font_centroids():
    loader = [
        {'image': torch.tensor([[1.0, 0.0], [3.0, 0.0]]), 'font_index': torch.tensor([0, 0])},
        {'image': torch.tensor([[0.0, 4.0], [float('nan'), 1.0]]), 'font_index': torch.tensor([2, 2])},
    ]
    centroids, valid = compute_font_centroids(IdentityEncoder(), loader, torch.device('cpu'), num_fonts=3)

    assert valid.tolist() == [True, False, True]
    assert centroids[0].tolist() == [2.0, 0.0]
    assert centroids[2].tolist() == [0.0, 4.0]


def test_metric_tracker_weights_by_count():
    tracker = MetricTracker('loss')
    tracker.update('loss', 1.0, n=3)
    tracker.update('loss', torch.tensor(5.0), n=1)
    assert tracker.avg('loss') == 2.0
    assert tracker.result() == {'loss': 2.0}

    tracker.reset()
    assert tracker.avg('loss') == 0.0


def test_prepare_device():
    assert prepare_device(0) == (torch.device('cpu'), [])
    assert prepare_device(1, 'cpu') == (torch.device('cpu'), [])

    device, device_ids = prepare_device(1)
    if torch.cuda.is_available():
        assert device.type == 'cuda' and device_ids == [0]
    elif torch.backends.mps.is_available():
        assert device.type == 'mps' and device_ids == []
    else:
        assert device.type == 'cpu' and device_ids == []

    with pytest.raises(ValueError):
        prepare_device(1, 'gpu')


def test_adversarial_losses():
    ones, zeros = torch.ones(2, 1, 3, 3), torch.zeros(2, 1, 3, 3)
    # the discriminator wants real fingerprints at 1 and drawn ones at 0
    assert discriminator_loss(ones, zeros).item() == 0.0
    assert discriminator_loss(zeros, ones).item() == 1.0
    # the decoder wants its fingerprints to score 1
    assert adversarial_loss(ones).item() == 0.0
    assert adversarial_loss(zeros).item() == 1.0


class StyleModel(torch.nn.Module):
    """
    a stand-in model whose latent is its input and whose style scores are the latent itself
    """

    def __init__(self):
        super().__init__()
        self.style_head = torch.nn.Linear(3, 3)

    def encode(self, x):
        return x

    def decode(self, latent):
        return torch.zeros(latent.shape[0], 1, 2, 2)

    def predict_style(self, latent):
        return latent


def test_style_accuracy():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 3.0, 1.0], [1.0, 0.0, 0.0]])
    assert style_accuracy(logits, torch.tensor([0, 2, 1, 0])) == 1.0
    assert style_accuracy(logits, torch.tensor([0, 2, 0, 1])) == 0.5


def test_evaluate_reports_style_accuracy():
    def batch(images, styles):
        images = torch.tensor(images)
        return {'image': images, 'target': torch.zeros(len(images), 1, 2, 2), 'style_index': torch.tensor(styles),
                'font_index': torch.zeros(len(images), dtype=torch.long)}

    # style 0: 3 of 4 right, style 1: 0 of 1 right, style 2 does not occur
    loader = [
        batch([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]], [0, 0, 0]),
        batch([[1.0, 0, 0], [1.0, 0, 0]], [0, 1]),
    ]
    log = evaluate(StyleModel(), loader, l1_loss, [], torch.device('cpu'))
    assert log['style_acc'] == pytest.approx(3 / 5)
    # the average over the styles that occur, which does not reward answering the common style
    assert log['style_balanced_acc'] == pytest.approx((3 / 4 + 0) / 2)

    # a model without a style head gets no style metrics
    plain = StyleModel()
    plain.style_head = None
    assert set(evaluate(plain, loader, l1_loss, [], torch.device('cpu'))) == {'loss'}


def test_ranks():
    topk_acc = TopKCosimAccuracy(torch.eye(4))
    latent = torch.tensor([
        [1.0, 0.5, 0.2, 0.0],  # ranks fonts 0, 1, 2, 3
        [0.0, 0.2, 0.5, 1.0],  # ranks fonts 3, 2, 1, 0
    ])
    rank, top_fonts, top_similarities = topk_acc.ranks(latent, torch.tensor([1, 0]), top=2)
    assert rank.tolist() == [2, 4]
    assert top_fonts.tolist() == [[0, 1], [3, 2]]
    assert top_similarities[0, 0] > top_similarities[0, 1]

    # a font without a centroid can not be found, it gets the last rank
    masked = TopKCosimAccuracy(torch.eye(4), torch.tensor([True, False, True, True]))
    assert masked.ranks(latent, torch.tensor([1, 0]))[0].tolist() == [4, 3]


class GlyphFonts:
    """
    stand-in for a FontSet with two fonts of three glyphs, the second font leaves its last slot empty
    """

    def glyphs(self, index):
        return ['abc', 'ab\0'][index]


def report_batch(texts, fonts, outputs, targets, styles):
    return ({'text': texts, 'font_index': torch.tensor(fonts), 'style_index': torch.tensor(styles)},
            torch.tensor(targets), torch.tensor(outputs))


def test_validation_report():
    # one pixel per glyph, so the error of a glyph is the difference of two numbers
    report = ValidationReport(GlyphFonts(), baseline=torch.zeros(3, 1, 1), worst_cases=2)
    topk_acc = TopKCosimAccuracy(torch.eye(2))

    def images(values):
        return [[[[value]] for value in row] for row in values]

    batch, target, output = report_batch(
        ['ab', 'c'], [0, 1], images([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), images([[1.0, 0.0, 0.0], [1.0, 1.0, 0.5]]),
        styles=[0, 1])
    latent = torch.tensor([[1.0, 0.1], [1.0, 0.2]])  # both look like font 0
    logits = torch.tensor([[2.0, 0.0], [1.0, 0.0]])  # both answer style 0
    report.update(batch, target, latent, output, topk_acc=topk_acc, style_logits=logits)

    assert report.samples == 2
    assert report.all_ranks.tolist() == [1, 2]
    scalars = report.scalars()
    assert scalars['mrr'] == pytest.approx((1 + 0.5) / 2)

    # sample 0 shows a and b, its errors are 0, 1, 0. sample 1 shows nothing of its font, its errors are 1, 0
    # and its third slot is empty and does not count.
    assert scalars['seen_glyph_loss'] == pytest.approx((0 + 1) / 2)
    assert scalars['unseen_glyph_loss'] == pytest.approx((0 + 1 + 0) / 3)
    seen, unseen = report.glyph_errors()
    assert seen[0] == 0 and seen[1] == 1 and np.isnan(seen[2])
    assert unseen.tolist() == [1, 0, 0]

    # the baseline draws nothing: its error is the mean of the targets, the model's is the mean difference
    model_error = (1 / 3 + 1.5 / 3) / 2
    baseline_error = (1 / 3 + 2.5 / 3) / 2
    assert scalars['recon_skill'] == pytest.approx(1 - model_error / baseline_error)

    assert report.style_confusion(2).tolist() == [[1, 0], [1, 0]]
    assert report.latent_mean().tolist() == pytest.approx([1.0, 0.15])
    # each font has one sample here, so all variation of the second feature lies between the fonts.
    # the first feature does not vary at all.
    assert report.font_signal().tolist() == pytest.approx([0.0, 1.0])
    assert report.latent_spread().tolist() == pytest.approx([0.0, 0.05])
    ranks, accuracy = report.accuracy_by_rank()
    assert ranks.tolist() == [1, 2] and accuracy.tolist() == [0.5, 1.0]
    groups, accuracy, counts = report.top1_by_group(np.array([7, 9]))
    assert groups.tolist() == [7, 9] and accuracy.tolist() == [1.0, 0.0] and counts.tolist() == [1, 1]

    # the worst ranked sample comes first, with the font it was taken for
    worst = report.worst()
    assert [(rank, index) for rank, index, _, _ in worst] == [(2, 1), (1, 0)]
    assert worst[0][2][0] == 0


def test_validation_report_keeps_only_the_worst_cases():
    report = ValidationReport(worst_cases=2)
    topk_acc = TopKCosimAccuracy(torch.eye(4))
    latent = torch.eye(4)
    for fonts in ([0, 1, 2, 3], [1, 0, 3, 2]):  # the first batch is all right, the second all wrong
        batch = {'text': ['ab'] * 4, 'font_index': torch.tensor(fonts)}
        report.update(batch, torch.zeros(4, 1, 2, 2), latent, torch.zeros(4, 1, 2, 2), topk_acc=topk_acc)
    assert sorted(index for _, index, _, _ in report.worst()) != [0, 1]
    assert all(rank > 1 and index >= 4 for rank, index, _, _ in report.worst())


def draw(figure):
    """
    render a figure the way tensorboard does and return its size in pixels
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return canvas.get_width_height()


def test_glyph_sheet():
    target, output = -np.ones((5, 8, 8), dtype=np.float32), np.ones((5, 8, 8), dtype=np.float32)
    seen = np.array([True, False, False, False, True])
    sheet = figures.glyph_sheet([target, output, np.abs(output - target)], ['glyphs', 'glyphs', 'error'],
                                seen=seen, per_row=3)
    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert sheet.shape[1] == 3 * 8 + 2 * 2  # three tiles with a gap between them
    assert sheet.min() >= 0 and sheet.max() <= 1
    # the first tile is black, with the blue bar of a seen glyph below it. The second tile has none.
    assert np.allclose(sheet[0, 0], 0)
    blue = np.array(matplotlib.colors.to_rgb(figures.SERIES_1))
    assert np.allclose(sheet[8 + 2, 0], blue)
    assert not np.allclose(sheet[8 + 2, 8 + 2], blue)


def test_figures_render():
    rng = np.random.default_rng(0)
    glyphs = 6

    def fingerprint():
        return rng.uniform(-1, 1, size=(glyphs, 16, 16)).astype(np.float32)

    sample = {
        'font': 'A Font', 'text': 'متن', 'clean': rng.uniform(-1, 1, (32, 128)), 'input': rng.uniform(-1, 1, (32, 128)),
        'target': fingerprint(), 'output': fingerprint(), 'seen': np.array([True, False] * 3), 'error': 0.3,
        'skill': 0.2, 'rank': 3, 'matches': [('Other', 0.9), ('A Font', 0.8)],
        'style': 'bold', 'style_prediction': 'regular', 'style_confidence': 0.6,
    }
    # the optional parts of a row may be missing
    bare = {key: sample[key] for key in ('font', 'text', 'clean', 'input', 'target', 'output', 'seen', 'error')}
    width, height = draw(figures.plot_panel([sample, bare], num_fonts=10))
    assert width > 800 and height > 300

    case = {'font': 'A Font', 'text': '۱۲۳', 'input': rng.uniform(-1, 1, (32, 128)), 'target': fingerprint(),
            'predicted': fingerprint(), 'predicted_font': 'Other', 'rank': 9, 'similarity': 0.7}
    draw(figures.plot_worst_cases([case], num_fonts=10))
    draw(figures.plot_worst_cases([]))

    draw(figures.plot_glyph_error(list('ابپ۱۲') + [''], np.array([0.1, 0.2, np.nan, 0.3, 0.2, np.nan]),
                                  np.array([0.2, 0.2, 0.3, np.nan, 0.1, np.nan])))
    draw(figures.plot_glyph_error(['a'], np.array([np.nan]), np.array([np.nan])))
    draw(figures.plot_accuracy_by_rank(np.arange(1, 4), np.array([0.5, 0.8, 1.0]), num_fonts=3))
    draw(figures.plot_accuracy_by_rank(np.arange(1, 101), np.linspace(0.2, 1, 100), num_fonts=500))
    draw(figures.plot_bars(['bold', 'regular'], np.array([0.25, 1.0]), np.array([10, 2000]), 'Title', 'Subtitle'))
    draw(figures.plot_confusion(['bold', 'regular', 'shadow'], np.array([[5, 1, 0], [2, 8, 0], [0, 0, 0]])))
    draw(figures.plot_latent_health(rng.uniform(0, 2, 16), rng.uniform(1, 9, 200)))
    draw(figures.plot_latent_comparison(
        rng.normal(size=(6, 16)), [0, 0, 0, 1, 1, 1], ['A Font'] * 3 + ['Other'] * 3, ['متن', '۱۲', 'abc'] * 2,
        mean=np.zeros(16), spread=np.ones(16), font_signal=rng.uniform(0, 1, 16)))
    rates = np.logspace(-5, 0, 50)
    losses = {'total_loss': np.linspace(3, 2, 50), 'loss': np.linspace(1, 0.5, 50),
              'contrastive_loss': np.linspace(4, 3, 50), 'style_loss': np.linspace(2, 1, 50)}
    draw(figures.plot_lr_range_test(rates, losses, losses['total_loss'], {'steepest': 1e-3, 'minimum': 1e-2}))
    draw(figures.plot_lr_range_test(rates, {'total_loss': losses['total_loss']}, losses['total_loss'][:40], {}))
    # a single font has no other font to be compared with
    draw(figures.plot_latent_comparison(
        rng.normal(size=(2, 4)), [0, 0], ['A Font'] * 2, ['a', 'b'], np.zeros(4), np.ones(4), np.ones(4)))


def test_figures_never_use_a_status_color_for_a_series():
    assert figures.SERIES_1 not in (figures.GOOD, figures.CRITICAL)
    assert figures.SERIES_2 not in (figures.GOOD, figures.CRITICAL)


def test_font_signal_tells_font_features_from_text_features():
    report = ValidationReport()
    # feature 0 only depends on the font, feature 1 only on the text
    latent = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    batch = {'text': ['ab'] * 4, 'font_index': torch.tensor([0, 0, 3, 3])}
    report.update(batch, torch.zeros(4, 1, 2, 2), latent, torch.zeros(4, 1, 2, 2))
    assert report.font_signal().tolist() == pytest.approx([1.0, 0.0])


def test_display_text_leaves_shaping_to_a_matplotlib_that_can_do_it(monkeypatch):
    word = 'گونه'
    # reshaping a second time would reverse the word again and break its joins
    monkeypatch.setattr(figures, 'MATPLOTLIB_SHAPES_TEXT', True)
    assert figures.display_text(word) == word

    # an older matplotlib needs the joined letter forms, in visual order
    monkeypatch.setattr(figures, 'MATPLOTLIB_SHAPES_TEXT', False)
    shaped = figures.display_text(word)
    assert shaped != word and len(shaped) == len(word)
    assert all(0xFB50 <= ord(char) <= 0xFEFF for char in shaped)
    assert figures.display_text('abc') == 'abc'
