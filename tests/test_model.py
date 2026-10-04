import pytest
import torch

from font_reconstructor.model import CompactAutoEncoder, GlyphDiscriminator


@pytest.mark.parametrize('base_channels', [4, 8, 16])
def test_output_shape(base_channels):
    model = CompactAutoEncoder(base_channels=base_channels, latent_dim=24)
    x = torch.zeros(2, 1, 32, 128)

    latent = model.encode(x)
    assert latent.shape == (2, 24)
    assert model.decode(latent).shape == (2, 42, 32, 32)
    assert model(x).shape == (2, 42, 32, 32)


def test_shapes_follow_dims():
    model = CompactAutoEncoder(decoder_output_channels=7, input_dims=(64, 96), output_dims=(16, 48))
    assert model(torch.zeros(2, 1, 64, 96)).shape == (2, 7, 16, 48)


def test_forward_can_return_the_latent():
    model = CompactAutoEncoder(latent_dim=16)
    x = torch.zeros(2, 1, 32, 128)
    output, latent = model(x, return_latent=True)
    assert output.shape == (2, 42, 32, 32)
    assert latent.shape == (2, 16)

    model.eval()
    assert torch.equal(model(x), model(x, return_latent=True)[0])


def test_encoder_accepts_any_width():
    # the encoder pools over the image, so texts of any length map to the same latent size
    model = CompactAutoEncoder().eval()
    for width in (64, 128, 300):
        assert model.encode(torch.zeros(1, 1, 32, width)).shape == (1, 32)


def test_default_model_is_small():
    parameters = sum(p.numel() for p in CompactAutoEncoder().parameters())
    assert parameters < 150_000


def test_unreachable_dims_are_rejected():
    with pytest.raises(ValueError):
        CompactAutoEncoder(output_dims=(30, 32))
    with pytest.raises(ValueError):
        CompactAutoEncoder(input_dims=(8, 128))


def parameters(module):
    return sum(p.numel() for p in module.parameters())


def test_decoder_size_leaves_the_encoder_alone():
    default = CompactAutoEncoder()
    wide = CompactAutoEncoder(decoder_channels=32, decoder_blocks=2)

    # only the encoder is used to identify fonts, and it stays the same
    for part in ('encoder', 'to_latent'):
        assert parameters(getattr(wide, part)) == parameters(getattr(default, part))
    assert parameters(wide.decoder) > 10 * parameters(default.decoder)
    assert wide(torch.zeros(2, 1, 32, 128)).shape == (2, 42, 32, 32)

    # the default layout is the one of checkpoints written before the decoder options existed
    assert parameters(default) == 121_714
    assert 'decoder.4.weight' in default.state_dict()


def test_conditioned_decoder():
    model = CompactAutoEncoder(decoder_type='conditioned', decoder_channels=8, decoder_output_channels=7,
                               output_dims=(16, 32))
    output, latent = model(torch.zeros(3, 1, 32, 128), return_latent=True)
    assert output.shape == (3, 7, 16, 32)
    assert latent.shape == (3, 32)

    # one decoder draws every glyph, told apart by the character embedding alone
    assert model.decoder[-2].out_channels == 1
    model.eval()
    glyphs = model.decode(torch.randn(1, 32))[0]
    assert not torch.allclose(glyphs[0], glyphs[1])

    # the glyphs of a sample do not depend on the other samples of a batch
    latent = torch.randn(4, 32)
    assert torch.allclose(model.decode(latent)[2], model.decode(latent[2:3])[0], atol=1e-5)

    # single glyphs are the same as when the whole fingerprint is drawn
    picked = torch.tensor([[0, 6], [3, 3], [5, 1], [2, 0]])
    some = model.decode_glyphs(latent, picked)
    assert some.shape == (4, 2, 16, 32)
    whole = model.decode(latent)
    for sample in range(4):
        for position in range(2):
            assert torch.allclose(some[sample, position], whole[sample, picked[sample, position]], atol=1e-5)
    output, _ = model(torch.zeros(4, 1, 32, 128), return_latent=True, glyphs=picked)
    assert output.shape == (4, 2, 16, 32)
    # all glyphs of many samples are drawn in several passes with the same result
    model._GLYPHS_PER_PASS = 5
    assert torch.allclose(model.decode(latent), whole, atol=1e-5)

    with pytest.raises(ValueError):
        CompactAutoEncoder(decoder_type='attention')
    with pytest.raises(ValueError):
        CompactAutoEncoder(decoder_blocks=0)


def test_joint_decoder_picks_single_glyphs_from_the_fingerprint():
    model = CompactAutoEncoder(decoder_output_channels=7).eval()
    latent = torch.randn(3, 32)
    picked = torch.tensor([[6, 0, 0], [1, 2, 3], [4, 4, 5]])
    some = model.decode_glyphs(latent, picked)
    assert some.shape == (3, 3, 32, 32)
    whole = model.decode(latent)
    assert torch.equal(some[1, 2], whole[1, 3]) and torch.equal(some[0, 1], whole[0, 0])


def test_glyph_discriminator():
    discriminator = GlyphDiscriminator(in_channels=7, base_channels=8)
    scores = discriminator(torch.zeros(3, 7, 32, 32))
    # a grid of scores, each judging a patch of the fingerprint
    assert scores.shape[:2] == (3, 1) and scores.shape[2] > 1 and scores.shape[3] > 1


def test_style_head():
    model = CompactAutoEncoder(latent_dim=16, style_classes=5)
    output, latent = model(torch.zeros(3, 1, 32, 128), return_latent=True)
    assert model.predict_style(latent).shape == (3, 5)
    # a single linear layer on the latent vector
    assert parameters(model.style_head) == 16 * 5 + 5

    without = CompactAutoEncoder(latent_dim=16)
    assert without.style_head is None
    with pytest.raises(ValueError):
        without.predict_style(latent)
