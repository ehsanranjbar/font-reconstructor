import pytest
import torch

from font_reconstructor.model import AE2, AutoEncoder


@pytest.mark.parametrize('base_conv_filters', [8, 16, 32])
@pytest.mark.parametrize('batch_norm', [False, True])
def test_output_shape(base_conv_filters, batch_norm):
    model = AutoEncoder(base_conv_filters=base_conv_filters, batch_norm=batch_norm, latent_dim=64)
    x = torch.zeros(2, 1, 32, 128)

    latent = model.encode(x)
    assert latent.shape == (2, 64)
    assert model.decode(latent).shape == (2, 42, 32, 32)
    assert model(x).shape == (2, 42, 32, 32)


def test_shapes_follow_dims():
    model = AutoEncoder(decoder_output_channels=7, input_dims=(64, 96), output_dims=(16, 48))
    assert model(torch.zeros(1, 1, 64, 96)).shape == (1, 7, 16, 48)


def test_default_layout_matches_original_ae2():
    # checkpoints of the original AE2 have to keep loading
    state = AE2(latent_dim=256).state_dict()
    assert state['encoder.fc.weight'].shape == (256, 1024)
    assert state['decoder.fc.weight'].shape == (2048, 256)
    assert state['encoder.conv_layer_0.conv2d.weight'].shape == (16, 1, 3, 3)
    assert state['decoder.t_conv_layer_0.conv_transposed_2d.weight'].shape == (512, 256, 3, 3)
    assert state['decoder.conv_transposed_2d.weight'].shape == (64, 42, 3, 3)


def test_unreachable_output_dims_are_rejected():
    with pytest.raises(ValueError):
        AutoEncoder(output_dims=(30, 30))
    with pytest.raises(ValueError):
        AutoEncoder(kernel_size=5)
