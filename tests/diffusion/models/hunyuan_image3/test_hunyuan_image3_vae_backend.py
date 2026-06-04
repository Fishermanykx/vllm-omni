# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import (
    DistributedAutoencoderKLHunyuan,
    DistributedAutoencoderKLHunyuanOnline,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _get_hunyuan_vae_backend,
)


def _tiny_hunyuan_vae_kwargs():
    return {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "block_out_channels": (32,),
        "layers_per_block": 1,
        "ffactor_spatial": 1,
        "ffactor_temporal": 1,
        "sample_size": 8,
        "sample_tsize": 1,
    }


def test_hunyuan_vae_backend_defaults_to_existing_backend():
    od_config = SimpleNamespace(model_config={})

    assert _get_hunyuan_vae_backend(od_config) == "default"


@pytest.mark.parametrize(
    "alias",
    [
        "online",
        "hunyuan-online",
        "hunyuan_online",
        "hunyuan_image_online",
    ],
)
def test_hunyuan_vae_backend_accepts_online_aliases(alias):
    od_config = SimpleNamespace(model_config={"vae_backend": alias})

    assert _get_hunyuan_vae_backend(od_config) == "hunyuan_image_online"


def test_hunyuan_vae_backend_rejects_unknown_backend():
    od_config = SimpleNamespace(model_config={"vae_backend": "other"})

    with pytest.raises(ValueError, match="Unsupported HunyuanImage3 vae_backend"):
        _get_hunyuan_vae_backend(od_config)


def test_hunyuan_online_vae_keeps_customer_tiling_overlap():
    default_vae = DistributedAutoencoderKLHunyuan(**_tiny_hunyuan_vae_kwargs())
    online_vae = DistributedAutoencoderKLHunyuanOnline(**_tiny_hunyuan_vae_kwargs())

    assert default_vae.tile_overlap_factor == 0.25
    assert online_vae.tile_overlap_factor == 0.125
