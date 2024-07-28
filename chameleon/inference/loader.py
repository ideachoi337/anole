# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import glob
import inspect
import json
from pathlib import Path

import torch

from chameleon.inference.transformer import ModelArgs, Transformer

from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def _convert(model_args: ModelArgs, consolidated_path: Path) -> Transformer:
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    #with init_empty_weights():
    model = Transformer(model_args)

    device_map = {
        'tok_embeddings': 0, 
        'layers.0': 0, 
        'layers.1': 0, 
        'layers.2': 0, 
        'layers.3': 0, 
        'layers.4': 0, 
        'layers.5': 0, 
        'layers.6': 0, 
        'layers.7': 0, 
        'layers.8': 0, 
        'layers.9': 1, 
        'layers.10': 1, 
        'layers.11': 1, 
        'layers.12': 1, 
        'layers.13': 1, 
        'layers.14': 1, 
        'layers.15': 1, 
        'layers.16': 1, 
        'layers.17': 1, 
        'layers.18': 2, 
        'layers.19': 2, 
        'layers.20': 2, 
        'layers.21': 2, 
        'layers.22': 2, 
        'layers.23': 2, 
        'layers.24': 2, 
        'layers.25': 3, 
        'layers.26': 3, 
        'layers.27': 3, 
        'layers.28': 3, 
        'layers.29': 3, 
        'layers.30': 3, 
        'layers.31': 3, 
        'norm': 3, 
        'output': 3, 
    }
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(consolidated_path),
        device_map=device_map
    )
    """
    model = Transformer(model_args)

    transfer_results = model.load_state_dict(
        torch.load(str(consolidated_path), map_location='cuda'),
        strict=False,
    )

    # TODO: More generally, assert missing or unexpected keys are buffers.
    assert transfer_results.missing_keys == []
    assert transfer_results.unexpected_keys == ["rope.freqs"]
    """

    model.eval()

    torch.set_default_dtype(old_default_dtype)
    return model


def _get_checkpoint_path(src_dir: Path, rank: int | None) -> Path:
    base_path = src_dir / "consolidated.pth"
    if not rank and base_path.exists():
        return base_path

    alt_path = src_dir / f"consolidated.{rank:02}.pth"
    if alt_path.exists():
        return alt_path

    raise ValueError("Consolidated checkpoint not found.")


def load_model(path: str, rank: int | None = None) -> Transformer:
    src_dir = Path(path)

    with open(src_dir / "params.json", "r") as f:
        params = json.loads(f.read())
    with open(src_dir / "consolidate_params.json", "r") as f:
        consolidate_params = json.loads(f.read())
    params = {**params, **params["model"], **consolidate_params}

    known_param = inspect.signature(ModelArgs.__init__).parameters
    filtered_params = {k: v for k, v in params.items() if k in known_param}

    return _convert(
        ModelArgs(**filtered_params),
        _get_checkpoint_path(src_dir, rank),
    )


def detect_shard_count(path: str) -> int:
    src_dir = Path(path)
    if (src_dir / "consolidated.pth").exists():
        return 1
    return len(glob.glob(str(src_dir / "consolidated.*.pth")))
