# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from pathlib import Path
import inspect
import warnings
import torch


from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs

Model = Union[Demucs, HDemucs, HTDemucs]


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get("__quantized"):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state["quantized"])
        else:
            _check_diffq()
            from diffq import restore_quantized_state

            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)

    if isinstance(model, HTDemucs):
        model.segment = float(model.segment)

    return state


def load_model(path_or_package, strict=False) -> Model:
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, "cpu")
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]
    set_state(model, state)

    return model
