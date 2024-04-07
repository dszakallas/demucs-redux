# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import functools


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


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
    return state
