# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""
from typing import Optional

import torch as th


def spectro(x, n_fft: int = 512, hop_length: int = 0, pad: int = 0):
    # *other, length = x.shape
    other, length = x.shape[:-1], x.shape[-1]
    x = x.reshape(-1, length)
    is_mps_xpu = x.device.type in ["mps", "xpu"]
    if is_mps_xpu:
        x = x.cpu()
    if hop_length == 0:
        hop_length = n_fft // 4
    z = th.stft(
        x,
        n_fft * (1 + pad),
        hop_length,
        window=th.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frame = z.shape
    return z.view(other + (freqs, frame))


def ispectro(z, hop_length: int = 0, length: Optional[int] = None, pad: int = 0):
    other, freqs, frames = z.shape[:-2], z.shape[-2], z.shape[-1]
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps_xpu = z.device.type in ["mps", "xpu"]
    if is_mps_xpu:
        z = z.cpu()
    x = th.istft(
        z,
        n_fft,
        hop_length,
        window=th.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, length = x.shape
    return x.view(other + (length,))
