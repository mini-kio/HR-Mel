#!/usr/bin/env python3
"""Standard Mel and Log-Mel helpers."""

from typing import Tuple

import numpy as np
import librosa


def mel_power(
    stft_power: np.ndarray,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmax: float,
    norm: str = "slaney",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax, norm=norm)
    mel = basis @ stft_power
    pinv = np.linalg.pinv(basis)
    return mel, basis, pinv


def log_compress(values: np.ndarray) -> np.ndarray:
    return np.log1p(values)


def log_decompress(values: np.ndarray) -> np.ndarray:
    return np.expm1(values)
