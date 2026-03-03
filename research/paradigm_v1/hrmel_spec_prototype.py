#!/usr/bin/env python3
"""Prototype API for profile-driven HR-Mel usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import librosa

from src.hr_mel import N_FFT, HOP_LENGTH, WIN_LENGTH, build_hr_mel_basis, encode_hr


@dataclass(frozen=True)
class HRProfile:
    name: str
    bands: List[Dict]


DEFAULT_PROFILES: Dict[str, HRProfile] = {
    "music-v1": HRProfile(
        name="music-v1",
        bands=[
            {"fmin": 0.0, "fmax": 1500.0, "bins": 40, "compression": "log1p"},
            {"fmin": 1500.0, "fmax": 6000.0, "bins": 32, "compression": "log1p"},
            {"fmin": 6000.0, "fmax": 20000.0, "bins": 24, "compression": "sqrt_log1p"},
        ],
    ),
    "music-lowharm": HRProfile(
        name="music-lowharm",
        bands=[
            {"fmin": 0.0, "fmax": 1800.0, "bins": 48, "compression": "log1p"},
            {"fmin": 1800.0, "fmax": 7000.0, "bins": 32, "compression": "log1p"},
            {"fmin": 7000.0, "fmax": 20000.0, "bins": 16, "compression": "log1p"},
        ],
    ),
    "music-transient": HRProfile(
        name="music-transient",
        bands=[
            {"fmin": 0.0, "fmax": 1200.0, "bins": 24, "compression": "log1p"},
            {"fmin": 1200.0, "fmax": 5000.0, "bins": 32, "compression": "log1p"},
            {"fmin": 5000.0, "fmax": 20000.0, "bins": 40, "compression": "sqrt_log1p"},
        ],
    ),
}


class HRMelSpec:
    def __init__(
        self,
        sr: int = 44_100,
        n_fft: int = N_FFT,
        hop: int = HOP_LENGTH,
        win: int = WIN_LENGTH,
        fmax: float = 20_000.0,
        profile: str = "music-v1",
    ) -> None:
        if profile not in DEFAULT_PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        self.fmax = float(min(fmax, sr / 2.0))
        self.profile = profile
        self.bands = DEFAULT_PROFILES[profile].bands

    def encode(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        power = np.abs(
            librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win)
        ) ** 2
        basis, slices = build_hr_mel_basis(self.sr, self.n_fft, self.fmax, bands=self.bands)
        mel = basis @ power
        encoded = encode_hr(mel, slices, self.bands)
        meta = {
            "sr": self.sr,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "win": self.win,
            "fmax": self.fmax,
            "profile": self.profile,
            "bands": self.bands,
        }
        return encoded, meta
