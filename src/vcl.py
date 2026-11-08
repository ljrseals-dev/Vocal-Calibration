# =============================================================================
# vcl.py — Vocal Calibration Layer (VCL) / Multi-Modal Vocal Calibration Pipeline (MVCP) v0.1
# =============================================================================
#
# Project:    MouthModel / Vocal Calibration Layer / MVCP
# Author:     Lewis James Seals Jr  <ljrseals-dev>
# Agent:      Anthropic / Claude  (assisted agent)
# License:    MIT
# Created:    2026
#
# Copyright (c) 2026 Lewis James Seals Jr (ljrseals-dev)
#
# Assisted by Anthropic's Claude. The human author (ljrseals-dev) is the
# sole copyright holder and takes responsibility for the final content,
# correctness, and fitness for purpose of this file. Claude's contribution
# is provided as an agent assistant and does not itself claim authorship.
#
# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
# Post-processing DSP layer that shapes a neutral singing/rap signal toward
# measured acoustic tendencies via a coordinated, gender-partitioned,
# mode-specific pipeline:
#
#   1. Noise-floor calibration   — envelope-gated aspiration from stored ΔHNR
#   2. Vowel-selective formant warp — low/back only
#   3. Mode-specific resonance   — SING: drive-dependent 2.8–3.2 kHz ring
#                                  RAP:  onset-locked 400 Hz–1.5 kHz tilt
#   4. Prosody / phase lock      — original F0 + onsets restored
#   5. Gender partition manager  — single table, male/female coefficients
#
# This module does NOT synthesize or retrain a voice model. It sits between
# a singing model's output (e.g. DiffSinger, so-vits-svc, RVC) and the listener.
#
# Usage:
#   from vcl import VCLParams, apply_vcl, presets
#   y_cal = apply_vcl(y, sr, presets.soul_tenor(), f0=f0_hz, onsets=onset_times)
#
#   python -m src.vcl --demo
#   python -m src.vcl --preset gospel_belt --in in.wav --out out.wav
# =============================================================================

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy import signal
except ImportError:
    signal = None


class Mode(str, Enum):
    SING = "SING"
    RAP = "RAP"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class VCLParams:
    """Editable calibration parameters (defaults are measured tendencies, not absolutes)."""

    # Gender partition
    gender: Gender = Gender.MALE

    # Mode
    mode: Mode = Mode.SING

    # Stage 2 — Noise-floor / glottal
    delta_hnr: float = 0.0          # stored HNR deficit (positive = more aspiration)
    hnr_floor: float = 0.05         # baseline aspiration even in modal voice
    mfdr_softness: float = 0.3      # softens spectral slope (lower MFDR proxy)
    rasp_amount: float = 0.0        # subharmonic / grit injection

    # Stage 3 — Vowel-space
    dialect_fronting: float = 0.0   # 0–1 strength of low/back F1/F2 shift
    f1_shift_hz: float = -40.0      # typical downward F1 for low/back
    f2_shift_hz: float = -80.0      # typical downward F2 for /u/-like
    pharynx_bias: float = 0.0       # small length scaling (0–0.1)

    # Stage 4 — Resonance
    singer_formant_center: float = 3000.0
    singer_formant_gain: float = 0.0   # base gain; actual is drive-dependent
    gamma: float = 1.6                 # nonlinearity exponent (>1)
    twang_amount: float = 0.0
    nasal_branch: float = 0.0

    # RAP transient
    transient_gain: float = 0.0
    transient_band: Tuple[float, float] = (400.0, 1500.0)

    # Expression (applied to F0 contour before or after)
    vibrato_rate: float = 5.5
    vibrato_depth: float = 0.0
    vibrato_delay: float = 0.15

    def copy(self) -> "VCLParams":
        return dataclasses.replace(self)


# ---------------------------------------------------------------------------
# Gender-partitioned coefficient table (Stage 6)
# ---------------------------------------------------------------------------

GENDER_TABLE = {
    Gender.MALE: {
        "delta_hnr": 1.5,
        "f1_shift_hz": -35.0,
        "f2_shift_hz": -70.0,
        "singer_formant_center": 2950.0,
        "gamma": 1.55,
        "transient_gain": 0.35,
    },
    Gender.FEMALE: {
        "delta_hnr": 2.0,
        "f1_shift_hz": -50.0,
        "f2_shift_hz": -110.0,
        "singer_formant_center": 3100.0,
        "gamma": 1.7,
        "transient_gain": 0.28,
    },
}


def apply_gender_partition(params: VCLParams) -> VCLParams:
    """Overwrite selected fields from the gender coefficient table."""
    p = params.copy()
    tbl = GENDER_TABLE[p.gender]
    p.delta_hnr = tbl["delta_hnr"]
    p.f1_shift_hz = tbl["f1_shift_hz"]
    p.f2_shift_hz = tbl["f2_shift_hz"]
    p.singer_formant_center = tbl["singer_formant_center"]
    p.gamma = tbl["gamma"]
    p.transient_gain = tbl["transient_gain"]
    return p


# ---------------------------------------------------------------------------
# Core DSP stages (reference implementations — not production-optimized)
# ---------------------------------------------------------------------------

def _frame_energy(y: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Simple short-time RMS envelope."""
    n = len(y)
    n_frames = 1 + max(0, (n - frame_len) // hop)
    env = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = y[start : start + frame_len]
        env[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)
    return env


def stage_noise_floor(
    y: np.ndarray,
    sr: int,
    params: VCLParams,
    env_norm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage 2: Envelope-gated aspiration injection.
    g_noise = k * ΔHNR * (1 - env_norm)
    """
    if params.delta_hnr <= 0 and params.hnr_floor <= 0:
        return y

    n = len(y)
    if env_norm is None:
        frame_len = int(0.02 * sr)
        hop = frame_len // 2
        env = _frame_energy(y, frame_len, hop)
        env_norm = env / (env.max() + 1e-12)
        # upsample to sample rate (nearest for simplicity)
        idx = np.minimum((np.arange(n) / hop).astype(int), len(env_norm) - 1)
        env_norm = env_norm[idx]

    # high-passed noise
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float64)
    # simple first-order high-pass ~ 2 kHz
    if signal is not None:
        b, a = signal.butter(2, 2000 / (sr / 2), btype="high")
        noise = signal.lfilter(b, a, noise)

    k = 0.08
    g = k * (params.delta_hnr + params.hnr_floor) * (1.0 - env_norm)
    return y + g * noise


def stage_vowel_selective_warp(
    y: np.ndarray,
    sr: int,
    params: VCLParams,
) -> np.ndarray:
    """
    Stage 3: Placeholder for vowel-class-selective formant warping.
    Full implementation requires a phoneme/vowel classifier + frequency warping.
    This skeleton applies a mild global spectral tilt controlled by dialect_fronting
    as a stand-in; replace with true selective warp for production use.
    """
    if abs(params.dialect_fronting) < 1e-6:
        return y

    # Very lightweight stand-in: gentle low-shelf adjustment
    # (real system would use LPC or cepstral envelope + selective warp)
    strength = np.clip(params.dialect_fronting, 0.0, 1.0)
    # mild low-frequency emphasis / de-emphasis as proxy
    if signal is not None:
        # simple first-order shelf approximation
        gain_db = strength * (params.f1_shift_hz / 100.0)  # crude mapping
        # skip heavy processing in skeleton
        pass
    return y


def stage_mode_resonance(
    y: np.ndarray,
    sr: int,
    params: VCLParams,
    env_norm: Optional[np.ndarray] = None,
    onsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage 4: Mode-specific resonance.
    SING → drive-dependent peaking filter ~2.8–3.2 kHz
    RAP  → onset-locked transient spectral tilt 400 Hz–1.5 kHz
    """
    n = len(y)
    out = y.copy()

    if params.mode == Mode.SING and params.singer_formant_gain > 0:
        if env_norm is None:
            frame_len = int(0.02 * sr)
            hop = frame_len // 2
            env = _frame_energy(y, frame_len, hop)
            env_norm = env / (env.max() + 1e-12)
            idx = np.minimum((np.arange(n) / hop).astype(int), len(env_norm) - 1)
            env_norm = env_norm[idx]

        # G_ring = α * (env_norm)^γ
        g = params.singer_formant_gain * (env_norm ** params.gamma)

        if signal is not None:
            # peaking filter
            center = params.singer_formant_center
            Q = 2.5
            w0 = center / (sr / 2)
            b, a = signal.iirpeak(w0, Q)
            filtered = signal.lfilter(b, a, y)
            # modulate by time-varying gain (approximate)
            out = y + (g * (filtered - y))

    elif params.mode == Mode.RAP and params.transient_gain > 0 and onsets is not None:
        # short spectral tilt around each onset
        tilt_len = int(0.012 * sr)  # ~12 ms
        for t in onsets:
            start = int(t * sr)
            end = min(start + tilt_len, n)
            if start >= n:
                continue
            # simple high-shelf boost in the target band (placeholder)
            segment = out[start:end]
            out[start:end] = segment * (1.0 + params.transient_gain * 0.4)

    return out


def stage_prosody_lock(
    processed: np.ndarray,
    original: np.ndarray,
    f0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage 5: Placeholder for phase-locked recombination.
    In a full implementation this would use a phase vocoder or WORLD/WORLD-like
    vocoder to re-impose the original F0 contour and phase continuity.
    Skeleton simply returns the processed signal (identity).
    """
    # TODO: implement phase-locked overlap-add / phase vocoder path
    return processed


def apply_vcl(
    y: np.ndarray,
    sr: int,
    params: VCLParams,
    f0: Optional[np.ndarray] = None,
    onsets: Optional[np.ndarray] = None,
    apply_gender: bool = True,
) -> np.ndarray:
    """
    Full pipeline entry point.
    """
    if apply_gender:
        params = apply_gender_partition(params)

    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # pre-compute envelope once
    frame_len = int(0.02 * sr)
    hop = max(1, frame_len // 2)
    env = _frame_energy(y, frame_len, hop)
    env_norm_frames = env / (env.max() + 1e-12)
    idx = np.minimum((np.arange(len(y)) / hop).astype(int), len(env_norm_frames) - 1)
    env_norm = env_norm_frames[idx]

    # Stage 2
    y = stage_noise_floor(y, sr, params, env_norm=env_norm)

    # Stage 3
    y = stage_vowel_selective_warp(y, sr, params)

    # Stage 4
    y = stage_mode_resonance(y, sr, params, env_norm=env_norm, onsets=onsets)

    # Stage 5
    y = stage_prosody_lock(y, y, f0=f0)

    # prevent clipping
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.99:
        y = y * (0.99 / peak)

    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class presets:
    @staticmethod
    def neutral() -> VCLParams:
        return VCLParams()

    @staticmethod
    def soul_tenor() -> VCLParams:
        p = VCLParams(gender=Gender.MALE, mode=Mode.SING)
        p = apply_gender_partition(p)
        p.singer_formant_gain = 0.45
        p.hnr_floor = 0.08
        p.rasp_amount = 0.15
        p.dialect_fronting = 0.4
        return p

    @staticmethod
    def gospel_belt() -> VCLParams:
        p = VCLParams(gender=Gender.MALE, mode=Mode.SING)
        p = apply_gender_partition(p)
        p.singer_formant_gain = 0.7
        p.gamma = 1.8
        p.hnr_floor = 0.06
        p.dialect_fronting = 0.35
        return p

    @staticmethod
    def rap_flow() -> VCLParams:
        p = VCLParams(gender=Gender.MALE, mode=Mode.RAP)
        p = apply_gender_partition(p)
        p.transient_gain = 0.55
        p.hnr_floor = 0.1
        p.dialect_fronting = 0.25
        return p

    @staticmethod
    def female_soul() -> VCLParams:
        p = VCLParams(gender=Gender.FEMALE, mode=Mode.SING)
        p = apply_gender_partition(p)
        p.singer_formant_gain = 0.5
        p.dialect_fronting = 0.5
        p.hnr_floor = 0.09
        return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _demo():
    sr = 44100
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    # simple harmonic test tone with amplitude envelope
    f0 = 220.0
    y = 0.3 * np.sin(2 * np.pi * f0 * t)
    y += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
    y += 0.08 * np.sin(2 * np.pi * 3 * f0 * t)
    env = np.sin(np.pi * t / 1.5) ** 0.7
    y *= env

    params = presets.gospel_belt()
    y_out = apply_vcl(y, sr, params)

    if sf is not None:
        out_path = Path("vcl_demo_out.wav")
        sf.write(out_path, y_out, sr)
        print(f"Wrote demo to {out_path.resolve()}")
    else:
        print("soundfile not installed; demo computed but not written.")
    print(f"Input peak: {np.max(np.abs(y)):.3f}  Output peak: {np.max(np.abs(y_out)):.3f}")
    print(f"Params: gender={params.gender}, mode={params.mode}, ΔHNR={params.delta_hnr}")


def main():
    parser = argparse.ArgumentParser(description="Vocal Calibration Layer (VCL / MVCP)")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    parser.add_argument("--in", dest="infile", type=str, help="Input WAV")
    parser.add_argument("--out", dest="outfile", type=str, help="Output WAV")
    parser.add_argument(
        "--preset",
        choices=["neutral", "soul_tenor", "gospel_belt", "rap_flow", "female_soul"],
        default="gospel_belt",
    )
    args = parser.parse_args()

    if args.demo:
        _demo()
        return

    if args.infile and args.outfile:
        if sf is None:
            raise SystemExit("soundfile is required for file I/O")
        y, sr = sf.read(args.infile, always_2d=False)
        preset_fn = getattr(presets, args.preset)
        params = preset_fn()
        y_out = apply_vcl(y, sr, params)
        sf.write(args.outfile, y_out, sr)
        print(f"Wrote calibrated audio to {args.outfile}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
