# =============================================================================
# adapter.py — Voice-Changer → VCL integration adapter
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
# Assisted by Anthropic's Claude. The human author is the sole copyright
# holder. Claude is credited as assisting agent only.
# =============================================================================

"""
Thin adapter that takes the output of an external voice-changer / SVC
process and routes it through the VCL.

This module does not implement so-vits-svc, RVC, or XTTS itself.
It only provides a clean call signature and configuration so that
any offline converter can be dropped in front of the VCL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Local VCL import (works when run from repo root or as package)
try:
    from src.vcl import VCLParams, apply_vcl, presets, Mode, Gender
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vcl import VCLParams, apply_vcl, presets, Mode, Gender


@dataclass
class VoiceChangerConfig:
    """Configuration for the combined SVC → VCL chain."""

    # Which VCL preset (or custom params) to apply after conversion
    vcl_preset: str = "gospel_belt"          # neutral | soul_tenor | gospel_belt | rap_flow | female_soul
    custom_params: Optional[VCLParams] = None

    # Optional F0 / onset hand-off from the upstream converter
    # (many SVC models already expose these; pass them through for better phase lock)
    pass_f0: bool = True
    pass_onsets: bool = True

    # Output peak safety
    peak_limit: float = 0.99


def load_preset(name: str) -> VCLParams:
    fn = getattr(presets, name, None)
    if fn is None:
        raise ValueError(f"Unknown VCL preset: {name}. Choose from presets.*")
    return fn()


def process_with_vcl(
    waveform: np.ndarray,
    sample_rate: int,
    config: Optional[VoiceChangerConfig] = None,
    f0: Optional[np.ndarray] = None,
    onsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply the Vocal Calibration Layer to a voice-changer / SVC output.

    Parameters
    ----------
    waveform : np.ndarray
        Mono (or mixed-to-mono) audio from the upstream converter.
    sample_rate : int
        Sample rate of the waveform.
    config : VoiceChangerConfig, optional
        Controls which VCL preset/params to use.
    f0 : np.ndarray, optional
        Original or converter-supplied F0 contour (Hz). Improves Stage 5 lock.
    onsets : np.ndarray, optional
        Onset times in seconds. Used by RAP-mode transient stage.

    Returns
    -------
    np.ndarray
        Calibrated mono waveform (float32).
    """
    if config is None:
        config = VoiceChangerConfig()

    if config.custom_params is not None:
        params = config.custom_params
    else:
        params = load_preset(config.vcl_preset)

    # Respect the pass_* flags
    f0_arg = f0 if config.pass_f0 else None
    onset_arg = onsets if config.pass_onsets else None

    out = apply_vcl(
        waveform,
        sample_rate,
        params,
        f0=f0_arg,
        onsets=onset_arg,
    )

    # Final peak protection
    peak = np.max(np.abs(out)) + 1e-12
    if peak > config.peak_limit:
        out = out * (config.peak_limit / peak)

    return out.astype(np.float32)


def process_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[VoiceChangerConfig] = None,
) -> Path:
    """
    Convenience: read a WAV produced by any voice changer, run VCL, write result.
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError("soundfile is required for process_file()") from e

    input_path = Path(input_path)
    output_path = Path(output_path)

    y, sr = sf.read(str(input_path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    y_out = process_with_vcl(y, sr, config=config)
    sf.write(str(output_path), y_out, sr)
    return output_path


# ---------------------------------------------------------------------------
# CLI for quick post-processing of any voice-changer output
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply VCL calibration to the output of a voice changer / SVC model"
    )
    parser.add_argument("--in", dest="infile", required=True, help="Input WAV from voice changer")
    parser.add_argument("--out", dest="outfile", required=True, help="Calibrated output WAV")
    parser.add_argument(
        "--preset",
        default="gospel_belt",
        choices=["neutral", "soul_tenor", "gospel_belt", "rap_flow", "female_soul"],
        help="VCL preset to apply",
    )
    args = parser.parse_args()

    cfg = VoiceChangerConfig(vcl_preset=args.preset)
    out = process_file(args.infile, args.outfile, config=cfg)
    print(f"Wrote VCL-calibrated file → {out}")


if __name__ == "__main__":
    main()
