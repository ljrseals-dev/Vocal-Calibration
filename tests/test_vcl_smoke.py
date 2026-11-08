"""Smoke tests for VCL pipeline (no audio quality claims)."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcl import VCLParams, apply_vcl, presets, Mode, Gender, apply_gender_partition


def test_identity_neutral():
    sr = 16000
    y = np.random.randn(sr).astype(np.float32) * 0.1
    out = apply_vcl(y, sr, presets.neutral(), apply_gender=False)
    assert out.shape == y.shape
    assert np.isfinite(out).all()


def test_gender_partition_changes_params():
    p = VCLParams(gender=Gender.FEMALE)
    p2 = apply_gender_partition(p)
    assert p2.delta_hnr != 0.0
    assert p2.singer_formant_center > 0


def test_presets_run():
    sr = 22050
    y = np.sin(2 * np.pi * 220 * np.linspace(0, 0.5, int(sr * 0.5)))
    for name in ["soul_tenor", "gospel_belt", "rap_flow", "female_soul"]:
        params = getattr(presets, name)()
        out = apply_vcl(y, sr, params)
        assert out.shape == y.shape
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


if __name__ == "__main__":
    test_identity_neutral()
    test_gender_partition_changes_params()
    test_presets_run()
    print("All smoke tests passed.")
