# Multi-Modal Vocal Calibration Pipeline (MVCP) / Vocal Calibration Layer (VCL)

**Author:** Lewis James Seals Jr (ljrseals-dev)  
**Agent assistance:** Anthropic / Claude  
**Copyright © 2026 Lewis James Seals Jr (ljrseals-dev)**

Offline, post-processing DSP layer that shapes a neutral singing or rap signal using a coordinated, gender-partitioned, mode-specific chain of transforms:

- Envelope-gated aspiration / noise-floor calibration driven by a stored harmonics-to-noise deficit (ΔHNR)
- Vowel-class-selective formant warping (low/back vowels only)
- Mode switch: drive-dependent F3–F5 clustering (SING) vs. onset-locked formant-preserving transient tilt (RAP)
- Strict phase-locked recombination of the original F0 contour and onset timing

## Repository contents

| Path | Purpose |
|------|---------|
| `src/vcl.py` | Core VCL / MVCP reference implementation + CLI + presets |
| `voice_changer/` | **Voice-changer / SVC integration layer** (adapter + docs) |
| `docs/MVCP_Provisional_Patent_Draft.docx` | Provisional patent application draft |
| `docs/architecture/pipeline.md` | Full system architecture diagram & notes |
| `headers/` | Drop-in attribution headers for source files |
| `tests/` | Smoke tests |

The VCL is intentionally a **DSP post-processing layer**, not a full synthesizer or identity-cloning system. It is designed to sit **after** a neutral SVS (e.g. DiffSinger) or a voice-changer / SVC stage (so-vits-svc, RVC, ElevenX, etc.).

## Quick start — VCL only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.vcl --demo
python -m src.vcl --preset gospel_belt --in dry_vocal.wav --out calibrated.wav
```

## Quick start — Voice Changer → VCL

1. Run any offline voice changer / SVC model and write a WAV.
2. Pass that WAV through the adapter:

```bash
python -m voice_changer.adapter --in converted.wav --out calibrated.wav --preset gospel_belt
```

Or from Python:

```python
from voice_changer.adapter import VoiceChangerConfig, process_with_vcl, process_file

cfg = VoiceChangerConfig(vcl_preset="soul_tenor")
process_file("svc_output.wav", "final.wav", config=cfg)
```

See `voice_changer/README.md` and `docs/architecture/pipeline.md` for the full recommended stack (DiffSinger / so-vits-svc / RVC / ElevenX → VCL).

## Available VCL presets

- `neutral` — all biases near zero  
- `soul_tenor` — male, singing, moderate ring + light rasp  
- `gospel_belt` — male, singing, stronger nonlinear ring  
- `rap_flow` — male, rap mode, onset-locked transient emphasis  
- `female_soul` — female partition, singing  

All parameters remain fully editable; presets are only starting points.

## Patent status

A provisional patent application draft is included under `docs/`. File with the USPTO to secure a priority date. Claims are drafted around the concrete signal-processing architecture to satisfy current §101 guidance.

## License

MIT — see `LICENSE` and `NOTICE`. The human author is the sole copyright holder.

## Consent notice

Any identity model (SVC / RVC / so-vits-svc / XTTS) used in front of this layer must be trained only on consented, licensed reference audio. Unauthorized voice cloning is outside the intended use of this project and is the sole responsibility of the user.
