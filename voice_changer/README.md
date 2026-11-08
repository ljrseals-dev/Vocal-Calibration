# Voice Changer Integration (Optional Part B)

This directory provides a thin, offline-friendly integration layer between a **singing-voice conversion (SVC) / voice-changer** front-end and the **Vocal Calibration Layer (VCL)**.

The VCL is deliberately a post-processing DSP stage. It does **not** perform identity cloning. When you need a specific target singer’s timbre, you run an SVC model first, then pass the result through the VCL for the gender-partitioned, mode-specific acoustic calibration described in the provisional patent.

## Supported / recommended offline stacks

| Framework              | Role                          | Notes                                      |
|------------------------|-------------------------------|--------------------------------------------|
| so-vits-svc            | Full offline SVC              | SoftVC content encoder + NSF-HiFiGAN       |
| RVC (Retrieval-based)  | Lighter SVC with retrieval    | Reduces timbre leakage                     |
| ElevenX (ljrseals-dev) | Local XTTS-v2 reference synth | Good for speech-to-singing experiments     |
| DiffSinger             | Score → mel SVS               | Use when you have lyrics + notes           |

None of these models or weights are vendored here. You must obtain them yourself and respect all consent / licensing requirements.

## Pipeline (recommended)

```
[ Lyrics + Notes ]  or  [ Source audio ]
          │
          ▼
┌─────────────────────────┐
│  SVS or SVC stage       │  ← DiffSinger / so-vits-svc / RVC / ElevenX
│  (optional identity)    │
└──────────┬──────────────┘
           │ dry vocal waveform
           ▼
┌─────────────────────────┐
│  VCL Spectral Shaper    │  ← this repo (src/vcl.py)
│  - noise-floor / ΔHNR   │
│  - vowel-selective warp │
│  - SING / RAP resonance │
│  - phase / prosody lock │
└──────────┬──────────────┘
           │ calibrated vocal
           ▼
        [ output ]
```

## Consent & legal

Any SVC / RVC / XTTS model trained on a real person’s voice **must** use only consented, licensed reference material. The VCL itself does not clone identity; misuse of upstream converters is the user’s responsibility. See the root `NOTICE` file.
