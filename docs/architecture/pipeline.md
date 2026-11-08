# MVCP / VCL + Voice Changer Architecture

## High-level data flow

```
                    ┌──────────────────────────────────────┐
                    │         Score / Lyrics / Notes       │
                    │              (optional)              │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  PART A — Score-to-Acoustic (SVS)  or  Source Audio                        │
│  DiffSinger / other offline SVS                                            │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │ mel / waveform
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  PART B — Voice Changer / Identity (SVC)          [OPTIONAL]               │
│  so-vits-svc  ·  RVC  ·  ElevenX (XTTS)  ·  other offline converter        │
│  • SoftVC / content encoder                                                │
│  • F0 conditioning                                                         │
│  • NSF-HiFiGAN or equivalent vocoder                                       │
│  Must be trained only on consented reference audio                         │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │ dry vocal waveform (+ optional F0, onsets)
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  PART C — Vocal Calibration Layer (VCL / MVCP)     [THIS REPO]             │
│                                                                            │
│  Stage 1  Source analysis & mode routing (SING | RAP)                      │
│  Stage 2  Gender-partitioned noise-floor (envelope-gated ΔHNR aspiration)  │
│  Stage 3  Vowel-selective formant warp (low/back only)                     │
│  Stage 4  Mode-specific resonance                                          │
│             SING → drive-dependent 2.8–3.2 kHz clustering                  │
│             RAP  → onset-locked 400 Hz–1.5 kHz transient tilt              │
│  Stage 5  Prosody & phase lock (original F0 + onsets restored)             │
│  Stage 6  Gender partition manager (single coefficient table)              │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │
                                 ▼
                          calibrated vocal
```

## Design principles

1. **Separation of concerns**  
   Identity conversion (Part B) is optional and external. The VCL never trains on or stores speaker embeddings.

2. **Reversibility**  
   Every VCL parameter defaults to a measured tendency but can be set to zero, returning the signal essentially unchanged.

3. **Mode awareness**  
   Singing and rap have conflicting acoustic requirements. The mode flag selects entirely different Stage-4 behaviour.

4. **Gender partition, not two systems**  
   One code path; coefficients are looked up from a table indexed by male/female profile.

5. **Patent posture**  
   Claims recite the ordered DSP steps and measurable technical effects, not demographic labels.

## Integration points

- `voice_changer/adapter.py` — `process_with_vcl()` and `process_file()`  
- `src/vcl.py` — core DSP and presets  
- Upstream converters should preferably expose F0 and onset times so Stage 5 can perform a true phase-locked recombination.
