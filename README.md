# The-Mouth-Bottom-Instrument
Perfect — a mouth/vocal tract physical model. That's a rich space: lips, teeth, tongue position, jaw, throat resonance, breath pressure. The surprise topology for this is a **Kelly-Lochbaum vocal tract waveguide** — a chain of acoustic tube segments with reflection coefficients at each junction, which is literally how the vocal tract works physically. Pair it with a glottal pulse source (Liljencrants-Fant model) for voiced sounds and a turbulent noise source for fricatives, and you get the full phoneme space.The surprise topology is a **Kelly-Lochbaum vocal tract waveguide** — the same physical model that underpins commercial vocoders, singing synthesizers, and speech research tools since the 1960s. Here's the architecture:

**The tract as a tube chain**

Eight cylindrical segments from glottis to lips, each with a forward and backward travelling wave. At every junction between segments, the area ratio determines how much energy reflects vs. transmits — exactly how a real throat works. Vowel formants (F1–F4) emerge naturally as standing-wave patterns without any explicit filter tuning.

**Two sources, one tract**

The LF glottal pulse models the asymmetric buzzy shape of real vocal folds — slow open phase, abrupt closure. `drive` steepens the closure exponential, which at high values produces vocal fry character. Turbulent noise can be injected at the glottis (for breathiness) or at segment 5 (the alveolar constriction, for /s/ and /f/ fricatives).

**Five vowel targets, continuously morphed**

/ah/ → /eh/ → /ee/ → /oh/ → /oo/ across the `vowel` knob. The area profiles are linguistically grounded — each profile reflects the actual cross-sectional geometry of the tract for that phoneme. `jaw`, `tongue`, and `lips` then act as articulatory offsets on top of the vowel target, so you can shape the timbre independently of the vowel identity.

**Where to take it next**

The natural extensions are a formant transition envelope (for consonant-vowel transitions), a proper side-branch nasal resonator (replacing the shelf approximation), and polyphonic glottal sources for choir/ensemble use.
