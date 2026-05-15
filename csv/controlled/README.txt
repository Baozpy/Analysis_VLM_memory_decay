Controlled Dialogues Mini-Suite
================================

Goal
- Generate small, controlled multi-turn, multi-image dialogues to probe MEMORY DECAY
  under three views: B2 (Δ since first mention), D (k-th revisit), B1 (normalized progress τ).

Folder
- controlled/
  - generate_controlled.py   # main generator: reads prompts/*.json, calls your models, auto-writes controlled_samples.py
  - runner_stub.py           # replace ask_models() with your BLIP2/LLaVA/Qwen callers
  - controlled_samples.py    # produced arrays: SAMPLE<n>_* and CONTROLLED_IDS
  - prompts/
      - delta_probe.json     # B2
      - revisit_probe.json   # D
      - span_probe.json      # B1

Quick Start
1) Put your images on disk. Edit prompts/*.json to point to them (absolute or relative paths).
2) Edit runner_stub.py -> implement ask_models(turn_text, image_paths) using your inference code.
   Must return: {"blip2": "...", "llava": "...", "qwen": "..."} (free-form strings).
3) Run:
   python controlled/generate_controlled.py
   -> controlled/controlled_samples.py will be generated.
4) Feed controlled_samples.py to your existing pipeline (build_stats_from_samples.py etc).

Scoring
- Prompts JSON include gold answers + aliases. We do case-folding, strip punctuation,
  and accept alias OR gold as substring.

Tips
- Use temperature=0 for stability.
- All probes are one-word/short-answer to simplify automatic grading.
