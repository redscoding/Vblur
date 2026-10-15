Global Feature Gate — OCR recovery optimization
This package adds a separate global_feature_gate.py module and integrates it into video_processor.py.
What it changes
- Adds a low-resolution global template-matching check for recovery OCR.
- Uses the gate only when all active tracks are currently lost. Normal periodic OCR and the initialization scan remain unchanged.
- Skips an OCR call only when every confirmed target template has very low global similarity.
- Runs OCR for uncertain/matching frames and forces a periodic OCR safety check.
- Learns templates only from OCR-confirmed target boxes; it does not use stale CSRT boxes as new templates.
- Does not change main.py or the mosaic export format.
Install / apply
1. Make a backup or commit your current branch first.
2. Put global_feature_gate.py beside video_processor.py.
3. Replace video_processor.py with the included version.
4. Keep your existing dependencies. The gate uses OpenCV and NumPy already used by the project.
The module import has a fallback, so if global_feature_gate.py is absent, the original OCR workflow still runs without this optimization.
Test
Run from the project folder:
python -m unittest -v test_global_feature_gate.py
python -m py_compile video_processor.py global_feature_gate.py
Tuning and safety
ABSENCE_THRESHOLD is deliberately low (0.22): OCR is skipped only when the match score is very poor. Matching thresholds depend on footage, scale, motion blur, overlays, and crop padding. Validate on representative videos before raising it. A periodic forced OCR check reduces, but cannot eliminate, the risk of missing a brief appearance between checks.
Git
git status
git add video_processor.py global_feature_gate.py test_global_feature_gate.py README_global_feature_gate.md
git commit -m "feat: add conservative global feature gate for OCR recovery"
git push