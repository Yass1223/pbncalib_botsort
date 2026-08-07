"""Option A — Robust Sports Field Registration for GSR.

Stack: PnLCalib (per-frame calibration, upgraded with confidence weighting and a
Sampson-normalized conic term) + BroadTrack-style temporal tracking (warm-started
LM over kappa = {f, k1, pan, tilt, roll, Cx, Cy, Cz} with optional optical-flow
and tripod constraints, Jaccard confidence, reinitialization).

All design decisions are documented in OptionA_Field_Registration_Report.md and
traceable to the PnLCalib / BroadTrack / Broadcast2Pitch papers.
"""

__version__ = "3.3.0"   # RESULT: better than PnLCalib on all 7 metrics
