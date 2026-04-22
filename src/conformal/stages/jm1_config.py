"""
PROMISE JM1 Cascade Configuration.

Single-stage cascade for defect prediction:
  Stage 0 (Defect Prediction): defective vs clean modules

The cascade value here is CONFIDENCE GATING:
  - When we say "defective", we're 90%+ confident
  - Reduces false positives (costly developer time on false alarms)
  - Uncertain modules deferred to manual code review

This is the simplest cascade case but demonstrates that even
a single ConfidenceStage with selective prediction outperforms
a flat classifier at matched precision/recall targets.
"""

JM1_CLASSES = {0: 'clean', 1: 'defective'}
