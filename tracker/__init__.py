"""Mòdul de finger tracking — captura strokes a partir de la càmera amb MediaPipe."""

from .finger_tracker import FingerTracker, strokes_to_quickdraw

__all__ = ["FingerTracker", "strokes_to_quickdraw"]
