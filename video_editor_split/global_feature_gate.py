"""Conservative global visual gate for reducing unnecessary recovery OCR calls.

The gate uses low-resolution template matching as a cheap *screening* stage.
It never declares that a target is absent; it only recommends skipping an OCR
call when every stored target template scores very poorly. Uncertain cases run OCR.
A periodic forced OCR check limits the time a target can remain undetected.

Requires: opencv-python (already required by the application), numpy.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class GateDecision:
    should_run_ocr: bool
    reason: str
    score: Optional[float] = None


class GlobalFeatureGate:
    """A conservative, low-resolution global appearance gate.

    Templates are crops from OCR-confirmed target boxes. Multiple templates are
    retained for videos containing several instances/appearances of the same text.
    The gate is intended for recovery periods, not as a replacement for OCR.
    """

    MAX_FRAME_DIM = 720
    MAX_TEMPLATES = 8
    MIN_TEMPLATE_SIDE = 10
    # Only skip OCR when every template match is below this low threshold.
    # Increase this value only after validating false-negative rates on real videos.
    ABSENCE_THRESHOLD = 0.22
    SCALES = (0.75, 0.90, 1.00, 1.10, 1.25)
    FORCE_OCR_AFTER_CHECKS = 6

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._templates: List[np.ndarray] = []
        self._checks_since_ocr = 0
        self._last_score: Optional[float] = None
        self.skip_count = 0
        self.search_count = 0
        self.forced_count = 0

    @property
    def template_count(self) -> int:
        return len(self._templates)

    @staticmethod
    def _resize_frame(frame: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            small = cv2.resize(frame, (max(1, int(round(w * scale))),
                                       max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        return small, scale

    @staticmethod
    def _gray_feature(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        # Histogram equalization helps with modest exposure changes.
        return cv2.equalizeHist(gray)

    def add_template(self, frame: np.ndarray, roi) -> bool:
        """Store an appearance template from a confirmed target ROI."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        try:
            small, scale = self._resize_frame(frame, self.MAX_FRAME_DIM)
            h, w = small.shape[:2]
            x, y, rw, rh = [int(round(float(v) * scale)) for v in roi]
            x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
            rw = max(0, min(rw, w - x)); rh = max(0, min(rh, h - y))
            if rw < self.MIN_TEMPLATE_SIDE or rh < self.MIN_TEMPLATE_SIDE:
                return False
            crop = small[y:y + rh, x:x + rw]
            feature = self._gray_feature(crop)
            if feature.size == 0 or min(feature.shape[:2]) < self.MIN_TEMPLATE_SIDE:
                return False
            # Flat templates produce unstable normalized correlation scores.
            if float(np.std(feature)) < 4.0:
                return False
            # Avoid storing near-identical templates repeatedly.
            for old in self._templates:
                if old.shape == feature.shape:
                    score = cv2.matchTemplate(feature, old, cv2.TM_CCOEFF_NORMED)[0, 0]
                    if np.isfinite(score) and score > 0.97:
                        return True
            self._templates.append(feature.copy())
            if len(self._templates) > self.MAX_TEMPLATES:
                self._templates.pop(0)
            return True
        except (cv2.error, TypeError, ValueError, IndexError):
            return False

    def _best_match_score(self, frame: np.ndarray) -> Optional[float]:
        if not self._templates or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        small, _ = self._resize_frame(frame, self.MAX_FRAME_DIM)
        search = self._gray_feature(small)
        sh, sw = search.shape[:2]
        best = None
        for template in self._templates:
            th0, tw0 = template.shape[:2]
            for scale in self.SCALES:
                tw = max(self.MIN_TEMPLATE_SIDE, int(round(tw0 * scale)))
                th = max(self.MIN_TEMPLATE_SIDE, int(round(th0 * scale)))
                if tw >= sw or th >= sh:
                    continue
                candidate = template if (tw, th) == (tw0, th0) else cv2.resize(
                    template, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                # A near-uniform template is not reliable enough to justify skipping OCR.
                if float(np.std(candidate)) < 4.0:
                    continue
                score_map = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
                score = float(np.nanmax(score_map)) if score_map.size else float('nan')
                if np.isfinite(score) and (best is None or score > best):
                    best = score
        return best

    def should_run_ocr(self, frame: np.ndarray) -> GateDecision:
        """Return whether an expensive OCR pass should run on this frame."""
        if not self.enabled:
            self.search_count += 1
            return GateDecision(True, 'disabled')
        if not self._templates:
            self.search_count += 1
            return GateDecision(True, 'no_template')
        self._checks_since_ocr += 1
        if self._checks_since_ocr >= self.FORCE_OCR_AFTER_CHECKS:
            self.forced_count += 1
            self.search_count += 1
            self._checks_since_ocr = 0
            return GateDecision(True, 'periodic_safety_check', self._last_score)
        try:
            score = self._best_match_score(frame)
        except cv2.error:
            score = None
        self._last_score = score
        # Missing score means the visual test was inconclusive, so OCR must run.
        if score is None or score >= self.ABSENCE_THRESHOLD:
            self.search_count += 1
            reason = 'uncertain' if score is None else 'possible_match'
            return GateDecision(True, reason, score)
        self.skip_count += 1
        return GateDecision(False, 'low_global_similarity', score)

    def mark_ocr_run(self) -> None:
        """Call after OCR actually ran, whether or not OCR found the target."""
        self._checks_since_ocr = 0

    def reset_statistics(self) -> None:
        self.skip_count = 0
        self.search_count = 0
        self.forced_count = 0
