"""Adaptive OCR scheduling policy for video_processor.py.

This module decides when to run OCR and whether to use a full-frame or ROI scan.
It deliberately does not decide whether a detection is a valid target; that remains
video_processor.py's responsibility.
"""
from dataclasses import dataclass


@dataclass
class GateDecision:
    run_ocr: bool
    mode: str  # "full" or "roi"
    reason: str
    interval: int = 0


class GlobalFeatureGate:
    """Adaptive schedule balancing OCR cost and rediscovery safety."""

    def __init__(self,
                 no_target_min_interval=3,
                 no_target_max_interval=12,
                 stable_roi_interval=12,
                 uncertain_full_interval=2,
                 full_frame_rescan_interval=30):
        self.no_target_min_interval = max(1, int(no_target_min_interval))
        self.no_target_max_interval = max(self.no_target_min_interval, int(no_target_max_interval))
        self.stable_roi_interval = max(1, int(stable_roi_interval))
        self.uncertain_full_interval = max(1, int(uncertain_full_interval))
        self.full_frame_rescan_interval = max(1, int(full_frame_rescan_interval))
        self.no_target_misses = 0
        self.last_full_scan_frame = None
        self.last_ocr_frame = None

    def reset_search(self):
        self.no_target_misses = 0

    def search_interval(self):
        """Back off after repeated misses, but never exceed the configured ceiling."""
        interval = self.no_target_min_interval * (2 ** min(self.no_target_misses, 8))
        return min(self.no_target_max_interval, interval)

    def report_search_result(self, found):
        if found:
            self.reset_search()
        else:
            self.no_target_misses += 1

    def note_full_scan(self, frame_no):
        self.last_full_scan_frame = int(frame_no)
        self.last_ocr_frame = int(frame_no)

    def decide_tracking(self, frame_no, active_tracks, has_uncertain_track=False):
        """Choose full-frame OCR for recovery/rescans, ROI OCR for stable tracks."""
        frame_no = int(frame_no)
        if not active_tracks:
            return GateDecision(True, "full", "no active tracks; search whole frame", 1)

        if has_uncertain_track:
            last = self.last_ocr_frame
            if last is None or frame_no - last >= self.uncertain_full_interval:
                self.last_ocr_frame = frame_no
                return GateDecision(True, "full", "track uncertain; recover with full-frame OCR",
                                    self.uncertain_full_interval)
            return GateDecision(False, "full", "waiting for next recovery OCR", self.uncertain_full_interval)

        last_full = self.last_full_scan_frame
        if last_full is None or frame_no - last_full >= self.full_frame_rescan_interval:
            self.note_full_scan(frame_no)
            return GateDecision(True, "full", "periodic full-frame rescan", self.full_frame_rescan_interval)

        last_ocr = self.last_ocr_frame
        if last_ocr is None or frame_no - last_ocr >= self.stable_roi_interval:
            self.last_ocr_frame = frame_no
            return GateDecision(True, "roi", "stable tracks; OCR only tracked regions", self.stable_roi_interval)

        return GateDecision(False, "roi", "stable tracking; skip redundant OCR", self.stable_roi_interval)
