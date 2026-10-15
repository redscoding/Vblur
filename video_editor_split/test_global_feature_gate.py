import unittest
import cv2
import numpy as np
from global_feature_gate import GlobalFeatureGate


class GlobalFeatureGateTests(unittest.TestCase):
    def make_frame(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, 'TARGET 123', (180, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)
        return frame

    def test_no_template_never_skips_ocr(self):
        gate = GlobalFeatureGate()
        decision = gate.should_run_ocr(self.make_frame())
        self.assertTrue(decision.should_run_ocr)
        self.assertEqual(decision.reason, 'no_template')

    def test_periodic_safety_check_forces_ocr(self):
        frame = self.make_frame()
        gate = GlobalFeatureGate()
        self.assertTrue(gate.add_template(frame, (170, 140, 230, 70)))
        # Make the threshold impossible to reach so the first checks are skipped.
        gate.ABSENCE_THRESHOLD = 1.1
        decisions = [gate.should_run_ocr(frame) for _ in range(gate.FORCE_OCR_AFTER_CHECKS)]
        self.assertTrue(decisions[-1].should_run_ocr)
        self.assertEqual(decisions[-1].reason, 'periodic_safety_check')

    def test_matching_frame_is_not_skipped(self):
        frame = self.make_frame()
        gate = GlobalFeatureGate()
        self.assertTrue(gate.add_template(frame, (170, 140, 230, 70)))
        decision = gate.should_run_ocr(frame)
        self.assertTrue(decision.should_run_ocr)


if __name__ == '__main__':
    unittest.main()
