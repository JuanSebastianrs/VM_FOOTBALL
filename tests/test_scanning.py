"""
Tests del modulo de visual scanning (core/scanning).

Cubren la logica pura (matematica circular, estimador, suavizado, deteccion de
recepciones y de scanning, homografia, exporter) sin depender de GPU ni del
modelo de pose. El modelo temporal (torch) se prueba si torch esta disponible.

    python -m pytest tests/test_scanning.py -v
    python tests/test_scanning.py
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.scanning import circular as C                              # noqa: E402
from core.scanning.data_io import (                                  # noqa: E402
    BallObservation, HomographyLookup, PlayerDetection, SequenceData)
from core.scanning.orientation_estimator import OrientationEstimator  # noqa: E402
from core.scanning.orientation_smoother import OrientationSmoother    # noqa: E402
from core.scanning.reception_detector import ReceptionDetector        # noqa: E402
from core.scanning.scanning_detector import (                         # noqa: E402
    FrameOrientation, ScanningDetector)


class TestCircular(unittest.TestCase):
    def test_wrap(self):
        # 2pi -> 0 (interior); el limite +-pi es ambiguo por convencion
        self.assertAlmostEqual(C.wrap_angle(2 * math.pi), 0.0, places=5)
        self.assertAlmostEqual(C.wrap_angle(math.radians(450)), math.pi / 2, places=5)
        for x in (3 * math.pi, -3 * math.pi, 7.0, -7.0):
            self.assertLessEqual(abs(C.wrap_angle(x)), math.pi + 1e-9)

    def test_delta_shortest_path(self):
        # +170 a -170 son 20 grados, no 340
        d = C.circular_delta(math.radians(-170), math.radians(170))
        self.assertAlmostEqual(math.degrees(d), 20.0, places=4)

    def test_bin_roundtrip(self):
        for b in range(8):
            ang = C.bin_to_angle(b, 8)
            self.assertEqual(C.angle_to_bin(ang, 8), b)

    def test_image_vector_negates_y(self):
        # vector "hacia arriba" en imagen (dy negativo) => +90 deg
        self.assertAlmostEqual(C.image_vector_to_angle(0, -1), math.pi / 2, places=5)

    def test_entropy_bounds(self):
        self.assertAlmostEqual(C.orientation_entropy([0.0] * 10), 0.0, places=5)
        spread = [C.bin_to_angle(b, 8) for b in range(8)]
        self.assertAlmostEqual(C.orientation_entropy(spread), 1.0, places=5)

    def test_ema_converges(self):
        ema = C.CircularEMA(alpha=0.5)
        for _ in range(50):
            ema.update(math.radians(30))
        self.assertAlmostEqual(math.degrees(ema.value()), 30.0, places=3)

    def test_ema_no_wrap_artifact(self):
        # alternar cerca de +-180 no debe producir 0
        ema = C.CircularEMA(alpha=0.3)
        for a in [179, -179, 178, -178, 179]:
            ema.update(math.radians(a))
        self.assertGreater(abs(math.degrees(ema.value())), 150)


class TestHomography(unittest.TestCase):
    def test_identity_projection(self):
        H = np.eye(3)
        hl = HomographyLookup({1: H})
        self.assertTrue(hl.available)
        self.assertEqual(hl.project(10, 20, 1), (10.0, 20.0))

    def test_nearest_frame_fallback(self):
        hl = HomographyLookup({10: np.eye(3)})
        self.assertIsNotNone(hl.get(12))      # dentro de 5 frames
        self.assertIsNone(hl.get(100))


class TestEstimator(unittest.TestCase):
    def _pose(self, landmarks, head_vis=1.0, torso_vis=1.0):
        lm = {k: [0.0, 0.0, 0.0, 0.0] for k in [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_hip", "right_hip"]}
        lm.update(landmarks)
        return {"pose_valid": True, "landmarks": lm,
                "head_visibility": head_vis, "torso_visibility": torso_vis}

    def test_movement_fallback(self):
        est = OrientationEstimator({"movement_min_speed_px": 1.0})
        pose = {"pose_valid": False, "landmarks": {}, "head_visibility": 0,
                "torso_visibility": 0}
        r = est.estimate(1, 7, pose, np.array([0, 0, 10, 20]),
                         velocity=(5.0, 0.0), player_xy=(5, 5))
        self.assertEqual(r.theta_source, "movement")
        self.assertAlmostEqual(math.degrees(r.theta_visual_img), 0.0, places=3)

    def test_previous_fallback(self):
        est = OrientationEstimator()
        pose = {"pose_valid": False, "landmarks": {}, "head_visibility": 0,
                "torso_visibility": 0}
        r = est.estimate(1, 7, pose, np.array([0, 0, 10, 20]),
                         prev_theta=math.radians(45))
        self.assertEqual(r.theta_source, "previous")

    def test_head_priority(self):
        est = OrientationEstimator()
        # nose arriba del centro de ojos/orejas -> mira "arriba" (+90)
        pose = self._pose({
            "nose": [0.5, 0.2, 0, 0.9],
            "left_eye": [0.45, 0.4, 0, 0.9], "right_eye": [0.55, 0.4, 0, 0.9],
            "left_shoulder": [0.4, 0.6, 0, 0.9], "right_shoulder": [0.6, 0.6, 0, 0.9],
        })
        r = est.estimate(1, 7, pose, np.array([0.0, 0.0, 1.0, 1.0]),
                         player_xy=(0.5, 0.5))
        self.assertEqual(r.theta_source, "head")
        self.assertIsNotNone(r.theta_visual_img)

    def test_ball_relative_angle(self):
        est = OrientationEstimator()
        pose = {"pose_valid": False, "landmarks": {}, "head_visibility": 0,
                "torso_visibility": 0}
        r = est.estimate(1, 7, pose, np.array([0, 0, 10, 20]),
                         ball_xy=(10, 5), player_xy=(5, 5))
        self.assertAlmostEqual(math.degrees(r.ball_relative_angle), 0.0, places=3)

    def test_torso_facing_is_perpendicular_to_shoulders(self):
        # Solo hombros (linea horizontal -> eje 0). El encaramiento debe ser
        # PERPENDICULAR (no el eje). Se desambigua con run_angle (mov. hacia abajo
        # en imagen => -90). Resultado esperado: ~ -90, NUNCA 0 (el eje crudo).
        est = OrientationEstimator()
        pose = self._pose({
            "left_shoulder": [0.4, 0.6, 0, 0.9], "right_shoulder": [0.6, 0.6, 0, 0.9],
        }, head_vis=0.0, torso_vis=1.0)
        r = est.estimate(1, 7, pose, np.array([0.0, 0.0, 1.0, 1.0]),
                         velocity=(0.0, 5.0), player_xy=(0.5, 0.5))
        self.assertEqual(r.theta_source, "torso")
        self.assertAlmostEqual(math.degrees(r.theta_visual_img), -90.0, places=2)
        # el eje de hombros crudo (0 deg) se exporta aparte como diagnostico
        self.assertAlmostEqual(math.degrees(r.shoulder_angle), 0.0, places=2)

    def test_ball_relative_field_with_homography(self):
        est = OrientationEstimator()
        hl = HomographyLookup({1: np.eye(3)})    # identidad img==field
        pose = {"pose_valid": False, "landmarks": {}, "head_visibility": 0,
                "torso_visibility": 0}
        r = est.estimate(1, 7, pose, np.array([0, 0, 10, 20]),
                         ball_xy=(10, 5), player_xy=(5, 5), homography=hl)
        # identidad: ball a la derecha; en marco math de cancha (Y negada) -> 0 deg
        self.assertIsNotNone(r.ball_relative_angle_field)
        self.assertAlmostEqual(math.degrees(r.ball_relative_angle_field), 0.0, places=3)


class TestSmoother(unittest.TestCase):
    def test_reduces_jitter(self):
        sm = OrientationSmoother({"smoothing_alpha": 0.3})
        base = math.radians(40)
        outs = []
        for i in range(40):
            noisy = base + math.radians(15 * math.sin(i))
            outs.append(sm.update(1, i, noisy, confidence=0.9).theta_visual_smooth)
        var_in = np.var([15 * math.sin(i) for i in range(40)])
        var_out = np.var([math.degrees(o) - 40 for o in outs[5:]])
        self.assertLess(var_out, var_in)

    def test_coast_on_invalid(self):
        sm = OrientationSmoother()
        sm.update(1, 0, math.radians(30), confidence=0.9)
        r = sm.update(1, 1, None, confidence=0.0)      # frame invalido
        self.assertIsNotNone(r.theta_visual_smooth)


class TestScanningDetector(unittest.TestCase):
    def _series(self, angles_deg, conf=0.9, start=0):
        return [FrameOrientation(start + i, math.radians(a), conf, None)
                for i, a in enumerate(angles_deg)]

    def test_detects_turn(self):
        det = ScanningDetector({"fps": 25.0, "scan_window_seconds_before": 3.0,
                                "scan_window_seconds_after": 0.0,
                                "scan_turn_threshold_deg": 45,
                                "min_scan_duration_frames": 4,
                                "min_scan_confidence": 0.4})
        # giro sostenido de 0 a 90 grados en 10 frames, recepcion en frame 80
        angles = list(np.linspace(0, 90, 30))
        frames = self._series(angles, start=20)
        m = det.evaluate_window("V", "e0", 1, 80, frames)
        self.assertGreaterEqual(m.scan_count, 1)
        self.assertGreater(m.max_orientation_change_deg, 45)
        self.assertEqual(m.scan_label, 1)

    def test_no_scan_when_flat(self):
        det = ScanningDetector({"fps": 25.0, "scan_window_seconds_before": 3.0,
                                "scan_window_seconds_after": 0.0})
        frames = self._series([30.0] * 30, start=20)
        m = det.evaluate_window("V", "e0", 1, 80, frames)
        self.assertEqual(m.scan_count, 0)
        self.assertEqual(m.scan_label, 0)


class TestReceptionDetector(unittest.TestCase):
    def _seq(self):
        # balon pegado al track 5 en frames 1-10, luego al track 9
        players_by_frame, ball_by_frame = {}, {}
        for f in range(1, 21):
            players_by_frame[f] = [
                PlayerDetection(f, 5, np.array([0, 0, 10, 20])),
                PlayerDetection(f, 9, np.array([100, 0, 110, 20])),
            ]
            att = 5 if f <= 10 else 9
            ball_by_frame[f] = BallObservation(f, 5.0, 10.0, attached_player_id=att)
        return SequenceData("V", 25.0, list(range(1, 21)), players_by_frame,
                            ball_by_frame, {5: 0, 9: 1}, {}, HomographyLookup({}))

    def test_detects_two_receptions(self):
        det = ReceptionDetector({"min_possession_frames": 3,
                                 "use_attached_player_id": True})
        events = det.detect(self._seq())
        receivers = [e.receiver_track_id for e in events]
        self.assertIn(5, receivers)
        self.assertIn(9, receivers)


class TestExporter(unittest.TestCase):
    def test_orientation_columns(self):
        from core.scanning.exporter import Exporter, ORIENTATION_COLUMNS
        with tempfile.TemporaryDirectory() as d:
            exp = Exporter(d)
            path = exp.export_orientation(
                [{"video_id": "V", "frame_id": 1, "track_id": 1}],
                Path(d) / "o.parquet")
            import pandas as pd
            df = pd.read_parquet(path)
            self.assertEqual(list(df.columns), ORIENTATION_COLUMNS)


class TestKeypointFacade(unittest.TestCase):
    def test_result_template_has_world_and_backend(self):
        from core.scanning.keypoint_extractor import _result_template
        t = _result_template("yolo_pose")
        self.assertEqual(t["backend"], "yolo_pose")
        self.assertIn("world_landmarks", t)
        self.assertIn("nose", t["landmarks"])

    def test_injected_backend_and_empty_crop(self):
        from core.scanning.keypoint_extractor import KeypointExtractor, PoseBackend, _result_template

        class Fake(PoseBackend):
            name = "fake"
            def infer(self, crop):
                r = _result_template("fake"); r["pose_valid"] = True
                return r
        ext = KeypointExtractor(backend=Fake())
        self.assertTrue(ext.extract(np.ones((10, 10, 3), np.uint8))["pose_valid"])
        # crop vacio -> template invalido
        self.assertFalse(ext.extract(np.zeros((0, 0, 3), np.uint8))["pose_valid"])

    def test_hybrid_fallback_to_yolo_when_mediapipe_fails(self):
        from core.scanning.keypoint_extractor import KeypointExtractor, _result_template

        class _Fail:
            def infer(self, crop):
                return _result_template("mediapipe")          # pose_valid=False
        class _Ok:
            def infer(self, crop):
                r = _result_template("yolo_pose"); r["pose_valid"] = True
                return r
        # construir un hybrid sin cargar modelos reales
        ext = KeypointExtractor.__new__(KeypointExtractor)
        ext.name = "hybrid"; ext._mp_threshold = 110.0
        ext._mp = _Fail(); ext._yolo = _Ok(); ext.backend = ext._yolo
        crop = np.ones((10, 10, 3), np.uint8)
        # crop grande -> intenta mediapipe, falla, cae a yolo
        res = ext.extract(crop, crop_height=200)
        self.assertTrue(res["pose_valid"])
        self.assertEqual(res["backend_used"], "yolo_pose")
        self.assertEqual(res["backend_attempted"], "mediapipe+yolo_pose")
        # crop pequeno -> directo a yolo
        res2 = ext.extract(crop, crop_height=80)
        self.assertEqual(res2["backend_attempted"], "yolo_pose")

    def test_mediapipe_model_resolution_keeps_existing_path(self):
        from core.scanning.keypoint_extractor import _resolve_mediapipe_model
        with tempfile.NamedTemporaryFile(suffix=".task", delete=False) as f:
            p = f.name
        try:
            self.assertEqual(_resolve_mediapipe_model(p, "full"), p)
        finally:
            import os
            os.unlink(p)


class TestModel(unittest.TestCase):
    def test_forward_and_loss(self):
        try:
            import torch
            from core.scanning.models.orientation_tcn import (
                OrientationLoss, OrientationTCN, build_model)
            from core.scanning.models.orientation_dataset import FEATURE_DIM
        except ImportError:
            self.skipTest("torch no disponible")
        model = build_model("tcn", FEATURE_DIM, num_bins=8)
        x = torch.randn(4, 16, FEATURE_DIM)
        out = model(x)
        self.assertEqual(out["sincos"].shape, (4, 2))
        self.assertEqual(out["bin_logits"].shape, (4, 8))
        loss = OrientationLoss()(out, torch.zeros(4), torch.zeros(4, dtype=torch.long),
                                 torch.ones(4))
        self.assertTrue(torch.isfinite(loss["total"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
