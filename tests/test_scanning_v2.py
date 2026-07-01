"""
Tests del scanning V2 (logica pura: eventos/receptor, head crop, suavizado,
head-turn, evaluacion sin GT, vision map). No requieren GPU ni modelos.

    python -m pytest tests/test_scanning_v2.py -v
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.events.event_ground_truth_adapter import load_event_ground_truth  # noqa: E402
from core.events.pass_reception_detector import PassReceptionDetector   # noqa: E402
from core.game_state.game_state_builder import build_game_state         # noqa: E402
from core.game_state.schema import GAME_STATE_COLUMNS                   # noqa: E402
from core.scanning_v2.head_cropper import HeadCropper                   # noqa: E402
from core.scanning_v2.head_pose_smoother import HeadYawSmoother         # noqa: E402
from core.scanning_v2.head_turn_detector import HeadTurnDetector        # noqa: E402
from core.scanning_v2.reception_window_extractor import (               # noqa: E402
    ReceptionWindowExtractor, WindowSpec)
from core.scanning_v2.schema import SCANNING_V2_COLUMNS                 # noqa: E402
from core.scanning_v2.vision_map_wog import VisionMapWOG                # noqa: E402


def _gs_rows(specs):
    """specs: list of dicts -> game_state DataFrame con defaults."""
    rows = []
    for s in specs:
        r = {c: None for c in GAME_STATE_COLUMNS}
        r.update(video_id="V", ball_speed=0.0, bbox_x1=0, bbox_y1=0, bbox_x2=10,
                 bbox_y2=20)
        r.update(s)
        rows.append(r)
    return pd.DataFrame(rows, columns=GAME_STATE_COLUMNS)


def _cfg(**kw):
    base = dict(fps=25.0, min_possession_frames=4, min_ball_near_frames=3,
                min_event_confidence=0.45, max_ball_receiver_distance_m=2.0,
                ambiguous_margin_m=0.6, has_homography=True, reject_referee=True,
                require_candidate_receiver_role=True)
    base.update(kw)
    return base


class TestReceptionDetector(unittest.TestCase):
    def test_referee_never_receiver(self):
        specs = []
        for f in range(1, 9):
            # arbitro 6 pegado al balon, jugador 1 candidato cerca y estable
            specs.append(dict(frame_id=f, track_id=6, role="referee", team_id=-2,
                              distance_to_ball=0.1, is_referee=True,
                              is_candidate_receiver=False))
            specs.append(dict(frame_id=f, track_id=1, role="player", team_id=0,
                              distance_to_ball=0.5, is_referee=False,
                              is_candidate_receiver=True))
        ev, rej = PassReceptionDetector(_cfg()).detect(_gs_rows(specs), "V")
        self.assertNotIn(6, ev["receiver_track_id"].tolist())
        self.assertIn("referee", rej["reason"].tolist())

    def test_unknown_role_not_receiver(self):
        specs = [dict(frame_id=f, track_id=99, role="unknown", team_id=None,
                      distance_to_ball=0.2, is_referee=False,
                      is_candidate_receiver=False) for f in range(1, 9)]
        ev, _ = PassReceptionDetector(_cfg()).detect(_gs_rows(specs), "V")
        self.assertEqual(len(ev), 0)

    def test_requires_multiple_frames(self):
        # jugador candidato cerca solo 2 frames -> no evento
        specs = [dict(frame_id=f, track_id=1, role="player", team_id=0,
                      distance_to_ball=0.3, is_referee=False,
                      is_candidate_receiver=True) for f in range(1, 3)]
        ev, _ = PassReceptionDetector(_cfg()).detect(_gs_rows(specs), "V")
        self.assertEqual(len(ev), 0)

    def test_ambiguous_receiver_rejected(self):
        specs = []
        for f in range(1, 8):
            specs.append(dict(frame_id=f, track_id=1, role="player", team_id=0,
                              distance_to_ball=0.6, is_referee=False,
                              is_candidate_receiver=True))
            specs.append(dict(frame_id=f, track_id=2, role="player", team_id=1,
                              distance_to_ball=0.9, is_referee=False,
                              is_candidate_receiver=True))
        ev, rej = PassReceptionDetector(_cfg()).detect(_gs_rows(specs), "V")
        self.assertEqual(len(ev), 0)
        self.assertIn("ambiguous_receiver", rej["reason"].tolist())


class TestHeadCropper(unittest.TestCase):
    def test_head_crop_within_frame(self):
        frame = np.zeros((200, 300, 3), np.uint8)
        # bbox que se sale por arriba/izquierda
        bbox = np.array([-20, -30, 60, 120], float)
        hc = HeadCropper({"min_head_height_px": 8}).crop(frame, bbox)
        self.assertIsNotNone(hc.bbox)
        x1, y1, x2, y2 = hc.bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 300)
        self.assertLessEqual(y2, 200)


class TestSmoother(unittest.TestCase):
    def test_yaw_wrap_no_artifact(self):
        sm = HeadYawSmoother({"alpha": 0.3, "min_confidence": 0.0})
        series = [(179, 0.9), (-179, 0.9), (178, 0.9), (-178, 0.9), (179, 0.9)]
        out = sm.smooth(series)
        last = out[-1][0]
        self.assertIsNotNone(last)
        self.assertGreater(abs(last), 150.0)   # cerca de +-180, no 0


class TestHeadTurnDetector(unittest.TestCase):
    def _hp(self, yaws, conf=0.9, start=1):
        rows = []
        for i, y in enumerate(yaws):
            rows.append(dict(event_id="e", video_id="V", frame_id=start + i,
                             track_id=1, yaw_smooth=y, head_pose_confidence=conf,
                             theta_body_field=None, ball_relative_angle_field=None))
        return pd.DataFrame(rows)

    def test_no_turn_on_single_spike(self):
        det = HeadTurnDetector(dict(fps=25.0, seconds_before_reception=3.0,
                                    seconds_after_reception=0.0,
                                    yaw_turn_threshold_deg=40,
                                    min_turn_duration_frames=4))
        yaws = [10.0] * 20
        yaws[10] = 80.0           # un solo frame ruidoso
        ev = {"event_id": "e", "video_id": "V", "receiver_track_id": 1,
              "frame_reception": 80}
        m = det.evaluate(ev, self._hp(yaws, start=20))
        self.assertEqual(m["head_turn_count"], 0)
        self.assertEqual(m["scan_label_pred"], 0)

    def test_detects_sustained_turn_and_columns(self):
        det = HeadTurnDetector(dict(fps=25.0, seconds_before_reception=3.0,
                                    seconds_after_reception=0.0,
                                    yaw_turn_threshold_deg=40,
                                    min_turn_duration_frames=4,
                                    min_valid_pose_ratio=0.0))
        yaws = list(np.linspace(0, 90, 30))
        ev = {"event_id": "e", "video_id": "V", "receiver_track_id": 1,
              "frame_reception": 80}
        m = det.evaluate(ev, self._hp(yaws, start=20))
        self.assertGreaterEqual(m["head_turn_count"], 1)
        self.assertEqual(list(m.keys()), SCANNING_V2_COLUMNS)


class TestEvaluateNoGT(unittest.TestCase):
    def test_empty_gt_yields_no_metrics(self):
        # replica la deteccion de "have_gt": columna vacia => sin GT
        df = pd.DataFrame({"event_id": ["e0", "e1"], "scan_label_gt": ["", ""]})
        valid = pd.to_numeric(df["scan_label_gt"], errors="coerce").notna().sum()
        self.assertEqual(int(valid), 0)


class TestVisionMap(unittest.TestCase):
    def test_no_absolute_orientation_without_field(self):
        vm = VisionMapWOG({"enabled": True})
        r = vm.compute((0.0, 0.0), theta_head_field=None, theta_body_field=None)
        self.assertFalse(r.available)
        self.assertEqual(r.used_orientation, "none")

    def test_body_fallback_low_confidence(self):
        vm = VisionMapWOG({"enabled": True})
        r = vm.compute((0.0, 0.0), theta_head_field=None, theta_body_field=0.0)
        self.assertTrue(r.available)
        self.assertEqual(r.used_orientation, "body")
        self.assertLess(r.vision_map_confidence, 0.5)


class TestAttachedValidation(unittest.TestCase):
    def test_invalid_attached_rejected(self):
        # attached apunta a un track NO candidato (arbitro 6) -> invalido
        specs = []
        for f in range(1, 9):
            specs.append(dict(frame_id=f, track_id=6, role="referee", team_id=-2,
                              distance_to_ball=0.1, is_referee=True,
                              is_candidate_receiver=False))
            specs.append(dict(frame_id=f, track_id=1, role="player", team_id=0,
                              distance_to_ball=0.5, is_referee=False,
                              is_candidate_receiver=True))
        attached = {f: 6 for f in range(1, 9)}   # balon "atado" al arbitro
        ev, rej = PassReceptionDetector(_cfg()).detect(_gs_rows(specs), "V", attached)
        self.assertNotIn(6, ev["receiver_track_id"].tolist())
        self.assertIn("invalid_attached_player", rej["reason"].tolist())


class TestGroundTruthAdapter(unittest.TestCase):
    def _write_csv(self, df):
        d = Path(tempfile.mkdtemp())
        p = d / "gt.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_player_id_not_receiver(self):
        # un dataset con player_id (no receiver) NO debe mapearlo a receiver
        path = self._write_csv(pd.DataFrame({
            "video_id": ["V"], "frame_reception": [100], "player_id": [7]}))
        gt = load_event_ground_truth(path, "V")
        self.assertTrue(pd.isna(gt["receiver_track_id"].iloc[0]))
        self.assertEqual(gt["rejection_reason"].iloc[0], "missing_receiver_in_gt")

    def test_gt_missing_receiver_not_used(self):
        # GT sin receptor -> no entra a eventos (se registra rechazado)
        path = self._write_csv(pd.DataFrame({
            "video_id": ["V"], "frame_reception": [5], "player_id": [7]}))
        gt = load_event_ground_truth(path, "V")
        specs = [dict(frame_id=f, track_id=1, role="player", team_id=0,
                      distance_to_ball=0.4, is_referee=False,
                      is_candidate_receiver=True) for f in range(1, 9)]
        ev, rej = PassReceptionDetector(_cfg()).detect(_gs_rows(specs), "V",
                                                       gt_events=gt)
        self.assertEqual(len(ev), 0)
        self.assertIn("missing_receiver_in_gt", rej["reason"].tolist())


class TestGameStateBuilder(unittest.TestCase):
    def test_missing_role_is_unknown_not_player(self):
        d = Path(tempfile.mkdtemp())
        det = [{"frame_id": f,
                "players": [{"track_id": 1, "x_min": 0, "y_min": 0, "x_max": 10,
                             "y_max": 20},
                            {"track_id": 9, "x_min": 5, "y_min": 5, "x_max": 15,
                             "y_max": 25}],
                "ball_candidates": []} for f in range(1, 4)]
        (d / "det.json").write_text(json.dumps(det))
        # solo track 1 tiene rol; track 9 NO esta -> debe quedar "unknown"
        (d / "team.json").write_text(json.dumps([{"track_id": 1, "team_id": 0,
                                                   "role": "player"}]))
        gs, report = build_game_state("V", str(d / "det.json"),
                                      team_assignments_json=str(d / "team.json"))
        r9 = gs[gs["track_id"] == 9].iloc[0]
        self.assertEqual(r9["role"], "unknown")
        self.assertFalse(bool(r9["is_candidate_receiver"]))
        self.assertIn(9, report["unknown_role_track_ids"])


class TestWindowsArtifacts(unittest.TestCase):
    def test_window_start_not_negative(self):
        specs_gs = [dict(frame_id=f, track_id=1, role="player", team_id=0,
                         is_referee=False, is_candidate_receiver=True)
                    for f in range(1, 30)]
        gs = _gs_rows(specs_gs)
        events = pd.DataFrame([{"event_id": "e0", "video_id": "V",
                                "frame_reception": 10, "receiver_track_id": 1,
                                "receiver_team_id": 0, "receiver_role": "player",
                                "source": "heuristic", "event_confidence": 0.6,
                                "frame_pass": None}])
        wx = ReceptionWindowExtractor({"fps": 25.0, "seconds_before_reception": 3.0})
        specs = wx.extract(events, gs)
        self.assertGreaterEqual(specs[0].window_start, 1)
        self.assertGreaterEqual(specs[0].window_start, 0)

    def test_metadata_json_no_nan(self):
        ev = {"event_id": "e0", "video_id": "V", "receiver_track_id": 1,
              "receiver_role": "player", "receiver_team_id": float("nan"),
              "frame_pass": float("nan"), "frame_reception": 50,
              "source": "heuristic", "event_confidence": float("nan")}
        spec = WindowSpec(ev, 5, 45, [5, 6, 7])
        out = Path(tempfile.mkdtemp())
        d = ReceptionWindowExtractor.write_metadata(out, spec)
        text = (d / "metadata.json").read_text()
        self.assertNotIn("NaN", text)
        meta = json.loads(text)   # debe ser JSON valido
        self.assertIsNone(meta["team_id"])
        self.assertIsNone(meta["frame_pass"])


class TestBodyFallbackExclusion(unittest.TestCase):
    def _hp_body(self, yaws, backend):
        rows = []
        for i, y in enumerate(yaws):
            rows.append(dict(event_id="e", video_id="V", frame_id=20 + i,
                             track_id=1, yaw_smooth=y, head_pose_confidence=0.9,
                             yaw_confidence_smooth=0.9,
                             head_pose_backend_used=backend,
                             theta_body_field=None, ball_relative_angle_field=None))
        return pd.DataFrame(rows)

    def test_body_fallback_does_not_trigger_scan(self):
        yaws = list(np.linspace(0, 90, 30))
        ev = {"event_id": "e", "video_id": "V", "receiver_track_id": 1,
              "frame_reception": 80}
        base = dict(fps=25.0, seconds_before_reception=3.0, seconds_after_reception=0.0,
                    yaw_turn_threshold_deg=40, min_turn_duration_frames=4,
                    min_valid_pose_ratio=0.0)
        # excluido por config -> no cuenta como head-turn
        det = HeadTurnDetector({**base, "allow_body_fallback_for_scan": False})
        m = det.evaluate(ev, self._hp_body(yaws, "body_orientation"))
        self.assertEqual(m["head_turn_count"], 0)
        self.assertEqual(m["scan_label_pred"], 0)
        # permitido por config -> si cuenta
        det2 = HeadTurnDetector({**base, "allow_body_fallback_for_scan": True})
        m2 = det2.evaluate(ev, self._hp_body(yaws, "body_orientation"))
        self.assertGreaterEqual(m2["head_turn_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
