"""
Tests de la fase supervisada de Scanning V2 (dataset, features, modelo, eval,
prediccion). Usan datos SINTETICOS (no GT real) para ejercitar la lógica; no se
inventan métricas sobre datos reales.

    python -m pytest tests/test_scanning_v2_supervised.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.scanning_v2.supervised.annotation_validator import validate_annotations  # noqa: E402
from core.scanning_v2.supervised.dataset_builder import DatasetBuilder      # noqa: E402
from core.scanning_v2.supervised.evaluator import ScanningEvaluator, _metrics  # noqa: E402
from core.scanning_v2.supervised.feature_extractor import FeatureExtractor  # noqa: E402
from core.scanning_v2.supervised.predictor import ScanningPredictor         # noqa: E402
from core.scanning_v2.supervised.readiness import compute_readiness         # noqa: E402
from core.scanning_v2.supervised.feature_extractor import _circular_range_deg  # noqa: E402
from core.scanning_v2.supervised.schema import (                            # noqa: E402
    FEATURE_VERSION, HEURISTIC_PRED_COLUMN, PREDICTION_COLUMNS,
    training_feature_columns)
from core.scanning_v2.supervised.trainer import ScanningTrainer             # noqa: E402
from scripts.scanning_v2.annotate_scanning_windows import _sync_master      # noqa: E402


# --------------------------------------------------------------------------
def _synth(n=30, n_pos=12, seed=0, video="V"):
    """Dataset separable: features parquet + labels parquet."""
    rng = np.random.default_rng(seed)
    rows, labs = [], []
    for i in range(n):
        pos = i < n_pos
        eid = f"{video}_rcp_{i:04d}"
        rows.append(dict(event_id=eid, video_id=video, receiver_track_id=i,
                         receiver_role="player", event_source="heuristic",
                         event_confidence=0.5, feature_version=FEATURE_VERSION,
                         f1=float(rng.normal(2 if pos else -2, 0.4)),
                         f2=float(rng.normal(1 if pos else -1, 0.4)),
                         n_frames=10,
                         heuristic_scan_label_pred=int(pos)))
        labs.append(dict(event_id=eid, video_id=video,
                         scan_label_gt=1 if pos else 0, head_turn_count_gt=None,
                         turn_direction_gt=None, visibility="high",
                         confidence="high", notes=None))
    return pd.DataFrame(rows), pd.DataFrame(labs)


def _head_pose(event_ids, frames=6):
    rows = []
    for eid in event_ids:
        for k in range(frames):
            rows.append(dict(event_id=eid, video_id="V", frame_id=100 + k, track_id=1,
                             yaw_smooth=float(k * 5), yaw_confidence_smooth=0.6,
                             head_pose_confidence=0.6,
                             head_pose_backend_used="yolo_pose_body",
                             head_crop_valid=False, head_crop_quality=0.1,
                             theta_body_field=10.0, ball_relative_angle_field=5.0,
                             distance_to_ball=2.0, time_to_reception=(frames - k) / 25.0))
    return pd.DataFrame(rows)


def _trainer(**kw):
    base = dict(type="logistic_regression", random_state=42, class_weight="balanced",
                min_labeled_samples=20, min_positive_samples=3, test_size=0.2,
                val_size=0.2, group_split_by_video=True)
    base.update(kw)
    return ScanningTrainer(base)


# --------------------------------------------------------------------------
class TestDataset(unittest.TestCase):
    def test_builder_ignores_empty_labels(self):
        fdf = pd.DataFrame({"event_id": ["e0", "e1", "e2"]})
        ann = pd.DataFrame({"event_id": ["e0", "e1", "e2"],
                            "scan_label_gt": [1, "", None]})
        out = DatasetBuilder()._labels_for(fdf, ann, "V")
        lab = pd.to_numeric(out["scan_label_gt"], errors="coerce")
        self.assertEqual(int(lab.notna().sum()), 1)     # solo e0 etiquetado
        self.assertEqual(int(lab.isna().sum()), 2)      # e1, e2 unlabeled


class TestFeatures(unittest.TestCase):
    def test_one_row_per_event(self):
        events = pd.DataFrame([
            dict(event_id="V_rcp_0000", video_id="V", receiver_track_id=1,
                 receiver_role="player", source="heuristic", event_confidence=0.5),
            dict(event_id="V_rcp_0001", video_id="V", receiver_track_id=2,
                 receiver_role="player", source="heuristic", event_confidence=0.5)])
        hp = _head_pose(["V_rcp_0000", "V_rcp_0001"])
        feats = FeatureExtractor().extract(events, hp)
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats["event_id"].nunique(), 2)
        self.assertIn("yaw_range_deg", feats.columns)

    def test_excludes_label_columns(self):
        feats, _ = _synth(5, 2)
        feats["scan_label_gt"] = [0, 1, 0, 1, 0]   # contaminacion deliberada
        cols = training_feature_columns(feats)
        self.assertNotIn("scan_label_gt", cols)
        self.assertNotIn("visibility", cols)

    def test_excludes_heuristic_pred_by_default(self):
        feats, _ = _synth(5, 2)
        cols = training_feature_columns(feats)
        self.assertNotIn(HEURISTIC_PRED_COLUMN, cols)
        cols2 = training_feature_columns(feats, include_heuristic_pred=True)
        self.assertIn(HEURISTIC_PRED_COLUMN, cols2)

    def test_excludes_training_eligible_and_merge_cols(self):
        # training_eligible deriva de la etiqueta/gates: NUNCA debe ser feature
        df = pd.DataFrame({"event_id": ["e0"], "video_id": ["V"], "f1": [1.0],
                           "training_eligible": [True], "scan_label_gt_lab": [1],
                           "sample_id": [0]})
        cols = training_feature_columns(df)
        self.assertIn("f1", cols)
        for bad in ("training_eligible", "scan_label_gt_lab", "sample_id"):
            self.assertNotIn(bad, cols)


class TestTrainer(unittest.TestCase):
    def test_split_reproducible(self):
        feats, labs = _synth(30, 12)
        r1 = _trainer().train(feats, labs)
        r2 = _trainer().train(feats, labs)
        self.assertEqual(r1["status"], "trained")
        self.assertEqual(r1["splits"]["test"].tolist(), r2["splits"]["test"].tolist())

    def test_no_train_below_min_labeled(self):
        feats, labs = _synth(10, 5)
        r = _trainer(min_labeled_samples=20).train(feats, labs)
        self.assertEqual(r["status"], "skipped")
        self.assertIn("Not enough labeled samples", r["reason"])

    def test_no_train_below_min_positive(self):
        feats, labs = _synth(25, 2)   # 25 labeled pero solo 2 positivos
        r = _trainer(min_labeled_samples=20, min_positive_samples=3).train(feats, labs)
        self.assertEqual(r["status"], "skipped")
        self.assertIn("positive", r["reason"])


class TestEvaluator(unittest.TestCase):
    def test_no_rocauc_single_class(self):
        m = _metrics([0, 0, 0, 0], [0, 0, 1, 0], proba=[0.1, 0.2, 0.9, 0.3])
        self.assertIsNone(m["roc_auc"])

    def test_reports_fp_fn(self):
        preds = pd.DataFrame({"event_id": ["e0", "e1", "e2", "e3"], "video_id": "V",
                              "scan_label_model": [0, 0, 1, 1],
                              "scan_probability_model": [0.2, 0.3, 0.8, 0.7]})
        labels = pd.DataFrame({"event_id": ["e0", "e1", "e2", "e3"], "video_id": "V",
                               "scan_label_gt": [1, 0, 1, 0]})
        res = ScanningEvaluator().evaluate(preds, labels)
        self.assertTrue(res["ground_truth_available"])
        self.assertEqual(res["false_negative_event_ids"], ["e0"])
        self.assertEqual(res["false_positive_event_ids"], ["e3"])

    def test_no_metrics_without_gt(self):
        preds = pd.DataFrame({"event_id": ["e0"], "video_id": "V",
                              "scan_label_model": [1], "scan_probability_model": [0.9]})
        labels = pd.DataFrame({"event_id": ["e0"], "video_id": "V",
                               "scan_label_gt": [np.nan]})
        res = ScanningEvaluator().evaluate(preds, labels)
        self.assertFalse(res["ground_truth_available"])


class TestPredictor(unittest.TestCase):
    def _train_and_save(self):
        feats, labs = _synth(30, 12)
        r = _trainer().train(feats, labs)
        d = Path(tempfile.mkdtemp())
        ScanningTrainer.write(r, str(d), overwrite=True)
        return feats, d / "scanning_classifier.pkl"

    def test_predict_columns(self):
        feats, model_path = self._train_and_save()
        pred = ScanningPredictor.load(str(model_path)).predict(feats)
        self.assertEqual(list(pred.columns), PREDICTION_COLUMNS)
        self.assertEqual(len(pred), len(feats))
        self.assertTrue(((pred["scan_probability_model"] >= 0) &
                         (pred["scan_probability_model"] <= 1)).all())

    def test_saved_model_loads_and_predicts(self):
        feats, model_path = self._train_and_save()
        self.assertTrue(model_path.exists())
        pred = ScanningPredictor.load(str(model_path)).predict(feats)
        self.assertIn("scan_label_model", pred.columns)

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            ScanningPredictor.load("does/not/exist/scanning_classifier.pkl")


class TestAnnotationValidator(unittest.TestCase):
    def test_detects_invalid_labels(self):
        ann = pd.DataFrame({"event_id": ["e0", "e1"], "video_id": "V",
                            "scan_label_gt": ["2", "1"],          # 2 invalido
                            "turn_direction_gt": ["sideways", "left"],  # sideways invalido
                            "visibility": ["high", "medium"],
                            "confidence": ["low", "high"]})
        rep = validate_annotations(ann, {"e0", "e1"}, "V")
        self.assertGreaterEqual(rep["n_invalid_labels"], 1)
        bad = {it["event_id"] for it in rep["invalid_labels"]}
        self.assertIn("e0", bad)

    def test_detects_duplicates(self):
        ann = pd.DataFrame({"event_id": ["e0", "e0", "e1"], "video_id": "V",
                            "scan_label_gt": ["1", "0", "1"]})
        rep = validate_annotations(ann, {"e0", "e1"}, "V")
        self.assertIn("e0", rep["duplicated_event_ids"])

    def test_detects_out_of_sync(self):
        # e9 anotado pero no existe en V2; e1 existe en V2 pero no anotado
        ann = pd.DataFrame({"event_id": ["e0", "e9"], "video_id": "V",
                            "scan_label_gt": ["1", "0"]})
        rep = validate_annotations(ann, {"e0", "e1"}, "V")
        self.assertIn("e9", rep["stale_event_ids_not_in_v2"])
        self.assertIn("e1", rep["v2_event_ids_missing_in_annotations"])
        self.assertFalse(rep["synchronized"])


class TestDatasetGates(unittest.TestCase):
    def test_match_by_event_and_video(self):
        fdf = pd.DataFrame({"event_id": ["V_rcp_0000"]})
        # mismo event_id pero OTRO video -> no debe matchear
        ann = pd.DataFrame({"event_id": ["V_rcp_0000"], "video_id": ["OTHER"],
                            "scan_label_gt": [1]})
        out = DatasetBuilder()._labels_for(fdf, ann, "V")
        self.assertTrue(pd.isna(out["scan_label_gt"].iloc[0]))
        # ahora con el video correcto
        ann2 = ann.assign(video_id=["V"])
        out2 = DatasetBuilder()._labels_for(fdf, ann2, "V")
        self.assertEqual(int(out2["scan_label_gt"].iloc[0]), 1)

    def test_respects_drop_low_visibility(self):
        fdf = pd.DataFrame({"event_id": ["a", "b"]})
        ann = pd.DataFrame({"event_id": ["a", "b"], "video_id": "V",
                            "scan_label_gt": [1, 1],
                            "visibility": ["low", "high"]})
        b = DatasetBuilder({"dataset": {"drop_low_visibility": True}})
        out = b._labels_for(fdf, ann, "V").set_index("event_id")
        self.assertFalse(bool(out.loc["a", "training_eligible"]))   # low -> excluido
        self.assertTrue(bool(out.loc["b", "training_eligible"]))

    def test_respects_min_valid_pose_ratio(self):
        fdf = pd.DataFrame({"event_id": ["a", "b"], "valid_pose_ratio": [0.8, 0.1]})
        ann = pd.DataFrame({"event_id": ["a", "b"], "video_id": "V",
                            "scan_label_gt": [1, 1]})
        b = DatasetBuilder({"dataset": {"min_valid_pose_ratio_for_training": 0.5}})
        out = b._labels_for(fdf, ann, "V").set_index("event_id")
        self.assertTrue(bool(out.loc["a", "training_eligible"]))
        self.assertFalse(bool(out.loc["b", "training_eligible"]))   # 0.1 < 0.5


class TestTrainerSplitFallback(unittest.TestCase):
    def test_single_video_no_group_split(self):
        feats, labs = _synth(30, 12, video="V")     # un solo video
        r = _trainer(group_split_by_video=True).train(feats, labs)
        self.assertEqual(r["status"], "trained")
        self.assertIn("fallback", r["metadata"]["split_method"])


class TestMultiVideoSafety(unittest.TestCase):
    def test_master_dedup_by_video_and_event(self):
        master = Path(tempfile.mkdtemp()) / "gt.csv"
        # video A ya tiene e0; sincronizamos video B que TAMBIEN tiene e0
        from scripts.scanning_v2.annotate_scanning_windows import GT_COLUMNS
        a = pd.DataFrame([{c: "" for c in GT_COLUMNS}])
        a.loc[0, ["event_id", "video_id", "scan_label_gt"]] = ["e0", "A", 1]
        a.to_csv(master, index=False)
        dfb = pd.DataFrame([{c: "" for c in GT_COLUMNS}])
        dfb.loc[0, ["event_id", "video_id"]] = ["e0", "B"]
        _sync_master(master, dfb, "B")
        out = pd.read_csv(master)
        # ambos (A,e0) y (B,e0) deben sobrevivir: clave es (video_id, event_id)
        keys = set(zip(out["video_id"].astype(str), out["event_id"].astype(str)))
        self.assertIn(("A", "e0"), keys)
        self.assertIn(("B", "e0"), keys)

    def test_trainer_merge_by_video_and_event(self):
        feats = pd.DataFrame([
            dict(event_id="e0", video_id="A", f1=1.0, feature_version=FEATURE_VERSION),
            dict(event_id="e0", video_id="B", f1=2.0, feature_version=FEATURE_VERSION)])
        labs = pd.DataFrame([
            dict(event_id="e0", video_id="A", scan_label_gt=1, training_eligible=True),
            dict(event_id="e0", video_id="B", scan_label_gt=0, training_eligible=True)])
        merged = _trainer()._merge(feats, labs)
        byv = merged.set_index("video_id")["scan_label_gt"].to_dict()
        self.assertEqual(int(byv["A"]), 1)   # no se cruzan labels entre videos
        self.assertEqual(int(byv["B"]), 0)

    def test_evaluator_merge_by_video_and_event(self):
        preds = pd.DataFrame([
            dict(event_id="e0", video_id="A", scan_label_model=1, scan_probability_model=0.9),
            dict(event_id="e0", video_id="B", scan_label_model=1, scan_probability_model=0.8)])
        labels = pd.DataFrame([
            dict(event_id="e0", video_id="A", scan_label_gt=1),
            dict(event_id="e0", video_id="B", scan_label_gt=0)])
        res = ScanningEvaluator().evaluate(preds, labels)
        self.assertEqual(res["model"]["n"], 2)
        # (B,e0) es FP; (A,e0) es TP -> no se mezclan
        fp_keys = {(r["video_id"], r["event_id"]) for r in res["false_positives"]}
        self.assertIn(("B", "e0"), fp_keys)
        self.assertNotIn(("A", "e0"), fp_keys)

    def test_conflicting_duplicates_fail(self):
        fdf = pd.DataFrame({"event_id": ["e0"]})
        ann = pd.DataFrame({"event_id": ["e0", "e0"], "video_id": "V",
                            "scan_label_gt": [1, 0]})   # conflicto
        with self.assertRaises(ValueError):
            DatasetBuilder()._labels_for(fdf, ann, "V")


class TestCircularRange(unittest.TestCase):
    def test_range_handles_wrap(self):
        # +179 y -179 estan a ~2 grados, NO a 358
        r = _circular_range_deg([179.0, -179.0])
        self.assertLess(r, 10.0)
        # rango amplio real
        self.assertGreater(_circular_range_deg([0.0, 90.0, 170.0]), 150.0)

    def test_feature_yaw_range_circular(self):
        events = pd.DataFrame([dict(event_id="e0", video_id="V", receiver_track_id=1,
                                    receiver_role="player", source="heuristic",
                                    event_confidence=0.5)])
        hp = pd.DataFrame([
            dict(event_id="e0", video_id="V", frame_id=1, yaw_smooth=179.0,
                 yaw_confidence_smooth=0.6, head_pose_confidence=0.6,
                 head_pose_backend_used="yolo_pose_body", head_crop_valid=False,
                 head_crop_quality=0.1, theta_body_field=None,
                 ball_relative_angle_field=None, distance_to_ball=2.0, time_to_reception=0.1),
            dict(event_id="e0", video_id="V", frame_id=2, yaw_smooth=-179.0,
                 yaw_confidence_smooth=0.6, head_pose_confidence=0.6,
                 head_pose_backend_used="yolo_pose_body", head_crop_valid=False,
                 head_crop_quality=0.1, theta_body_field=None,
                 ball_relative_angle_field=None, distance_to_ball=2.0, time_to_reception=0.05)])
        feats = FeatureExtractor().extract(events, hp)
        self.assertLess(float(feats["yaw_range_deg"].iloc[0]), 10.0)


class TestPredictorVersion(unittest.TestCase):
    def _model(self):
        feats, labs = _synth(30, 12)
        r = _trainer().train(feats, labs)
        d = Path(tempfile.mkdtemp())
        ScanningTrainer.write(r, str(d), overwrite=True)
        return feats, str(d / "scanning_classifier.pkl")

    def test_feature_version_mismatch_fails(self):
        feats, mp = self._model()
        bad = feats.copy()
        bad["feature_version"] = "OTHER_VERSION"
        with self.assertRaises(ValueError):
            ScanningPredictor.load(mp).predict(bad)
        # con flag explicito, no falla
        ScanningPredictor.load(mp, allow_feature_version_mismatch=True).predict(bad)

    def test_missing_feature_fails(self):
        feats, mp = self._model()
        pred = ScanningPredictor.load(mp)
        missing = feats.drop(columns=[pred.feature_columns[0]])
        with self.assertRaises(ValueError):
            pred.predict(missing)


class TestClipValidation(unittest.TestCase):
    def test_uses_annotation_pack_clip_path(self):
        pack = Path(tempfile.mkdtemp())
        (pack / "clips").mkdir()
        (pack / "clips" / "e0_video.mp4").write_bytes(b"x")   # clip de e0 existe
        ann = pd.DataFrame({"event_id": ["e0", "e1"], "video_id": "V",
                            "scan_label_gt": [1, 0],
                            "clip_path": ["clips/e0_video.mp4", ""]})
        rep = validate_annotations(ann, {"e0", "e1"}, "V",
                                   annotation_pack_dir=str(pack))
        self.assertIn("e1", rep["missing_clip_path"])      # clip_path vacio
        self.assertIn("e1", rep["missing_clip_file"])      # fallback tampoco existe
        self.assertNotIn("e0", rep["missing_clip_file"])


class TestReadiness(unittest.TestCase):
    def _write_manifest(self, sup, **kw):
        import json
        (sup / "dataset").mkdir(parents=True, exist_ok=True)
        (sup / "dataset" / "dataset_manifest.json").write_text(json.dumps(kw))

    def test_uses_trainable_positives(self):
        d = Path(tempfile.mkdtemp())
        v2 = d / "v2"; v2.mkdir()
        ev = pd.DataFrame({"event_id": [f"V_rcp_{i:04d}" for i in range(25)],
                           "video_id": "V", "frame_reception": range(25),
                           "receiver_track_id": range(25)})
        ev.to_parquet(v2 / "pass_reception_events.parquet", index=False)
        # annotations: 25 etiquetadas, 5 positivas (CSV)
        ann = pd.DataFrame({"event_id": [f"V_rcp_{i:04d}" for i in range(25)],
                            "video_id": "V",
                            "scan_label_gt": [1] * 5 + [0] * 20})
        apath = d / "gt.csv"; ann.to_csv(apath, index=False)
        sup = d / "sup"
        # manifest dice que solo 2 positivos son ELEGIBLES
        self._write_manifest(sup, n_trainable=20, n_trainable_positive=2,
                             n_trainable_negative=18, n_excluded_low_visibility=3)
        r = compute_readiness("V", str(v2), str(sup), str(apath),
                              min_labeled=20, min_positive=3)
        self.assertEqual(r["positive_trainable"], 2)       # del manifest, no 5
        self.assertEqual(r["positives_needed"], 1)         # 3 - 2
        self.assertFalse(r["can_train"])

    def test_can_train_false_if_out_of_sync(self):
        d = Path(tempfile.mkdtemp())
        v2 = d / "v2"; v2.mkdir()
        ev = pd.DataFrame({"event_id": ["V_rcp_0000"], "video_id": "V",
                           "frame_reception": [10], "receiver_track_id": [1]})
        ev.to_parquet(v2 / "pass_reception_events.parquet", index=False)
        # annotacion con un evento OBSOLETO (no existe en V2) -> desincronizado
        ann = pd.DataFrame({"event_id": ["V_rcp_0000", "V_rcp_9999"], "video_id": "V",
                            "scan_label_gt": [1, 0]})
        apath = d / "gt.csv"; ann.to_csv(apath, index=False)
        sup = d / "sup"
        self._write_manifest(sup, n_trainable=20, n_trainable_positive=10,
                             n_trainable_negative=10)
        r = compute_readiness("V", str(v2), str(sup), str(apath),
                              min_labeled=2, min_positive=1)
        self.assertFalse(r["annotations_synchronized"])
        self.assertFalse(r["can_train"])     # aunque haya labels, desync => no entrena

    def test_computes_labels_needed(self):
        d = Path(tempfile.mkdtemp())
        v2 = d / "v2"; v2.mkdir()
        events = pd.DataFrame({"event_id": ["V_rcp_0000", "V_rcp_0001"],
                               "video_id": "V", "frame_reception": [10, 20],
                               "receiver_track_id": [1, 2]})
        events.to_parquet(v2 / "pass_reception_events.parquet", index=False)
        sup = d / "sup"; sup.mkdir()
        r = compute_readiness("V", str(v2), str(sup), annotations_path=None,
                              min_labeled=4, min_positive=2)
        self.assertEqual(r["labels_needed"], 4)      # 0 labeled, min 4
        self.assertEqual(r["positives_needed"], 2)
        self.assertFalse(r["can_train"])
        self.assertIn("V_rcp_0000", r["annotation"]["v2_event_ids_missing_in_annotations"])


class TestModelZoo(unittest.TestCase):
    """Las arquitecturas nuevas de la fabrica entrenan y predicen probas."""

    def _fit_and_check(self, mtype, extra=None):
        from core.scanning_v2.supervised.models import build_model
        feats, labs = _synth(60, 30, seed=3)
        cols = training_feature_columns(feats)
        cfg = {"type": mtype, "random_state": 42}
        cfg.update(extra or {})
        model, params = build_model(cfg)
        model.fit(feats[cols], labs["scan_label_gt"])
        proba = model.predict_proba(feats[cols])[:, 1]
        self.assertEqual(len(proba), 60)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertEqual(params["type"], mtype)
        # dataset separable -> debe superar por mucho el azar
        pred = (proba >= 0.5).astype(int)
        acc = float((pred == labs["scan_label_gt"].values).mean())
        self.assertGreater(acc, 0.8, f"{mtype} accuracy={acc}")

    def test_hist_gradient_boosting(self):
        self._fit_and_check("hist_gradient_boosting")

    def test_mlp(self):
        self._fit_and_check("mlp")

    def test_unsupported_type_raises(self):
        from core.scanning_v2.supervised.models import build_model
        with self.assertRaises(ValueError):
            build_model({"type": "transformer_xxl"})


class TestSequenceModel(unittest.TestCase):
    """GRU temporal: aprende un patron secuencial trivial y es clonable."""

    @staticmethod
    def _seq_dataset(n=80, T=16, seed=0):
        rng = np.random.default_rng(seed)
        rows, y = [], []
        for i in range(n):
            pos = i < n // 2
            # positivos: yaw oscila (scanning); negativos: yaw constante
            if pos:
                ang = np.cumsum(rng.choice([-0.6, 0.6], size=T))
            else:
                ang = np.full(T, rng.normal(0, 0.05)) + rng.normal(0, 0.02, T)
            row = {"event_id": f"e{i}", "video_id": "V", "aux_feature": float(pos)
                   * 0.0}
            for k in range(T):
                row[f"seq_cos_{k:02d}"] = float(np.cos(ang[k]))
                row[f"seq_sin_{k:02d}"] = float(np.sin(ang[k]))
                row[f"seq_valid_{k:02d}"] = 1
            rows.append(row)
            y.append(int(pos))
        return pd.DataFrame(rows), np.array(y)

    def test_gru_learns_oscillation(self):
        from core.scanning_v2.supervised.models import build_model
        X, y = self._seq_dataset()
        cols = training_feature_columns(X)
        model, params = build_model({"type": "gru_sequence", "random_state": 42,
                                     "max_epochs": 60, "patience": 15,
                                     "device": "cpu"})
        model.fit(X[cols], y)
        proba = model.predict_proba(X[cols])[:, 1]
        acc = float(((proba >= 0.5).astype(int) == y).mean())
        self.assertGreater(acc, 0.85, f"gru accuracy={acc}")

    def test_gru_is_joblib_picklable(self):
        import joblib
        from core.scanning_v2.supervised.models import build_model
        X, y = self._seq_dataset(n=30, T=8, seed=1)
        cols = training_feature_columns(X)
        model, _ = build_model({"type": "gru_sequence", "random_state": 42,
                                "max_epochs": 5, "patience": 3, "device": "cpu"})
        model.fit(X[cols], y)
        p = Path(tempfile.mkdtemp()) / "gru.pkl"
        joblib.dump(model, p)
        loaded = joblib.load(p)
        np.testing.assert_allclose(loaded.predict_proba(X[cols]),
                                   model.predict_proba(X[cols]), rtol=1e-5)

    def test_gru_requires_sequence_columns(self):
        from core.scanning_v2.supervised.models import build_model
        feats, labs = _synth(30, 12)
        cols = training_feature_columns(feats)
        model, _ = build_model({"type": "gru_sequence", "device": "cpu",
                                "max_epochs": 2})
        with self.assertRaises(ValueError):
            model.fit(feats[cols], labs["scan_label_gt"])

    def test_sequence_features_in_extractor(self):
        events = pd.DataFrame([
            dict(event_id="V_rcp_0000", video_id="V", receiver_track_id=1,
                 receiver_role="player", source="heuristic", event_confidence=0.5)])
        hp = _head_pose(["V_rcp_0000"], frames=10)
        fe = FeatureExtractor({"include_sequence_features": True,
                               "sequence_length": 8, "sequence_seconds": 3.0})
        feats = fe.extract(events, hp)
        for k in range(8):
            self.assertIn(f"seq_cos_{k:02d}", feats.columns)
            self.assertIn(f"seq_valid_{k:02d}", feats.columns)
        # los primeros pasos (lejos de la recepcion) no tienen pose -> invalidos
        self.assertEqual(int(feats["seq_valid_00"].iloc[0]), 0)
        # el ultimo paso (t=0) si tiene pose valida
        self.assertEqual(int(feats["seq_valid_07"].iloc[0]), 1)


class TestWeakLabeler(unittest.TestCase):
    @staticmethod
    def _row(**kw):
        base = dict(event_id="V_rcp_0000", video_id="V", receiver_track_id=1,
                    valid_pose_ratio=0.9, yaw_valid_count=40, n_frames=75,
                    sustained_turn_count_20deg=0, sustained_turn_count_30deg=0,
                    sustained_turn_count_40deg=0, yaw_range_deg=10.0,
                    yaw_mean_abs_delta_deg=1.0, yaw_num_direction_changes=0)
        base.update(kw)
        return pd.Series(base)

    def test_positive_rule(self):
        from core.scanning_v2.supervised.weak_labeler import weak_label_row
        lab, conf, why = weak_label_row(self._row(
            sustained_turn_count_30deg=3, yaw_range_deg=120.0))
        self.assertEqual(lab, 1)
        self.assertGreaterEqual(conf, 0.6)

    def test_negative_rule(self):
        from core.scanning_v2.supervised.weak_labeler import weak_label_row
        lab, _, _ = weak_label_row(self._row())
        self.assertEqual(lab, 0)

    def test_gray_zone_unlabeled(self):
        from core.scanning_v2.supervised.weak_labeler import weak_label_row
        lab, _, why = weak_label_row(self._row(
            sustained_turn_count_20deg=1, yaw_range_deg=40.0))
        self.assertIsNone(lab)
        self.assertEqual(why, "gray_zone")

    def test_low_quality_unlabeled(self):
        from core.scanning_v2.supervised.weak_labeler import weak_label_row
        lab, _, why = weak_label_row(self._row(valid_pose_ratio=0.2))
        self.assertIsNone(lab)
        self.assertEqual(why, "low_quality_window")

    def test_human_labels_excluded(self):
        from core.scanning_v2.supervised.weak_labeler import generate_weak_labels
        feats = pd.DataFrame([self._row().to_dict(),
                              self._row(event_id="V_rcp_0001").to_dict()])
        human = pd.DataFrame({"event_id": ["V_rcp_0000"], "video_id": ["V"],
                              "scan_label_gt": [1]})
        out = generate_weak_labels(feats, human)
        self.assertNotIn("V_rcp_0000", out["event_id"].tolist())
        self.assertIn("V_rcp_0001", out["event_id"].tolist())
        self.assertTrue((out["label_source"] == "weak_rules_v1").all())

    def test_refuses_overwriting_non_weak_file(self):
        from core.scanning_v2.supervised.weak_labeler import (generate_weak_labels,
                                                              write_weak_labels)
        d = Path(tempfile.mkdtemp())
        p = d / "weak.csv"
        pd.DataFrame({"event_id": ["x"], "label_source": ["human"]}).to_csv(
            p, index=False)
        feats = pd.DataFrame([self._row().to_dict()])
        out = generate_weak_labels(feats, None)
        with self.assertRaises(ValueError):
            write_weak_labels(out, str(p))


if __name__ == "__main__":
    unittest.main(verbosity=2)
