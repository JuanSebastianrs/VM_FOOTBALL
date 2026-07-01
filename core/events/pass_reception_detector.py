# core/events/pass_reception_detector.py
"""
FASE 2 — Detector de pase/recepcion (V2), conservador.

Prioriza ground truth; si no hay, usa una heuristica con MULTIPLES condiciones
(no solo "mas cercano al balon"). Reglas duras:

  - el receptor SOLO puede ser role player/goalkeeper (is_candidate_receiver);
  - los arbitros NUNCA son receptores (y se registran como rechazados);
  - se exige cercania durante varios frames + posesion estable;
  - ante ambiguedad fuerte o baja evidencia, NO se crea evento.

Salidas:
  - DataFrame de eventos (EVENT_COLUMNS)
  - DataFrame de candidatos rechazados (REJECTION_COLUMNS) para auditoria
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .possession_estimator import PossessionEstimator
from .schema import EVENT_COLUMNS, REJECTION_COLUMNS


class PassReceptionDetector:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.fps = float(c.get("fps", 25.0))
        self.min_ball_near = int(c.get("min_ball_near_frames", 3))
        self.min_possession = int(c.get("min_possession_frames", 4))
        self.min_conf = float(c.get("min_event_confidence", 0.45))
        self.reject_referee = bool(c.get("reject_referee", True))
        self.require_role = bool(c.get("require_candidate_receiver_role", True))
        self.has_homography = bool(c.get("has_homography", True))
        # coherencia de equipo pasador->receptor
        self.team_coherence_required = bool(c.get("team_coherence_required", False))
        self.team_mismatch_penalty = float(c.get("team_mismatch_penalty", 0.5))
        # transicion pasador->receptor: gap maximo razonable (frames)
        self.max_pass_gap_frames = int(c.get("max_pass_gap_frames", int(3.0 * self.fps)))
        self.possession = PossessionEstimator(c)

    # ------------------------------------------------------------------
    def detect(self, gs: pd.DataFrame, video_id: str,
               attached_by_frame: Optional[Dict[int, int]] = None,
               gt_events: Optional[pd.DataFrame] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        role_by_track = (gs.drop_duplicates("track_id")
                         .set_index("track_id")["role"].to_dict())
        team_by_track = (gs.drop_duplicates("track_id")
                         .set_index("track_id")["team_id"].to_dict())
        rejected: List[dict] = []

        if gt_events is not None and len(gt_events) > 0:
            ev = self._from_ground_truth(gt_events, gs, role_by_track, rejected)
            return self._finalize(ev, rejected)

        if not self.has_homography:
            # distancias en pixeles (menos fiables): se documenta como diagnostico
            rejected.append(self._rej(None, None, None, None, None, "missing_homography"))

        # ---------- heuristica ----------
        poss = self.possession.per_frame(gs, attached_by_frame)
        frames = sorted(poss.keys())

        # diagnosticos por frame: arbitro mas cercano + attached invalido
        self._log_referee_closeness(gs, poss, rejected)
        self._log_invalid_attached(poss, role_by_track, team_by_track, rejected)

        # runs de posesion confirmada
        events: List[dict] = []
        prev_owner: Optional[int] = None
        prev_run_end: Optional[int] = None
        i, n = 0, len(frames)
        while i < n:
            owner = poss[frames[i]].track_id
            if owner is None:
                i += 1
                continue
            j = i
            while j < n and poss[frames[j]].track_id == owner:
                j += 1
            run = frames[i:j]
            run_len = len(run)

            if run_len >= max(self.min_possession, self.min_ball_near) and owner != prev_owner:
                ev = self._make_event(video_id, gs, poss, run, owner, prev_owner,
                                      prev_run_end, role_by_track, team_by_track,
                                      rejected)
                if ev is not None:
                    events.append(ev)
                prev_owner = owner
                prev_run_end = run[-1]
            elif run_len >= self.min_possession:
                prev_owner = owner
                prev_run_end = run[-1]
            i = j

        return self._finalize(pd.DataFrame(events, columns=EVENT_COLUMNS), rejected)

    # ------------------------------------------------------------------
    def _make_event(self, video_id, gs, poss, run, owner, prev_owner, prev_run_end,
                    role_by_track, team_by_track, rejected) -> Optional[dict]:
        role = str(role_by_track.get(owner, "unknown")).lower()
        f_rec = run[0]
        # rol no candidato (defensa en profundidad; possession ya filtra)
        if self.require_role and role not in ("player", "goalkeeper"):
            rejected.append(self._rej(f_rec, owner, role, team_by_track.get(owner),
                                      poss[f_rec].distance,
                                      "referee" if role == "referee" else "unknown_role"))
            return None

        pf = poss[f_rec]
        # ball_too_far: defensa (no deberia pasar si owner no es None)
        if pf.distance is not None and pf.distance > self.possession.max_dist:
            rejected.append(self._rej(f_rec, owner, role, team_by_track.get(owner),
                                      pf.distance, "ball_too_far"))
            return None

        # unstable_track: el receptor debe existir de forma continua en el run
        if not self._track_stable(gs, owner, run):
            rejected.append(self._rej(f_rec, owner, role, team_by_track.get(owner),
                                      pf.distance, "unstable_track"))
            return None

        ambiguous = pf.ambiguous

        # --- transicion pasador->receptor razonable ---
        f_pass = prev_run_end
        passer = prev_owner
        passer_ok = (passer is not None and f_pass is not None
                     and 0 <= (f_rec - f_pass) <= self.max_pass_gap_frames)
        if not passer_ok:
            passer, f_pass = None, None     # pase no atribuible con confianza

        # --- confianza multi-condicion (no solo distancia) ---
        dist_conf = float(np.mean([poss[f].confidence for f in run]))
        stability = float(np.clip(len(run) / (self.min_possession * 2), 0.0, 1.0))
        decel = self._ball_deceleration(gs, f_rec)
        recv_team = team_by_track.get(owner)
        team_known = recv_team not in (None, -2)
        team_factor = 1.0 if team_known else 0.85
        conf = (0.5 * dist_conf + 0.3 * stability + 0.2 * decel) * team_factor
        if ambiguous:
            conf *= 0.6
        if passer is None:
            conf *= 0.9                      # sin pasador atribuible: leve penalizacion

        # --- coherencia de equipo pasador/receptor (si ambos conocidos) ---
        pass_team = team_by_track.get(passer) if passer is not None else None
        if (passer is not None and team_known and pass_team not in (None, -2)
                and pass_team != recv_team):
            if self.team_coherence_required:
                rejected.append(self._rej(f_rec, owner, role, recv_team, pf.distance,
                                          "low_confidence"))
                return None
            conf *= self.team_mismatch_penalty   # interceptacion/ruido: baja fuerte

        if ambiguous and conf < self.min_conf:
            rejected.append(self._rej(f_rec, owner, role, recv_team, pf.distance,
                                      "ambiguous_receiver"))
            return None
        if conf < self.min_conf:
            rejected.append(self._rej(f_rec, owner, role, recv_team, pf.distance,
                                      "low_confidence"))
            return None

        return {
            "event_id": None,  # se asigna en finalize
            "video_id": video_id,
            "frame_pass": int(f_pass) if f_pass is not None else None,
            "frame_reception": int(f_rec),
            "passer_track_id": int(passer) if passer is not None else None,
            "receiver_track_id": int(owner),
            "passer_team_id": pass_team if passer is not None else None,
            "receiver_team_id": recv_team,
            "receiver_role": role,
            "event_type": "reception",
            "event_confidence": round(float(np.clip(conf, 0.0, 1.0)), 3),
            "source": "heuristic",
            "rejection_reason": None,
        }

    @staticmethod
    def _track_stable(gs: pd.DataFrame, track_id: int, run: list) -> bool:
        """El receptor debe aparecer en (casi) todos los frames del run."""
        if len(run) < 2:
            return False
        present = gs[(gs["track_id"] == track_id)
                     & (gs["frame_id"] >= run[0]) & (gs["frame_id"] <= run[-1])]
        n_present = present["frame_id"].nunique()
        return n_present >= max(2, int(0.8 * len(run)))

    def _ball_deceleration(self, gs: pd.DataFrame, f_rec: int) -> float:
        """1.0 si el balon desacelera claramente antes de la recepcion."""
        win = gs[(gs["frame_id"] >= f_rec - 8) & (gs["frame_id"] <= f_rec)]
        sp = win.drop_duplicates("frame_id").sort_values("frame_id")["ball_speed"]
        sp = sp[sp.notna()]
        if len(sp) < 4:
            return 0.3
        before = float(sp.iloc[:len(sp) // 2].mean())
        after = float(sp.iloc[len(sp) // 2:].mean())
        if before <= 1e-6:
            return 0.3
        return float(np.clip((before - after) / before, 0.0, 1.0))

    def _log_referee_closeness(self, gs, poss, rejected):
        ref = gs[gs["is_referee"] & gs["distance_to_ball"].notna()]
        for fid, pf in poss.items():
            if not pf.referee_was_closest:
                continue
            sub = ref[ref["frame_id"] == fid]
            if sub.empty:
                continue
            row = sub.sort_values("distance_to_ball").iloc[0]
            rejected.append(self._rej(fid, int(row["track_id"]), "referee",
                                      row["team_id"], float(row["distance_to_ball"]),
                                      "referee"))

    def _log_invalid_attached(self, poss, role_by_track, team_by_track, rejected):
        """Frames donde habia attached_player_id pero NO paso validacion
        (no candidato / fuera de distancia / ambiguo)."""
        for fid, pf in poss.items():
            if not pf.invalid_attached:
                continue
            tid = pf.track_id
            role = str(role_by_track.get(tid, "unknown")).lower() if tid is not None else "unknown"
            rejected.append(self._rej(fid, tid, role,
                                      team_by_track.get(tid) if tid is not None else None,
                                      pf.distance, "invalid_attached_player"))

    def _from_ground_truth(self, gt, gs, role_by_track, rejected) -> pd.DataFrame:
        """GT-first con validacion estricta del receptor:
        receptor debe existir en game_state, ser candidato y no arbitro/unknown.
        Receptor faltante (missing_receiver_in_gt) -> NO entra (se registra)."""
        known_tracks = set(gs["track_id"].unique().tolist())
        keep = []
        for _, r in gt.iterrows():
            rid = r.get("receiver_track_id")
            reason = r.get("rejection_reason")
            f_rec = r.get("frame_reception")
            # 1) receptor ausente en el GT
            if pd.isna(rid) or reason == "missing_receiver_in_gt":
                rejected.append(self._rej(f_rec, None, None, None, None,
                                          "missing_receiver_in_gt"))
                continue
            rid = int(rid)
            # 2) receptor no existe en el game_state
            if rid not in known_tracks:
                rejected.append(self._rej(f_rec, rid, "unknown", None, None,
                                          "unknown_role"))
                continue
            role = str(role_by_track.get(rid, "unknown")).lower()
            # 3) arbitro
            if self.reject_referee and role == "referee":
                rejected.append(self._rej(f_rec, rid, role,
                                          self._team(gs, rid), None, "referee"))
                continue
            # 4) rol no candidato (unknown/otros)
            if self.require_role and role not in ("player", "goalkeeper"):
                rejected.append(self._rej(f_rec, rid, role,
                                          self._team(gs, rid), None, "unknown_role"))
                continue
            d = r.to_dict()
            d["receiver_track_id"] = rid
            d["receiver_role"] = role
            d["rejection_reason"] = None
            keep.append(d)
        return pd.DataFrame(keep, columns=EVENT_COLUMNS) if keep else pd.DataFrame(columns=EVENT_COLUMNS)

    @staticmethod
    def _team(gs, track_id):
        sub = gs[gs["track_id"] == track_id]
        return None if sub.empty else sub["team_id"].iloc[0]

    @staticmethod
    def _rej(frame_id, tid, role, team, dist, reason) -> dict:
        return {
            "frame_id": int(frame_id) if frame_id is not None else None,
            "candidate_track_id": int(tid) if tid is not None else None,
            "role": role, "team_id": team,
            "distance_to_ball": round(float(dist), 3) if dist is not None else None,
            "reason": reason,
        }

    def _finalize(self, events: pd.DataFrame, rejected: list):
        events = events.reset_index(drop=True)
        if "event_id" in events.columns:
            events["event_id"] = [
                f"{events['video_id'].iloc[i]}_rcp_{i:04d}" if len(events) else None
                for i in range(len(events))]
        rej_df = pd.DataFrame(rejected, columns=REJECTION_COLUMNS)
        return events, rej_df
