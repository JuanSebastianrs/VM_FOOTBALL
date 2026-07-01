# core/scanning/data_io.py
"""
Adaptadores de entrada/salida del modulo de scanning.

Convierte los formatos JSON que ya produce el pipeline tactico
(`*_detections.json`, `*_trajectory.json`, `*_team_assignments.json`,
`calibration_hinv.json`) en estructuras limpias y tipadas que el resto del
modulo consume. Asi el modulo de scanning no depende de los detalles de
serializacion de las fases previas.

Formatos esperados (ver outputs/SNMOT-*/):
  detections.json   : [{frame_id, ball_candidates:[{x_center,y_center,w,h,score}],
                        players:[{track_id, x_min, y_min, x_max, y_max, class_id}]}]
  trajectory.json   : [{frame_id, x, y, w, h, is_dummy, attached_player_id}]
  team_assignments  : [{track_id, team_id, role}]
  calibration_hinv  : {"<frame_id>": {"H_inv": 3x3, "time_s": float}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .circular import field_vector_to_angle


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class PlayerDetection:
    frame_id: int
    track_id: int
    bbox: np.ndarray              # [x1, y1, x2, y2] en pixeles
    class_id: int = 0
    team_id: Optional[int] = None
    role: str = "unknown"         # sin asignacion explicita -> NO se asume "player"

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        return (float(self.bbox[0] + self.bbox[2]) / 2.0,
                float(self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def foot(self) -> Tuple[float, float]:
        """Punto de apoyo (centro inferior de la bbox), usado para mapear a cancha."""
        return (float(self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3]))


@dataclass
class BallObservation:
    frame_id: int
    x: Optional[float]
    y: Optional[float]
    is_dummy: bool = False
    attached_player_id: Optional[int] = None
    score: float = 1.0

    @property
    def valid(self) -> bool:
        return self.x is not None and self.y is not None and not self.is_dummy


# ---------------------------------------------------------------------------
# Homografia
# ---------------------------------------------------------------------------

class HomographyLookup:
    """
    Envuelve `calibration_hinv.json`. H_inv mapea coordenadas de imagen a
    coordenadas de mundo PnLCalib (origen en el centro de la cancha, en metros).
    Las coords de cancha que devuelve estan CENTRADAS: X in [-L/2, L/2],
    Y in [-W/2, W/2], con Y creciente hacia abajo (igual que el minimapa).
    """

    def __init__(self, hinv_by_frame: Dict[int, np.ndarray]):
        self._hinv = hinv_by_frame

    @classmethod
    def from_json(cls, path: Optional[str | Path]) -> "HomographyLookup":
        if path is None or not Path(path).exists():
            return cls({})
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        hinv: Dict[int, np.ndarray] = {}
        for k, v in raw.items():
            H = v.get("H_inv") if isinstance(v, dict) else v
            if H is None:
                continue
            try:
                hinv[int(k)] = np.asarray(H, dtype=np.float64).reshape(3, 3)
            except Exception:
                continue
        return cls(hinv)

    @property
    def available(self) -> bool:
        return len(self._hinv) > 0

    def get(self, frame_id: int) -> Optional[np.ndarray]:
        if frame_id in self._hinv:
            return self._hinv[frame_id]
        # fallback: homografia del frame mas cercano disponible
        if not self._hinv:
            return None
        nearest = min(self._hinv.keys(), key=lambda k: abs(k - frame_id))
        if abs(nearest - frame_id) <= 5:
            return self._hinv[nearest]
        return None

    def project(self, x_img: float, y_img: float,
                frame_id: int) -> Optional[Tuple[float, float]]:
        """Proyecta un punto imagen -> cancha (metros, centrado). None si falla."""
        H = self.get(frame_id)
        if H is None:
            return None
        p = H @ np.array([x_img, y_img, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        x, y = p[0] / p[2], p[1] / p[2]
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return float(x), float(y)

    def angle_image_to_field(
        self,
        x_img: float,
        y_img: float,
        theta_img: float,
        frame_id: int,
        step_px: float = 30.0,
    ) -> Optional[float]:
        """
        Convierte una orientacion en espacio-imagen a espacio-cancha proyectando
        dos puntos (el jugador y un punto adelantado en direccion theta_img) y
        recomputando el angulo en cancha. Devuelve angulo en convencion
        matematica (+y arriba) o None si no hay homografia valida.

        theta_img viene en convencion matematica de imagen (+y arriba): el punto
        adelantado en imagen usa -sin(theta) porque el eje y de imagen baja.
        """
        p1 = self.project(x_img, y_img, frame_id)
        dx_img = step_px * np.cos(theta_img)
        dy_img = -step_px * np.sin(theta_img)  # imagen: +y hacia abajo
        p2 = self.project(x_img + dx_img, y_img + dy_img, frame_id)
        if p1 is None or p2 is None:
            return None
        dX = p2[0] - p1[0]
        dY = p2[1] - p1[1]
        # Coords de cancha tienen Y hacia abajo -> negar para +y arriba
        return field_vector_to_angle(dX, -dY)


# ---------------------------------------------------------------------------
# Contenedor de secuencia
# ---------------------------------------------------------------------------

@dataclass
class SequenceData:
    """Toda la informacion necesaria para correr scanning sobre una secuencia."""
    video_id: str
    fps: float
    frame_ids: List[int]
    players_by_frame: Dict[int, List[PlayerDetection]]
    ball_by_frame: Dict[int, BallObservation]
    team_by_track: Dict[int, int]
    role_by_track: Dict[int, str]
    homography: HomographyLookup
    image_paths: Dict[int, Path] = field(default_factory=dict)
    frame_size: Optional[Tuple[int, int]] = None   # (w, h)

    def players(self, frame_id: int) -> List[PlayerDetection]:
        return self.players_by_frame.get(frame_id, [])

    def player(self, frame_id: int, track_id: int) -> Optional[PlayerDetection]:
        for p in self.players_by_frame.get(frame_id, []):
            if p.track_id == track_id:
                return p
        return None

    def ball(self, frame_id: int) -> Optional[BallObservation]:
        return self.ball_by_frame.get(frame_id)

    def track_ids(self) -> List[int]:
        ids = set()
        for ps in self.players_by_frame.values():
            for p in ps:
                ids.add(p.track_id)
        return sorted(ids)

    def player_field_xy(self, frame_id: int, track_id: int) -> Optional[Tuple[float, float]]:
        p = self.player(frame_id, track_id)
        if p is None:
            return None
        fx, fy = p.foot
        return self.homography.project(fx, fy, frame_id)

    def ball_field_xy(self, frame_id: int) -> Optional[Tuple[float, float]]:
        b = self.ball(frame_id)
        if b is None or not b.valid:
            return None
        return self.homography.project(b.x, b.y, frame_id)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sequence(
    video_id: str,
    detections_json: str | Path,
    trajectory_json: Optional[str | Path] = None,
    team_assignments_json: Optional[str | Path] = None,
    calibration_json: Optional[str | Path] = None,
    sequence_dir: Optional[str | Path] = None,
    fps: float = 25.0,
) -> SequenceData:
    """Carga todos los insumos de una secuencia en un `SequenceData`."""
    det = _load_json(detections_json)

    players_by_frame: Dict[int, List[PlayerDetection]] = {}
    frame_ids: List[int] = []
    for frame in det:
        fid = int(frame["frame_id"])
        frame_ids.append(fid)
        plist: List[PlayerDetection] = []
        for p in frame.get("players", []):
            bbox = np.array([p["x_min"], p["y_min"], p["x_max"], p["y_max"]],
                            dtype=np.float64)
            plist.append(PlayerDetection(
                frame_id=fid,
                track_id=int(p["track_id"]),
                bbox=bbox,
                class_id=int(p.get("class_id", 0)),
            ))
        players_by_frame[fid] = plist
    frame_ids = sorted(set(frame_ids))

    # --- Balon: preferir trajectory (refinado, con attached_player_id) ---
    ball_by_frame: Dict[int, BallObservation] = {}
    if trajectory_json is not None and Path(trajectory_json).exists():
        for t in _load_json(trajectory_json):
            fid = int(t["frame_id"])
            ball_by_frame[fid] = BallObservation(
                frame_id=fid,
                x=float(t["x"]) if t.get("x") is not None else None,
                y=float(t["y"]) if t.get("y") is not None else None,
                is_dummy=bool(t.get("is_dummy", False)),
                attached_player_id=t.get("attached_player_id"),
            )
    else:
        # fallback: mejor ball_candidate por frame
        for frame in det:
            fid = int(frame["frame_id"])
            cands = frame.get("ball_candidates", [])
            if cands:
                best = max(cands, key=lambda c: c.get("score", 0.0))
                ball_by_frame[fid] = BallObservation(
                    frame_id=fid, x=float(best["x_center"]),
                    y=float(best["y_center"]), score=float(best.get("score", 1.0)))
            else:
                ball_by_frame[fid] = BallObservation(fid, None, None, is_dummy=True)

    # --- Equipos / roles ---
    team_by_track: Dict[int, int] = {}
    role_by_track: Dict[int, str] = {}
    if team_assignments_json is not None and Path(team_assignments_json).exists():
        for a in _load_json(team_assignments_json):
            tid = int(a["track_id"])
            team_by_track[tid] = int(a.get("team_id", -1))
            # rol explicito; si viene vacio/None -> "unknown" (no se asume "player")
            raw_role = a.get("role")
            role_by_track[tid] = str(raw_role).lower() if raw_role else "unknown"
    # inyectar team/role en las detecciones; sin asignacion -> "unknown"
    for plist in players_by_frame.values():
        for p in plist:
            p.team_id = team_by_track.get(p.track_id)
            p.role = role_by_track.get(p.track_id, "unknown")

    homography = HomographyLookup.from_json(calibration_json)

    # --- Mapeo frame_id -> imagen ---
    image_paths: Dict[int, Path] = {}
    frame_size = None
    if sequence_dir is not None:
        seq = Path(sequence_dir)
        img_dir = seq / "img1" if (seq / "img1").exists() else seq
        imgs = sorted([p for p in img_dir.glob("*.jpg")] +
                      [p for p in img_dir.glob("*.png")])
        # SoccerNet: 000001.jpg corresponde a frame_id 1
        for p in imgs:
            try:
                fid = int(p.stem)
            except ValueError:
                continue
            image_paths[fid] = p
        if imgs:
            import cv2
            im = cv2.imread(str(imgs[0]))
            if im is not None:
                frame_size = (im.shape[1], im.shape[0])

    return SequenceData(
        video_id=video_id,
        fps=fps,
        frame_ids=frame_ids,
        players_by_frame=players_by_frame,
        ball_by_frame=ball_by_frame,
        team_by_track=team_by_track,
        role_by_track=role_by_track,
        homography=homography,
        image_paths=image_paths,
        frame_size=frame_size,
    )
