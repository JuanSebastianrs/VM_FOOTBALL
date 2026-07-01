# core/scanning/orientation_estimator.py
"""
Estimacion de orientacion visual APROXIMADA por jugador y frame.

Importante: esto NO es gaze exacto. A partir de keypoints 2D de broadcast se
estima una orientacion-proxy. La jerarquia de robustez (de mas a menos fiable)
es:

  1. Cabeza   (nose vs centro de ojos/orejas)
  2. Torso    (eje de hombros / hombros-caderas)
  3. Carrera  (direccion de movimiento, run_angle)
  4. Anterior (orientacion suavizada previa, baja confianza)

Todos los angulos se manejan en convencion matematica (+y arriba). Como los
keypoints vienen NORMALIZADOS al crop, primero se remapean al frame completo
usando la bbox expandida (evita la distorsion de aspecto del resize) y luego, si
hay homografia, se proyectan a orientacion en cancha.

El baseline heuristico es deliberadamente transparente y reproducible; el modelo
temporal entrenable (core/scanning/models/) aprende un mapeo mas fino sobre estos
mismos features + las etiquetas propias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

from .circular import (
    circular_delta,
    circular_mean,
    field_vector_to_angle,
    image_vector_to_angle,
    wrap_angle,
)
from .data_io import HomographyLookup
from .keypoint_extractor import HEAD_LANDMARKS, TORSO_LANDMARKS


@dataclass
class OrientationResult:
    frame_id: int
    track_id: int
    theta_visual_img: Optional[float]
    theta_visual_field: Optional[float]
    theta_source: str            # head | torso | movement | previous | invalid
    orientation_confidence: float
    head_angle: Optional[float] = None
    torso_angle: Optional[float] = None
    shoulder_angle: Optional[float] = None     # eje de hombros (linea L->R), diagnostico
    run_angle: Optional[float] = None
    ball_relative_angle: Optional[float] = None        # imagen
    ball_relative_angle_field: Optional[float] = None   # cancha (mismo marco que theta_visual_field)
    pose_valid: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class OrientationEstimator:
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.min_head_vis = float(config.get("min_head_visibility", 0.4))
        self.min_torso_vis = float(config.get("min_torso_visibility", 0.4))
        self.min_kp_vis = float(config.get("min_pose_confidence", 0.3))
        self.move_min_speed = float(config.get("movement_min_speed_px", 1.5))
        sc = config.get("source_confidence", {})
        self.w = {
            "head": float(sc.get("head", 1.0)),
            "torso": float(sc.get("torso", 0.7)),
            "movement": float(sc.get("movement", 0.4)),
            "previous": float(sc.get("previous", 0.2)),
        }

    # -- mapeo de landmarks normalizados -> pixeles de frame --
    @staticmethod
    def _to_frame(landmarks: Dict[str, list], bbox_exp: np.ndarray
                  ) -> Dict[str, Tuple[float, float, float]]:
        x1, y1, x2, y2 = bbox_exp
        w, h = (x2 - x1), (y2 - y1)
        out = {}
        for name, (nx, ny, _z, vis) in landmarks.items():
            out[name] = (x1 + nx * w, y1 + ny * h, vis)
        return out

    @staticmethod
    def _mean_point(pts: Dict[str, Tuple[float, float, float]], names, min_vis):
        xs, ys, ws = [], [], []
        for n in names:
            if n in pts and pts[n][2] >= min_vis:
                xs.append(pts[n][0]); ys.append(pts[n][1]); ws.append(pts[n][2])
        if not xs:
            return None
        ws = np.asarray(ws)
        return (float(np.average(xs, weights=ws)),
                float(np.average(ys, weights=ws)),
                float(ws.mean()))

    def _head_angle(self, pts) -> Tuple[Optional[float], float]:
        """nose - centro(ojos/orejas) -> angulo + confianza."""
        if "nose" not in pts or pts["nose"][2] < self.min_kp_vis:
            return None, 0.0
        center = self._mean_point(pts, ["left_eye", "right_eye", "left_ear",
                                        "right_ear"], self.min_kp_vis)
        if center is None:
            return None, 0.0
        nose = pts["nose"]
        ang = image_vector_to_angle(nose[0] - center[0], nose[1] - center[1])
        if ang is None:
            return None, 0.0
        conf = min(nose[2], center[2])
        return ang, conf

    def _torso_angles(self, pts):
        """Devuelve (shoulder_axis_angle, torso_axis_angle, confianza).

        - shoulder_axis_angle: angulo de la LINEA que une los hombros (L->R). Es
          un EJE, no la direccion de encaramiento: la mirada/encaramiento es
          PERPENDICULAR a el (ver `_perp_facing`).
        - torso_axis_angle: eje vertical del torso (caderas->hombros), diagnostico.
        """
        ls = pts.get("left_shoulder"); rs = pts.get("right_shoulder")
        shoulder_axis = None
        torso_axis = None
        conf = 0.0
        if (ls and rs and ls[2] >= self.min_kp_vis and rs[2] >= self.min_kp_vis):
            shoulder_axis = image_vector_to_angle(rs[0] - ls[0], rs[1] - ls[1])
            sh_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            conf = min(ls[2], rs[2])
            lh = pts.get("left_hip"); rh = pts.get("right_hip")
            if (lh and rh and lh[2] >= self.min_kp_vis and rh[2] >= self.min_kp_vis):
                hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
                torso_axis = image_vector_to_angle(sh_mid[0] - hip_mid[0],
                                                   sh_mid[1] - hip_mid[1])
                conf = min(conf, lh[2], rh[2])
        return shoulder_axis, torso_axis, conf

    @staticmethod
    def _perp_facing(axis_angle: float, refs) -> Tuple[float, bool]:
        """Direccion de encaramiento = perpendicular al eje de hombros.

        Hay dos perpendiculares (axis +-90 grados); el eje de hombros NO resuelve
        frente/espalda. Se elige la perpendicular mas cercana a la primera
        referencia disponible (`refs`, en orden de prioridad). Devuelve
        (facing_angle, disambiguated): `disambiguated=False` si no habia ninguna
        referencia (no se pudo resolver el frente/espalda)."""
        c1 = wrap_angle(axis_angle + math.pi / 2.0)
        c2 = wrap_angle(axis_angle - math.pi / 2.0)
        for ref in refs:
            if ref is not None:
                use_c1 = abs(circular_delta(c1, ref)) <= abs(circular_delta(c2, ref))
                return (c1 if use_c1 else c2), True
        return c1, False

    def estimate(
        self,
        frame_id: int,
        track_id: int,
        pose: dict,
        bbox_exp: np.ndarray,
        velocity: Optional[Tuple[float, float]] = None,
        ball_xy: Optional[Tuple[float, float]] = None,
        player_xy: Optional[Tuple[float, float]] = None,
        prev_theta: Optional[float] = None,
        homography: Optional[HomographyLookup] = None,
        crop_quality: float = 1.0,
    ) -> OrientationResult:
        # run_angle desde velocidad (imagen, +y abajo -> negar)
        run_angle = None
        speed = 0.0
        if velocity is not None:
            vx, vy = velocity
            speed = float(np.hypot(vx, vy))
            if speed >= self.move_min_speed:
                run_angle = image_vector_to_angle(vx, vy)

        # ball_relative_angle: direccion jugador->balon en imagen
        ball_relative_angle = None
        if ball_xy is not None and player_xy is not None:
            ball_relative_angle = image_vector_to_angle(
                ball_xy[0] - player_xy[0], ball_xy[1] - player_xy[1])

        head_angle = torso_axis = shoulder_axis = None
        theta = None
        source = "invalid"
        confidence = 0.0

        if pose.get("pose_valid"):
            pts = self._to_frame(pose["landmarks"], bbox_exp)
            head_vis = pose.get("head_visibility", 0.0)
            torso_vis = pose.get("torso_visibility", 0.0)

            head_angle, head_conf = self._head_angle(pts)
            shoulder_axis, torso_axis, torso_conf = self._torso_angles(pts)

            head_ok = head_angle is not None and head_vis >= self.min_head_vis
            torso_ok = shoulder_axis is not None and torso_vis >= self.min_torso_vis

            # Prioridad 1: cabeza (direccion de encaramiento del rostro)
            if head_ok:
                source = "head"
                confidence = self.w["head"] * head_conf
                if torso_ok:
                    # El encaramiento del torso es PERPENDICULAR al eje de hombros.
                    # Se desambigua frente/espalda con la cabeza. Ahora ambas
                    # estan en el mismo marco (direcciones de encaramiento) y SI
                    # se pueden promediar circularmente.
                    facing, _ = self._perp_facing(
                        shoulder_axis, [head_angle, run_angle, ball_relative_angle, prev_theta])
                    theta = circular_mean(
                        [head_angle, facing],
                        [self.w["head"] * head_conf, self.w["torso"] * torso_conf])
                else:
                    theta = head_angle
            # Prioridad 2: torso (encaramiento perpendicular al eje de hombros)
            elif torso_ok:
                facing, disambiguated = self._perp_facing(
                    shoulder_axis, [run_angle, ball_relative_angle, prev_theta])
                theta = facing
                source = "torso"
                # Sin referencia no se resuelve frente/espalda -> penalizar.
                confidence = self.w["torso"] * torso_conf * (1.0 if disambiguated else 0.5)

        # Prioridad 3: carrera
        if theta is None and run_angle is not None:
            theta = run_angle
            source = "movement"
            confidence = self.w["movement"] * min(1.0, speed / (self.move_min_speed * 4))
        # Prioridad 4: anterior
        if theta is None and prev_theta is not None:
            theta = prev_theta
            source = "previous"
            confidence = self.w["previous"]

        # Penalizacion por calidad de crop (crops pequenos/borrosos -> menos fiable)
        confidence *= float(np.clip(0.4 + 0.6 * crop_quality, 0.0, 1.0))

        # Proyeccion imagen -> cancha (orientacion y direccion al balon)
        theta_field = None
        ball_relative_angle_field = None
        if homography is not None and homography.available and player_xy is not None:
            if theta is not None:
                theta_field = homography.angle_image_to_field(
                    player_xy[0], player_xy[1], theta, frame_id)
            if ball_xy is not None:
                p_field = homography.project(player_xy[0], player_xy[1], frame_id)
                b_field = homography.project(ball_xy[0], ball_xy[1], frame_id)
                if p_field is not None and b_field is not None:
                    # Coords de cancha con Y hacia abajo -> negar para marco math.
                    ball_relative_angle_field = field_vector_to_angle(
                        b_field[0] - p_field[0], -(b_field[1] - p_field[1]))

        return OrientationResult(
            frame_id=frame_id,
            track_id=track_id,
            theta_visual_img=theta,
            theta_visual_field=theta_field,
            theta_source=source,
            orientation_confidence=float(np.clip(confidence, 0.0, 1.0)),
            head_angle=head_angle,
            torso_angle=torso_axis,
            shoulder_angle=shoulder_axis,
            run_angle=run_angle,
            ball_relative_angle=ball_relative_angle,
            ball_relative_angle_field=ball_relative_angle_field,
            pose_valid=bool(pose.get("pose_valid", False)),
        )
