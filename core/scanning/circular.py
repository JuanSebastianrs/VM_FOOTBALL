# core/scanning/circular.py
"""
Utilidades de matematica circular para orientaciones.

Una orientacion es una variable circular en [-pi, pi]: NO se puede promediar,
restar ni suavizar linealmente (el salto entre +pi y -pi crea artefactos). Todo
el modulo de scanning representa los angulos como vectores unitarios
[cos(theta), sin(theta)] cuando necesita combinarlos.

Convencion de angulos
---------------------
Se usa convencion matematica estandar: 0 = derecha (+x), pi/2 = arriba, y los
angulos crecen en sentido antihorario. Como en imagen el eje +y apunta hacia
ABAJO, al construir angulos en espacio-imagen se debe negar la componente y
(ver `image_vector_to_angle`). En espacio-cancha el eje y ya apunta hacia
arriba, asi que se usa atan2 directo.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(theta: float) -> float:
    """Normaliza un angulo al rango [-pi, pi]."""
    return (float(theta) + math.pi) % TWO_PI - math.pi


def angle_to_sincos(theta: float) -> Tuple[float, float]:
    """theta -> (sin, cos)."""
    return math.sin(theta), math.cos(theta)


def sincos_to_angle(sin_v: float, cos_v: float) -> float:
    """(sin, cos) -> theta en [-pi, pi]. Robusto a vectores no normalizados."""
    return math.atan2(sin_v, cos_v)


def circular_delta(theta1: float, theta2: float) -> float:
    """
    Diferencia angular con signo (theta1 - theta2) tomando el camino mas corto.
    Resultado en [-pi, pi].
    """
    return wrap_angle(theta1 - theta2)


def circular_abs_delta(theta1: float, theta2: float) -> float:
    """Magnitud de la diferencia angular en [0, pi]."""
    return abs(circular_delta(theta1, theta2))


def image_vector_to_angle(dx: float, dy: float) -> Optional[float]:
    """
    Angulo (convencion matematica, +y arriba) de un vector dado en espacio-imagen
    donde +y apunta hacia abajo. Devuelve None si el vector es ~0.
    """
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.atan2(-dy, dx)


def field_vector_to_angle(dx: float, dy: float) -> Optional[float]:
    """Angulo de un vector ya expresado en espacio-cancha (+y arriba)."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.atan2(dy, dx)


def circular_mean(
    angles: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> Optional[float]:
    """
    Media circular ponderada. Devuelve None si no hay datos o el resultante es 0.
    """
    if len(angles) == 0:
        return None
    if weights is None:
        weights = [1.0] * len(angles)
    sx = sum(w * math.cos(a) for a, w in zip(angles, weights))
    sy = sum(w * math.sin(a) for a, w in zip(angles, weights))
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return None
    return math.atan2(sy, sx)


def circular_resultant_length(angles: Sequence[float]) -> float:
    """
    Longitud del vector resultante R en [0, 1]. R~1 => angulos concentrados,
    R~0 => dispersos. Util como medida inversa de dispersion.
    """
    n = len(angles)
    if n == 0:
        return 0.0
    sx = sum(math.cos(a) for a in angles) / n
    sy = sum(math.sin(a) for a in angles) / n
    return math.hypot(sx, sy)


def angle_to_bin(theta: float, num_bins: int = 8) -> int:
    """
    Discretiza un angulo (convencion matematica) en `num_bins` bins.
    Para num_bins=8: 0=derecha, 1=diag-sup-der, 2=arriba, 3=diag-sup-izq,
    4=izquierda, 5=diag-inf-izq, 6=abajo, 7=diag-inf-der.
    """
    step = TWO_PI / num_bins
    # +step/2 centra el bin 0 en theta=0
    idx = int(math.floor((wrap_angle(theta) + step / 2.0) % TWO_PI / step))
    return idx % num_bins


def bin_to_angle(bin_idx: int, num_bins: int = 8) -> float:
    """Angulo central (rad) de un bin."""
    step = TWO_PI / num_bins
    return wrap_angle(bin_idx * step)


def orientation_entropy(angles: Sequence[float], num_bins: int = 8) -> float:
    """
    Entropia de Shannon (en bits) de la distribucion de orientaciones por bin,
    normalizada por log2(num_bins) -> [0, 1]. Mide cuanto barrio el jugador el
    espacio angular: ~0 = mira siempre igual, ~1 = mira en todas direcciones.
    """
    if len(angles) == 0:
        return 0.0
    counts = np.zeros(num_bins, dtype=np.float64)
    for a in angles:
        counts[angle_to_bin(a, num_bins)] += 1.0
    p = counts / counts.sum()
    p = p[p > 0]
    h = -np.sum(p * np.log2(p))
    if num_bins <= 1:
        return 0.0
    return float(max(0.0, h / math.log2(num_bins)))


class CircularEMA:
    """
    Media exponencial circular incremental sobre el vector unitario del angulo.

        v_t       = [cos(theta_t), sin(theta_t)]
        v_smooth  = alpha * v_t + (1 - alpha) * v_smooth_{t-1}
        theta_out = atan2(v_smooth_y, v_smooth_x)

    En frames invalidos el estado se conserva y se atenua con `decay` para que
    la confianza del suavizado decaiga sin saltos.
    """

    def __init__(self, alpha: float = 0.35, decay: float = 0.85):
        self.alpha = float(alpha)
        self.decay = float(decay)
        self._vx: Optional[float] = None
        self._vy: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._vx is not None

    @property
    def magnitude(self) -> float:
        """Longitud del vector suavizado en [0, 1]; proxy de estabilidad."""
        if self._vx is None:
            return 0.0
        return math.hypot(self._vx, self._vy)

    def update(self, theta: float, alpha: Optional[float] = None) -> float:
        """Incorpora una observacion valida y devuelve el angulo suavizado."""
        a = self.alpha if alpha is None else float(alpha)
        cx, sx = math.cos(theta), math.sin(theta)
        if self._vx is None:
            self._vx, self._vy = cx, sx
        else:
            self._vx = a * cx + (1.0 - a) * self._vx
            self._vy = a * sx + (1.0 - a) * self._vy
        return math.atan2(self._vy, self._vx)

    def coast(self) -> Optional[float]:
        """Frame invalido: atenua el estado y devuelve el angulo actual (o None)."""
        if self._vx is None:
            return None
        self._vx *= self.decay
        self._vy *= self.decay
        return math.atan2(self._vy, self._vx)

    def value(self) -> Optional[float]:
        if self._vx is None:
            return None
        return math.atan2(self._vy, self._vx)
