# team_classifier.py
"""Clasificador de equipos basado en HSV con modos K-Means y DBSCAN."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN


@dataclass
class TeamAssignment:
    """Asignación de equipo para un jugador."""
    track_id: int
    team_id: int          # 0..k-1 para clusters; -1 = outlier (solo en DBSCAN)
    confidence: float     # Confianza de la asignación [0, 1]
    embedding: Optional[np.ndarray] = None
    is_outlier: bool = False


class TeamClassifier:
    """
    Clasificador de equipos para jugadores de fútbol.

    Modos disponibles:
        "hsv"    : K-Means sobre descriptores HSV
        "dbscan" : DBSCAN para detectar outliers + K-Means sobre inliers

    Uso típico (una vez por secuencia, datos acumulados de todos los tracks):
        classifier = TeamClassifier(mode="dbscan")
        # Acumular crops y bboxes representativas por track_id...
        track_to_team = classifier.fit(crops, track_ids, bboxes)
        # Luego en tiempo real:
        assignment = classifier.predict(new_crop, track_id=tid)
    """

    MODES = ["hsv", "dbscan"]

    def __init__(
        self,
        mode: str = "hsv",
        n_clusters: int = 2,
        random_state: int = 42,
        dbscan_eps: float = 0.12,
        dbscan_min_samples: int = 3,
        use_value: bool = True,
    ):
        """
        Args:
            mode            : "hsv" | "dbscan"
            n_clusters      : Número de clusters para K-Means
            random_state    : Semilla para K-Means
            dbscan_eps      : Radio de vecindad DBSCAN (espacio HSV normalizado)
            dbscan_min_samples : Mínimo de vecinos para punto central DBSCAN
        """
        if mode not in self.MODES:
            raise ValueError(f"mode debe ser uno de {self.MODES}, se recibió '{mode}'")

        self.mode = mode
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        self.use_value = use_value
        self.kmeans: Optional[KMeans] = None
        self.track_to_team: Dict[int, int] = {}
        self._cluster_centers: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        crops: List[np.ndarray],
        track_ids: Optional[List[int]] = None,
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> Dict[int, int]:
        """
        Ajustar el clasificador sobre una colección de crops representativos.

        En la práctica se pasa un crop (o la media acumulada) por track_id,
        equivalente a lo que hace el script legacy cv_model/team_clustering.py
        con `tracks_sum[tid] / tracks_count[tid]`.

        Args:
            crops     : Lista de crops BGR de torsos (uno por track)
            track_ids : IDs de track correspondientes
            bboxes    : Bboxes [x1,y1,x2,y2] para la heurística izq/der

        Returns:
            Diccionario {track_id: team_id} con team_id en {0..k-1} o -1.
        """
        embeddings: List[np.ndarray] = []
        valid_indices: List[int] = []

        for i, crop in enumerate(crops):
            emb = self._get_embedding(crop)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)

        if len(embeddings) < self.n_clusters:
            raise RuntimeError(
                f"Se necesitan al menos {self.n_clusters} crops válidos para "
                f"clustering, se obtuvieron {len(embeddings)}."
            )

        X = np.vstack(embeddings)  # (N, D)

        labels = self._fit_kmeans(X) if self.mode == "hsv" else self._fit_dbscan_kmeans(X)

        # Heurística izquierda/derecha sobre bboxes válidas
        if bboxes is not None:
            valid_bboxes = [bboxes[i] for i in valid_indices]
            labels = self._normalize_team_labels(labels, valid_bboxes)

        # Construir mapa track_id -> team_id
        if track_ids is not None:
            valid_track_ids = [track_ids[i] for i in valid_indices]
            self.track_to_team = {
                int(tid): int(lbl)
                for tid, lbl in zip(valid_track_ids, labels)
            }

        return self.track_to_team

    def predict(self, crop: np.ndarray, track_id: Optional[int] = None) -> int:
        """
        Predecir equipo para un crop nuevo.

        Si el track ya fue visto en fit(), devuelve la asignación almacenada.
        Si es un track nuevo, usa el centroide K-Means más cercano.

        Args:
            crop     : Crop BGR del torso
            track_id : ID de track (opcional, para lookup rápido)

        Returns:
            team_id: 0..k-1 o -1 (outlier / crop inválido)
        """
        if track_id is not None and track_id in self.track_to_team:
            return self.track_to_team[track_id]

        if self._cluster_centers is None:
            raise RuntimeError("Llamar fit() antes de predict().")

        emb = self._get_embedding(crop)
        if emb is None:
            return -1

        dists = np.linalg.norm(self._cluster_centers - emb, axis=1)
        return int(np.argmin(dists))

    def predict_batch(
        self,
        crops: List[np.ndarray],
        track_ids: Optional[List[int]] = None,
    ) -> List[TeamAssignment]:
        """
        Predecir equipos para múltiples crops.

        Args:
            crops     : Lista de crops BGR
            track_ids : IDs de track correspondientes

        Returns:
            Lista de TeamAssignment ordenada igual que crops
        """
        if self._cluster_centers is None:
            raise RuntimeError("Llamar fit() antes de predict_batch().")

        results: List[TeamAssignment] = []
        for i, crop in enumerate(crops):
            tid = int(track_ids[i]) if track_ids else i

            if tid in self.track_to_team:
                team_id = self.track_to_team[tid]
                results.append(TeamAssignment(
                    track_id=tid,
                    team_id=team_id,
                    confidence=1.0,
                    is_outlier=(team_id == -1),
                ))
                continue

            emb = self._get_embedding(crop)
            if emb is None:
                results.append(TeamAssignment(
                    track_id=tid, team_id=-1, confidence=0.0, is_outlier=True,
                ))
                continue

            dists = np.linalg.norm(self._cluster_centers - emb, axis=1)
            team_id = int(np.argmin(dists))

            sorted_d = np.sort(dists)
            confidence = float(1.0 - sorted_d[0] / (sorted_d[0] + sorted_d[1] + 1e-9)) \
                if len(sorted_d) >= 2 else 1.0

            results.append(TeamAssignment(
                track_id=tid,
                team_id=team_id,
                confidence=confidence,
                is_outlier=False,
            ))

        return results

    def evaluate(self, ground_truth: Dict[int, int]) -> Dict[str, float]:
        """
        Evaluar clasificación contra etiquetas ground truth.

        Args:
            ground_truth : {track_id: team_id_real}

        Returns:
            {"accuracy": float, "n_tracks": int, "n_outliers": int}
        """
        if not self.track_to_team:
            raise RuntimeError("Clasificador no ajustado. Llamar fit() primero.")

        common = [t for t in self.track_to_team if t in ground_truth]
        if not common:
            return {"accuracy": 0.0, "n_tracks": 0, "n_outliers": 0}

        pred = np.array([self.track_to_team[t] for t in common])
        truth = np.array([ground_truth[t] for t in common])

        non_outlier = pred != -1
        n_outliers = int((pred == -1).sum())

        if non_outlier.sum() == 0:
            return {"accuracy": 0.0, "n_tracks": len(common), "n_outliers": n_outliers}

        p, t = pred[non_outlier], truth[non_outlier]
        acc = float((p == t).mean())

        return {
            "accuracy": acc,
            "n_tracks": len(common),
            "n_outliers": n_outliers,
        }

    def fit_from_descriptors(
        self,
        descriptors: List[np.ndarray],
        track_ids: Optional[List[int]] = None,
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> Dict[int, int]:
        """
        Igual que fit() pero recibe descriptores ya computados (vectores float32)
        en lugar de crops BGR crudos. Úsalo cuando ya acumulaste el descriptor
        medio por track fuera del clasificador (evita re-computar el histograma).

        Args:
            descriptors : Lista de vectores float32, uno por track
            track_ids   : IDs de track correspondientes
            bboxes      : Bboxes para heurística izq/der

        Returns:
            {track_id: team_id}
        """
        valid_descs: List[np.ndarray] = []
        valid_indices: List[int] = []

        for i, d in enumerate(descriptors):
            if d is not None and d.size > 0:
                valid_descs.append(d.astype(np.float32))
                valid_indices.append(i)

        if len(valid_descs) < self.n_clusters:
            raise RuntimeError(
                f"Se necesitan al menos {self.n_clusters} descriptores válidos, "
                f"se obtuvieron {len(valid_descs)}."
            )

        X = np.vstack(valid_descs)

        labels = self._fit_kmeans(X) if self.mode == "hsv" else self._fit_dbscan_kmeans(X)

        if bboxes is not None:
            valid_bboxes = [bboxes[i] for i in valid_indices]
            labels = self._normalize_team_labels(labels, valid_bboxes)

        if track_ids is not None:
            valid_track_ids = [track_ids[i] for i in valid_indices]
            self.track_to_team = {
                int(tid): int(lbl)
                for tid, lbl in zip(valid_track_ids, labels)
            }

        return self.track_to_team

    # ------------------------------------------------------------------
    # Métodos de clustering interno
    # ------------------------------------------------------------------

    def _fit_kmeans(self, X: np.ndarray) -> np.ndarray:
        """K-Means puro sobre descriptores HSV."""
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(X)
        self._cluster_centers = self.kmeans.cluster_centers_.copy()
        return labels

    def _fit_dbscan_kmeans(self, X: np.ndarray) -> np.ndarray:
        """
          DBSCAN para outliers + K-Means sobre inliers.
        """
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        db_labels = db.fit_predict(X)

        inlier_mask = db_labels != -1
        n_inliers = int(inlier_mask.sum())

        if n_inliers < self.n_clusters:
            return self._fit_kmeans(X)

        X_inliers = X[inlier_mask]
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        km_labels = self.kmeans.fit_predict(X_inliers)
        self._cluster_centers = self.kmeans.cluster_centers_.copy()

        final_labels = np.full(len(X), -1, dtype=int)
        final_labels[inlier_mask] = km_labels
        return final_labels

    # ------------------------------------------------------------------
    # Descriptores / embeddings
    # ------------------------------------------------------------------

    def _get_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        return self._hsv_descriptor(crop)

    def _hsv_descriptor(
        self,
        crop: np.ndarray,
        bins: int = 12,
    ) -> Optional[np.ndarray]:
        """
        Histograma H+S normalizado con máscara anti-verde (pasto).

        12 bins por canal (H y S, opcionalmente V) normalizados en [0,1].

        Args:
            crop : Crop BGR del torso
            bins : Bins por canal HSV

        Returns:
            Vector float32 normalizado, o None si el crop es inválido/vacío.
        """
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(green_mask)

        if cv2.countNonZero(mask) < 16:
            return None

        h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])

        channels = [h_hist.ravel(), s_hist.ravel()]
        if self.use_value:
            v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])
            channels.append(v_hist.ravel())

        feat = np.concatenate(channels).astype(np.float32)
        total = float(feat.sum())
        if total > 0.0:
            feat /= total
            return feat
        return None

    # ------------------------------------------------------------------
    # Normalización de etiquetas
    # ------------------------------------------------------------------

    def _normalize_team_labels(
        self,
        labels: np.ndarray,
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Normaliza etiquetas ordenando clusters de izquierda a derecha.

        Los outliers (label=-1) no participan en la decisión y se preservan.

        Args:
            labels : Array de etiquetas raw del clustering
            bboxes : Lista de bboxes [x1,y1,x2,y2] paralela a labels

        Returns:
            Array de etiquetas reordenadas (misma longitud que labels)
        """
        if bboxes is None or len(bboxes) == 0:
            return labels

        xs_by_team: Dict[int, List[float]] = {}
        for lbl, bbox in zip(labels, bboxes):
            lbl = int(lbl)
            if lbl < 0:
                continue
            x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
            xs_by_team.setdefault(lbl, []).append(x_center)

        if len(xs_by_team) < 2:
            return labels

        team_means = {t: float(np.mean(xs)) for t, xs in xs_by_team.items()}
        sorted_teams = sorted(team_means, key=lambda t: team_means[t])
        remap = {old_label: new_label for new_label, old_label in enumerate(sorted_teams)}
        labels = np.array([remap.get(int(l), int(l)) for l in labels])

        if self._cluster_centers is not None:
            k = self._cluster_centers.shape[0]
            if len(sorted_teams) == k and set(sorted_teams) == set(range(k)):
                self._cluster_centers = self._cluster_centers[sorted_teams, :]

        return labels

    # ------------------------------------------------------------------
    # Utilidades estáticas
    # ------------------------------------------------------------------

    @staticmethod
    def crop_torso(
        image: np.ndarray,
        bbox: np.ndarray,
        top_ratio: float = 0.15,
        bottom_ratio: float = 0.60,
        left_ratio: float = 0.20,
        right_ratio: float = 0.80,
    ) -> Optional[np.ndarray]:
        """
        Recortar zona del torso (camiseta) de un bbox [x1, y1, x2, y2].

        Los ratios excluyen cabeza/piernas y reducen ancho para evitar ruido.

        Args:
            image        : Imagen completa BGR
            bbox         : [x1, y1, x2, y2]
            top_ratio    : Fracción desde y1 donde empieza el torso
            bottom_ratio : Fracción desde y1 donde termina el torso
            left_ratio   : Fracción desde x1 del lado izquierdo del torso
            right_ratio  : Fracción desde x1 del lado derecho del torso

        Returns:
            Crop BGR de la zona de camiseta, o None si bbox es inválido.
        """
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None

        top = max(y1, min(y1 + int(top_ratio * h), y2 - 1))
        bottom = max(top + 1, min(y1 + int(bottom_ratio * h), y2))
        left = max(x1, min(x1 + int(left_ratio * w), x2 - 1))
        right = max(left + 1, min(x1 + int(right_ratio * w), x2))

        return image[top:bottom, left:right]
