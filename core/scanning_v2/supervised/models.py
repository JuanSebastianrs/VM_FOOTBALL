# core/scanning_v2/supervised/models.py
"""
Fabrica de modelos supervisados SIMPLES y defendibles para scanning V2.

Permitidos: logistic_regression, random_forest, gradient_boosting. Nada de deep
learning: con pocas etiquetas un modelo simple es mas honesto y reproducible.

Cada modelo se envuelve en un Pipeline con imputacion de NaN (y escalado para la
regresion logistica). Devuelve (pipeline, resolved_params).
"""

from __future__ import annotations

from typing import Optional, Tuple

from sklearn.compose import ColumnTransformer  # noqa: F401  (reservado)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "gradient_boosting")


def build_model(config: Optional[dict] = None) -> Tuple[Pipeline, dict]:
    c = config or {}
    mtype = str(c.get("type", "logistic_regression"))
    if mtype not in SUPPORTED_MODELS:
        raise ValueError(f"model.type '{mtype}' no soportado; usa {SUPPORTED_MODELS}")
    rs = int(c.get("random_state", 42))
    class_weight = c.get("class_weight", "balanced")

    if mtype == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight,
                                 random_state=rs)
        steps = [("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()), ("clf", clf)]
        params = {"type": mtype, "random_state": rs, "class_weight": class_weight,
                  "max_iter": 1000}
    elif mtype == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(c.get("n_estimators", 200)),
            max_depth=c.get("max_depth", None),
            class_weight=class_weight, random_state=rs)
        steps = [("imputer", SimpleImputer(strategy="median")), ("clf", clf)]
        params = {"type": mtype, "random_state": rs, "class_weight": class_weight,
                  "n_estimators": int(c.get("n_estimators", 200))}
    else:  # gradient_boosting (no soporta class_weight)
        clf = GradientBoostingClassifier(
            n_estimators=int(c.get("n_estimators", 150)), random_state=rs)
        steps = [("imputer", SimpleImputer(strategy="median")), ("clf", clf)]
        params = {"type": mtype, "random_state": rs,
                  "class_weight": None, "note": "gradient_boosting ignora class_weight",
                  "n_estimators": int(c.get("n_estimators", 150))}
    return Pipeline(steps), params
