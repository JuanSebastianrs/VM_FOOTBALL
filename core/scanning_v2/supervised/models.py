# core/scanning_v2/supervised/models.py
"""
Fabrica de modelos supervisados para scanning V2.

Arquitecturas disponibles (de mas simple a mas expresiva):
  - logistic_regression      : baseline lineal, interpretable.
  - random_forest            : ensamble de arboles.
  - gradient_boosting        : boosting clasico (sin class_weight).
  - hist_gradient_boosting   : boosting con soporte NATIVO de NaN y class_weight.
  - mlp                      : perceptron multicapa (sklearn) sobre features tabulares.
  - gru_sequence             : GRU temporal sobre la serie de yaw re-muestreada
                               (requiere dataset.include_sequence_features=true)
                               + rama tabular. Ver sequence_model.py.

Los modelos simples siguen siendo el default defendible con GT pequeno; las
arquitecturas mas expresivas (mlp, gru_sequence) solo tienen sentido con
suficientes etiquetas (p. ej. etiquetado debil multi-video documentado).

Cada modelo se envuelve en un Pipeline con imputacion de NaN (y escalado donde
aplica). Devuelve (pipeline, resolved_params).
"""

from __future__ import annotations

from typing import Optional, Tuple

from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "gradient_boosting",
                    "hist_gradient_boosting", "mlp", "gru_sequence")


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
    elif mtype == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=int(c.get("n_estimators", 150)), random_state=rs)
        steps = [("imputer", SimpleImputer(strategy="median")), ("clf", clf)]
        params = {"type": mtype, "random_state": rs,
                  "class_weight": None, "note": "gradient_boosting ignora class_weight",
                  "n_estimators": int(c.get("n_estimators", 150))}
    elif mtype == "hist_gradient_boosting":
        # maneja NaN de forma nativa -> sin imputer (mejor trato de "sin dato")
        clf = HistGradientBoostingClassifier(
            max_iter=int(c.get("n_estimators", 300)),
            learning_rate=float(c.get("learning_rate", 0.08)),
            max_depth=c.get("max_depth", None),
            class_weight=class_weight,
            early_stopping=True, validation_fraction=0.15,
            random_state=rs)
        steps = [("clf", clf)]
        params = {"type": mtype, "random_state": rs, "class_weight": class_weight,
                  "max_iter": int(c.get("n_estimators", 300)),
                  "learning_rate": float(c.get("learning_rate", 0.08)),
                  "note": "NaN nativo (sin imputer)"}
    elif mtype == "mlp":
        hidden = tuple(int(h) for h in c.get("hidden_layer_sizes", (64, 32)))
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000,
                            early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=25, alpha=float(c.get("alpha", 1e-3)),
                            random_state=rs)
        steps = [("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()), ("clf", clf)]
        params = {"type": mtype, "random_state": rs,
                  "class_weight": None, "note": "MLPClassifier ignora class_weight",
                  "hidden_layer_sizes": list(hidden)}
    else:  # gru_sequence (torch; import perezoso para no exigir torch siempre)
        from .sequence_model import GRUSequenceClassifier
        clf = GRUSequenceClassifier(
            hidden_size=int(c.get("hidden_size", 32)),
            num_layers=int(c.get("num_layers", 1)),
            bidirectional=bool(c.get("bidirectional", True)),
            dropout=float(c.get("dropout", 0.2)),
            lr=float(c.get("lr", 1e-3)),
            batch_size=int(c.get("batch_size", 32)),
            max_epochs=int(c.get("max_epochs", 300)),
            patience=int(c.get("patience", 30)),
            use_tabular=bool(c.get("use_tabular", True)),
            class_weight=class_weight, random_state=rs,
            device=str(c.get("device", "auto")))
        steps = [("clf", clf)]   # NaN/imputacion se maneja dentro del estimador
        params = {"type": mtype, "random_state": rs, "class_weight": class_weight,
                  "hidden_size": int(c.get("hidden_size", 32)),
                  "bidirectional": bool(c.get("bidirectional", True)),
                  "note": ("GRU temporal sobre seq_cos/sin/valid + rama tabular; "
                           "requiere include_sequence_features")}
    return Pipeline(steps), params
