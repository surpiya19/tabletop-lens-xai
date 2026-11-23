"""
Step 2: the offline FACET analysis. Run once; the webapp only loads the results.

  1. Model selection   - FACET LearnerSelector compares Random Forest vs LightGBM
                         over a small hyperparameter grid with repeated K-fold CV.
  2. Inspection        - FACET LearnerInspector: SHAP importance, plus the pairwise
                         synergy and redundancy matrices and redundancy dendrogram.
  3. Simulation        - FACET UnivariateUpliftSimulator: "what if every game had
                         value X for this feature?" for the numeric features.

Outputs go to artifacts/ (model, JSON/CSV results).

Run:  python analyze.py          (takes a few minutes)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from facet.data import Sample
from facet.data.partition import ContinuousRangePartitioner, IntegerRangePartitioner
from facet.inspection import LearnerInspector
from facet.selection import LearnerSelector, ParameterSpace
from facet.simulation import UnivariateUpliftSimulator
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression import RandomForestRegressorDF
from sklearndf.regression.extra import LGBMRegressorDF

ART = Path("artifacts")
ART.mkdir(exist_ok=True)
SEED = 42
N_INSPECT = 3000  # rows used for SHAP interaction values (keeps runtime reasonable)

NUMERIC = ["year", "min_players", "max_players", "play_time", "min_play_time",
           "min_age", "n_mechanics"]
NICE = {
    "year": "Release year", "min_players": "Min players", "max_players": "Max players",
    "play_time": "Play time (min)", "min_play_time": "Min play time (min)",
    "min_age": "Minimum age", "n_mechanics": "Number of mechanics",
}


def matrix_to_df(m) -> pd.DataFrame:
    """pytools Matrix -> labelled DataFrame."""
    rows = list(m.names[0]) if m.names[0] is not None else None
    cols = list(m.names[1]) if m.names[1] is not None else None
    return pd.DataFrame(np.asarray(m.values), index=rows, columns=cols)


def main() -> None:
    t0 = time.time()
    games = pd.read_csv("data/games.csv")
    labels = {**NICE, **json.loads(Path("data/binary_labels.json").read_text())}
    features = [c for c in games.columns if c not in ("name", "rating")]

    sample = Sample(observations=games.drop(columns="name"), target_name="rating",
                    feature_names=features)

    # ---------------------------------------------------------------- 1. selection
    rf = ParameterSpace(RegressorPipelineDF(
        regressor=RandomForestRegressorDF(random_state=SEED, n_jobs=1)))
    rf.regressor.n_estimators = [200]
    rf.regressor.min_samples_leaf = [5, 10]
    rf.regressor.max_features = [0.33, 0.5]

    lgbm = ParameterSpace(RegressorPipelineDF(
        regressor=LGBMRegressorDF(random_state=SEED, verbose=-1, n_jobs=1)))
    lgbm.regressor.n_estimators = [300]
    lgbm.regressor.learning_rate = [0.03, 0.06]
    lgbm.regressor.num_leaves = [15, 31]
    lgbm.regressor.min_child_samples = [20]

    selector = LearnerSelector(
        searcher_type=GridSearchCV,
        parameter_space=[rf, lgbm],
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED),
        scoring="r2",
        n_jobs=-1,
    ).fit(sample)

    report = selector.summary_report()
    report.to_csv(ART / "model_selection.csv")
    # flat, readable leaderboard for the app
    flat = report.copy()
    flat.columns = ["_".join(str(p) for p in c if p not in ("-", "")) for c in flat.columns]
    keep = {"score_test_rank": "Rank", "candidate": "Model", "score_test_mean": "CV R² (mean)",
            "score_test_std": "CV R² (std)"}
    params = [c for c in flat.columns if c.startswith("param_regressor_")]
    board = flat[list(keep) + params].rename(columns=keep)
    board = board.rename(columns={c: c.replace("param_regressor_", "") for c in params})
    board["Model"] = board["Model"].str.replace("RegressorDF", "").str.replace("LGBM", "LightGBM")
    board.sort_values("Rank").to_csv(ART / "leaderboard.csv", index=False)
    best = selector.best_estimator_
    best_row = report.iloc[0]
    print(report.head(8).to_string())
    print(f"[{time.time() - t0:.0f}s] best model selected")

    # ---------------------------------------------------------------- 2. inspection
    inspect_sample = sample.subsample(
        iloc=np.random.default_rng(SEED).choice(len(sample), N_INSPECT, replace=False))
    inspector = LearnerInspector(model=best, n_jobs=-1).fit(inspect_sample)

    importance = inspector.feature_importance()
    importance.rename("importance").to_csv(ART / "importance.csv")

    synergy = matrix_to_df(inspector.feature_synergy_matrix(clustered=False))
    redundancy = matrix_to_df(inspector.feature_redundancy_matrix(clustered=False))
    synergy.to_csv(ART / "synergy.csv")
    redundancy.to_csv(ART / "redundancy.csv")

    # Symmetric redundancy, used to cluster features into a dendrogram
    # (distance = 1 - redundancy), the same idea as inspector.feature_redundancy_linkage().
    sym = matrix_to_df(inspector.feature_redundancy_matrix(symmetrical=True, clustered=False))
    sym.to_csv(ART / "redundancy_symmetric.csv")
    # (the dendrogram itself is drawn by make_figures.py)
    print(f"[{time.time() - t0:.0f}s] inspection done")

    # ---------------------------------------------------------------- 3. simulation
    simulator = UnivariateUpliftSimulator(model=best, sample=sample, n_jobs=-1)
    simulations = {}
    # explicit, sensible ranges to simulate over (FACET's automatic bounds fail
    # on narrow integer features such as min_players)
    sim_bounds = {
        "year": (1980, 2021), "min_players": (1, 5), "max_players": (1, 10),
        "play_time": (10, 240), "min_play_time": (10, 180), "min_age": (4, 16),
        "n_mechanics": (0, 10),
    }
    for feat in NUMERIC:
        integer = feat in ("min_players", "max_players", "min_age", "n_mechanics", "year")
        partitioner = (IntegerRangePartitioner(max_partitions=20) if integer
                       else ContinuousRangePartitioner(max_partitions=20))
        lo, hi = sim_bounds[feat]
        res = simulator.simulate_feature(feature_name=feat, partitioner=partitioner,
                                         lower_bound=lo, upper_bound=hi)
        d = res.data
        simulations[feat] = {
            "values": [float(v) for v in res.partitioner.partitions_],
            "frequencies": [int(f) for f in res.partitioner.frequencies_],
            "mean": d[res.COL_MEAN].astype(float).tolist(),
            "lower": d[res.COL_LOWER_BOUND].astype(float).tolist(),
            "upper": d[res.COL_UPPER_BOUND].astype(float).tolist(),
        }
    (ART / "simulations.json").write_text(json.dumps(simulations, indent=2))
    print(f"[{time.time() - t0:.0f}s] simulations done")

    # ---------------------------------------------------------------- save for the app
    joblib.dump(best.final_estimator.native_estimator, ART / "model.joblib")
    X = games[features]
    meta = {
        "features": features,
        "numeric": NUMERIC,
        "labels": labels,
        "defaults": {c: float(X[c].median()) for c in features},
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in features},
        "baseline_rating": float(games["rating"].mean()),
        "n_games": int(len(games)),
        "best_model": str(best.final_estimator.native_estimator.__class__.__name__),
        "best_params": {k: str(v) for k, v in best.final_estimator.native_estimator.get_params().items()
                        if k in ("n_estimators", "learning_rate", "num_leaves",
                                 "min_samples_leaf", "max_features")},
        "cv_r2_mean": float(best_row.iloc[1]) if np.isscalar(best_row.iloc[1]) else None,
    }
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.time() - t0:.0f}s] all artifacts saved to {ART}/")


if __name__ == "__main__":
    main()
