"""
Step 3 (optional but recommended): validate the explanations.

  1. Held-out test       - does the model generalise to games it never saw?
  2. Redundancy ablation - if FACET says two features are redundant, removing one
                           should barely hurt accuracy. Removing a non-redundant,
                           important feature should hurt clearly (control).
  3. Method comparison   - do other XAI methods tell the same story?
                           a) global: SHAP vs permutation importance
                           b) local:  SHAP vs LIME on 50 test games
                           c) FACET uplift simulation vs sklearn partial dependence
  4. Stability           - retrain on 5 different 80% subsamples; do the importance
                           ranking and the key synergy/redundancy values stay put?

Output: artifacts/validation.json

Run:  python validate.py        (takes ~5 minutes; run analyze.py first)
"""
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
import shap

from facet.data import Sample
from facet.inspection import LearnerInspector
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

warnings.filterwarnings("ignore")
ART = Path("artifacts")
SEED = 42

meta = json.loads((ART / "meta.json").read_text())
FEATURES = meta["features"]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              random_state=SEED, verbose=-1, n_jobs=1)
for k in ("n_estimators", "learning_rate", "num_leaves"):   # use whatever analyze.py chose
    if k in meta["best_params"]:
        PARAMS[k] = type(PARAMS[k])(float(meta["best_params"][k]))

games = pd.read_csv("data/games.csv")
X, y = games[FEATURES], games["rating"]
out = {}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 1. held-out test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
model = LGBMRegressor(**PARAMS).fit(X_tr, y_tr)
pred = model.predict(X_te)
out["holdout"] = {
    "n_train": len(X_tr), "n_test": len(X_te),
    "r2": r2_score(y_te, pred),
    "mae": mean_absolute_error(y_te, pred),
    "baseline_mae": mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())),
    "cv_r2": meta["cv_r2_mean"],
}
log(f"holdout {out['holdout']}")

# ---------------------------------------------------------------- 2. ablation
imp = pd.read_csv(ART / "importance.csv", index_col=0)["importance"]
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
experiments = [
    ("None (full model)", [], "reference"),
    ("Min play time", ["min_play_time"], "redundant"),
    ("Simulation mechanic", ["mech_simulation"], "redundant"),
    ("Wargame category", ["cat_wargame"], "redundant"),
    ("Wargame + Simulation", ["cat_wargame", "mech_simulation"], "redundant"),
    ("Play time", ["play_time"], "redundant"),
    ("Number of mechanics", ["n_mechanics"], "control"),
    ("Release year", ["year"], "control"),
]
abl = []
for name, drop, kind in experiments:
    cols = [f for f in FEATURES if f not in drop]
    s = cross_val_score(LGBMRegressor(**PARAMS), X[cols], y, cv=cv, scoring="r2", n_jobs=-1)
    abl.append({"dropped": name, "kind": kind, "r2": float(s.mean()), "r2_std": float(s.std()),
                "importance_share": float(imp[drop].sum()) if drop else 0.0})
ref = abl[0]["r2"]
for a in abl:
    a["delta_r2"] = a["r2"] - ref
out["ablation"] = abl
log("ablation done")

# ---------------------------------------------------------------- 3. method comparison
# a) global: mean |SHAP| vs permutation importance, both on the test set
sv = shap.TreeExplainer(model)(X_te).values
shap_imp = pd.Series(np.abs(sv).mean(0), index=FEATURES)
perm = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=SEED, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=FEATURES)
rho = spearmanr(shap_imp, perm_imp).statistic
top_s, top_p = set(shap_imp.nlargest(5).index), set(perm_imp.nlargest(5).index)
out["global_compare"] = {
    "spearman": float(rho), "top5_overlap": len(top_s & top_p),
    "shap": (shap_imp / shap_imp.sum()).round(4).to_dict(),
    "permutation": (perm_imp.clip(lower=0) / perm_imp.clip(lower=0).sum()).round(4).to_dict(),
}
log(f"SHAP vs permutation spearman={rho:.2f}")

# b) local: SHAP vs LIME on 50 test games
lime_exp = LimeTabularExplainer(X_tr.values, feature_names=FEATURES, mode="regression",
                                discretize_continuous=True, random_state=SEED)
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_te), 50, replace=False)
overlaps, sign_agree, rhos = [], [], []
for i in idx:
    e = lime_exp.explain_instance(X_te.values[i], model.predict, num_features=len(FEATURES),
                                  num_samples=2000)
    w = pd.Series(dict(e.as_map()[1])).sort_index()   # label 1 = the regression output
    w.index = [FEATURES[j] for j in w.index]
    w = w.reindex(FEATURES).fillna(0)
    s = pd.Series(sv[i], index=FEATURES)
    ts, tl = set(s.abs().nlargest(5).index), set(w.abs().nlargest(5).index)
    overlaps.append(len(ts & tl) / 5)
    top3 = s.abs().nlargest(3).index
    sign_agree.append(float((np.sign(s[top3]) == np.sign(w[top3])).mean()))
    rhos.append(spearmanr(s.abs(), w.abs()).statistic)
out["local_compare"] = {
    "n_games": len(idx), "top5_overlap_mean": float(np.mean(overlaps)),
    "top3_sign_agreement": float(np.mean(sign_agree)),
    "rank_corr_mean": float(np.nanmean(rhos)),
    "top5_overlap_hist": pd.Series(np.round(np.array(overlaps) * 5).astype(int))
                          .value_counts().sort_index().to_dict(),
}
log(f"LIME {out['local_compare']}")

# c) FACET uplift simulation vs sklearn partial dependence (same full-data model)
full_model = joblib.load(ART / "model.joblib")
sims = json.loads((ART / "simulations.json").read_text())
pdp_cmp = {}
for feat in ("play_time", "n_mechanics", "year"):
    vals = np.array(sims[feat]["values"])
    pd_res = partial_dependence(full_model, X, [FEATURES.index(feat)],
                                custom_values={FEATURES.index(feat): vals}, kind="average")
    pdp = pd_res["average"][0] - y.mean()          # FACET reports uplift vs mean rating
    facet = np.array(sims[feat]["mean"])
    pdp_cmp[feat] = {"values": vals.tolist(), "facet": facet.tolist(), "pdp": pdp.tolist(),
                     "max_abs_diff": float(np.max(np.abs(pdp - facet)))}
out["pdp_compare"] = pdp_cmp
log("PDP: " + ", ".join(f"{k} max diff {v['max_abs_diff']:.4f}" for k, v in pdp_cmp.items()))

# ---------------------------------------------------------------- 4. stability
N_RUNS, N_INSPECT = 5, 1500
KEY_RED = [("play_time", "min_play_time"), ("cat_wargame", "mech_simulation"),
           ("play_time", "cat_wargame")]
KEY_SYN = [("min_age", "year"), ("min_play_time", "year"), ("cat_fighting", "year")]
runs = []
for r in range(N_RUNS):
    sub = games.sample(frac=0.8, random_state=100 + r)
    pipe = RegressorPipelineDF(regressor=LGBMRegressorDF(**{**PARAMS, "random_state": 100 + r}))
    pipe.fit(sub[FEATURES], sub["rating"])
    insp_rows = sub.sample(N_INSPECT, random_state=200 + r)
    smp = Sample(observations=insp_rows[FEATURES + ["rating"]], target_name="rating")
    insp = LearnerInspector(model=pipe, n_jobs=-1).fit(smp)
    fi = insp.feature_importance()
    red = insp.feature_redundancy_matrix(clustered=False)
    syn = insp.feature_synergy_matrix(clustered=False)
    red = pd.DataFrame(red.values, index=red.names[0], columns=red.names[1])
    syn = pd.DataFrame(syn.values, index=syn.names[0], columns=syn.names[1])
    runs.append({"importance": fi,
                 "red": {f"{a}|{b}": float(red.loc[a, b]) for a, b in KEY_RED},
                 "syn": {f"{a}|{b}": float(syn.loc[a, b]) for a, b in KEY_SYN}})
    log(f"stability run {r + 1}/{N_RUNS}")

imps = pd.DataFrame([x["importance"] for x in runs])
pair_rho = [spearmanr(imps.iloc[a], imps.iloc[b]).statistic for a, b in combinations(range(N_RUNS), 2)]
top5_counts = pd.Series([f for _, row in imps.iterrows() for f in row.nlargest(5).index]).value_counts()


def summarise(key):
    df = pd.DataFrame([x[key] for x in runs])
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std()),
                "min": float(df[k].min()), "max": float(df[k].max())} for k in df.columns}


out["stability"] = {
    "n_runs": N_RUNS, "n_inspect": N_INSPECT,
    "importance_spearman_mean": float(np.mean(pair_rho)),
    "importance_spearman_min": float(np.min(pair_rho)),
    "top5_frequency": {k: int(v) for k, v in top5_counts.items()},
    "importance_mean": imps.mean().round(4).to_dict(),
    "importance_std": imps.std().round(4).to_dict(),
    "redundancy": summarise("red"),
    "synergy": summarise("syn"),
}
log(f"stability spearman mean={np.mean(pair_rho):.3f}")

(ART / "validation.json").write_text(json.dumps(out, indent=2, default=float))
log("saved artifacts/validation.json")
