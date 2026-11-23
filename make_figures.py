"""
Step 4: draw the static figures, in the app's style.

  artifacts/redundancy_dendrogram.png   used by the app
  figures/*.png                         high-resolution (300 dpi) figures for a poster or report

Run:  python make_figures.py        (after analyze.py and validate.py)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import squareform

ART, FIG = Path("artifacts"), Path("figures")
FIG.mkdir(exist_ok=True)
INK, PAPER, TEAL, BRICK, MEEPLE, MUTED, GRID = ("#1B2A41", "#EDF0F4", "#14807C", "#C24A32",
                                               "#E8A917", "#8692A6", "#D5DCE5")

font = Path("static/Archivo.ttf")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 15, "axes.titlelocation": "left",
    "axes.titlepad": 14, "font.size": 11,
})

meta = json.loads((ART / "meta.json").read_text())
LABELS = meta["labels"]
lab = lambda f: LABELS.get(f, f)  # noqa: E731
BASE = meta["baseline_rating"]


def save(fig, name, app_copy=False):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    if app_copy:
        fig.savefig(ART / f"{name}.png", dpi=160)
    plt.close(fig)


# ---- redundancy dendrogram (app + poster)
sym = pd.read_csv(ART / "redundancy_symmetric.csv", index_col=0)
dist = (1 - sym.clip(0, 1)).values
np.fill_diagonal(dist, 0)
Z = linkage(squareform((dist + dist.T) / 2, checks=False), method="single")
set_link_color_palette([BRICK])
fig, ax = plt.subplots(figsize=(7.5, 8.5))
dendrogram(Z, labels=[lab(f) for f in sym.index], orientation="left", ax=ax,
           color_threshold=0.85, above_threshold_color=MUTED)
ax.set_xlabel("Distance (1 − redundancy): further right = more overlap")
ax.set_title("Features that carry the same information")
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
save(fig, "redundancy_dendrogram", app_copy=True)

# ---- global importance
imp = pd.read_csv(ART / "importance.csv", index_col=0)["importance"].sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([lab(i) for i in imp.index], imp.values * 100,
        color=[MUTED] * (len(imp) - 3) + [INK] * 3, height=0.65)
for y, v in enumerate(imp.values * 100):
    ax.text(v + 0.4, y, f"{v:.0f}%", va="center", fontsize=10)
ax.set_xlabel("Share of total mean |SHAP| (%)")
ax.set_title("Release year, mechanics and play time drive the predictions")
ax.tick_params(axis="y", length=0)
save(fig, "importance")

# ---- what-if: FACET uplift simulation for 3 features
sims = json.loads((ART / "simulations.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, f in zip(axes, ["play_time", "n_mechanics", "max_players"]):
    s = sims[f]
    x = np.array(s["values"])
    ax.fill_between(x, BASE + np.array(s["lower"]), BASE + np.array(s["upper"]), color=MUTED, alpha=0.3, lw=0)
    ax.plot(x, BASE + np.array(s["mean"]), color=INK, lw=2.5)
    ax.axhline(BASE, color=MUTED, ls=":", lw=1)
    ax.set_xlabel(lab(f))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Predicted rating")
axes[0].set_title("What if every game changed one thing? (FACET simulation)", loc="left")
save(fig, "what_if")

val_path = ART / "validation.json"
if val_path.exists():
    val = json.loads(val_path.read_text())

    # ---- ablation
    ab = pd.DataFrame(val["ablation"]).iloc[1:]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [MUTED if k == "redundant" else INK for k in ab["kind"]]
    ax.barh(ab["dropped"], ab["delta_r2"], color=colors, height=0.6)
    ax.axvline(0, color=INK, lw=1)
    lim = ab["delta_r2"].abs().max()
    for y, (d, s) in enumerate(zip(ab["delta_r2"], ab["importance_share"])):
        ax.text(d - lim * 0.03 if d < 0 else d + lim * 0.03, y, f"{d:+.3f}  (importance {s:.0%})",
                va="center", ha="right" if d < 0 else "left", fontsize=9.5)
    ax.set_xlim(-lim * 2.1, lim * 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Change in cross-validated R² after removing the feature")
    ax.set_title("Dropping a redundant feature barely hurts accuracy")
    ax.tick_params(axis="y", length=0)
    save(fig, "ablation")

    # ---- method agreement
    g = val["global_compare"]
    gs, gp = pd.Series(g["shap"]) * 100, pd.Series(g["permutation"]) * 100
    fig, ax = plt.subplots(figsize=(5, 5))
    mx = max(gs.max(), gp.max()) * 1.08
    ax.plot([0, mx], [0, mx], ls=":", color=MUTED)
    ax.scatter(gs, gp, color=INK, s=28, zorder=3)
    for f in gs.nlargest(4).index:
        ax.annotate(lab(f), (gs[f], gp[f]), xytext=(-6, 4), textcoords="offset points", ha="right", fontsize=9.5)
    ax.set_xlabel("SHAP importance (%)")
    ax.set_ylabel("Permutation importance (%)")
    ax.set_title(f"SHAP and permutation importance agree (rank correlation {g['spearman']:.2f})")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "shap_vs_permutation")

# ---- Google Trends: popularity vs rating
pop_path = Path("data/trends_popularity.csv")
if pop_path.exists():
    import joblib
    from scipy.stats import spearmanr
    games = pd.read_csv("data/games.csv")
    mdl = joblib.load(ART / "model.joblib")
    d = pd.read_csv(pop_path).merge(games, left_on="game", right_on="name")
    d["pred"] = mdl.predict(d[meta["features"]])
    mass = d["game"].isin(["Exploding Kittens", "Cards Against Humanity", "Jenga", "Pictionary", "Twister",
                           "Connect Four", "Guess Who?", "Taboo", "Boggle"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.vlines(d["popularity_vs_catan"], d["pred"], d["rating"], color=GRID, lw=2, zorder=1)
    for m_, c, lab_ in [(False, INK, "Hobby games"), (True, BRICK, "Mass-market games")]:
        sub = d[mass == m_]
        ax.scatter(sub["popularity_vs_catan"], sub["rating"], color=c, s=40, zorder=3, label=f"{lab_}: real rating")
        ax.scatter(sub["popularity_vs_catan"], sub["pred"], facecolor=PAPER, edgecolor=c, lw=1.6, s=36, zorder=3,
                   label=f"{lab_}: model prediction")
    for n in ["Jenga", "Gloomhaven", "Codenames", "Twister", "Kingdomino"]:
        r = d[d["game"] == n].iloc[0]
        ax.annotate(n, (r.popularity_vs_catan, r.rating), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9.5)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [2, 5, 10, 20, 50]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Google search interest 2019–2026 (Catan = 100, log scale)")
    ax.set_ylabel("BoardGameGeek rating")
    rho = spearmanr(d["popularity_vs_catan"], d["rating"]).statistic
    ax.set_title(f"More-searched games are rated lower (rank correlation {rho:+.2f})")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "trends_popularity_vs_rating")

print("Saved:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
