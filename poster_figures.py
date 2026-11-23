"""Figures sized for an A4 poster (printed ~3.55 in wide, 8-9 pt text). Output: figures/poster_*.png"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ART, FIG = Path("artifacts"), Path("figures")
INK, PAPER, TEAL, BRICK, MUTED, GRID = "#1B2A41", "#EDF0F4", "#14807C", "#C24A32", "#8692A6", "#D5DCE5"
font = Path("static/SchibstedGrotesk.ttf")
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(font)).get_name(), "font.size": 8,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "axes.labelsize": 8,
})
meta = json.loads((ART / "meta.json").read_text())
lab = lambda f: meta["labels"].get(f, f).replace(" (min)", "")  # noqa: E731
W = 3.55


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / f"poster_{name}.png", dpi=300)
    plt.close(fig)


# importance (top 8)
imp = pd.read_csv(ART / "importance.csv", index_col=0)["importance"].sort_values().tail(8)
fig, ax = plt.subplots(figsize=(W, 2.15))
ax.barh([lab(i) for i in imp.index], imp.values * 100, color=[MUTED] * 5 + [INK] * 3, height=0.62)
for y, v in enumerate(imp.values * 100):
    ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=7.5)
ax.set_xlabel("Share of total SHAP importance (%)")
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, imp.max() * 100 * 1.15)
save(fig, "importance")

# ablation
val = json.loads((ART / "validation.json").read_text())
ab = pd.DataFrame(val["ablation"]).iloc[1:]
fig, ax = plt.subplots(figsize=(W, 2.0))
ax.barh(ab["dropped"], ab["delta_r2"], color=[MUTED if k == "redundant" else INK for k in ab["kind"]], height=0.6)
ax.axvline(0, color=INK, lw=0.8)
for y, d in enumerate(ab["delta_r2"]):
    ax.text(d - 0.002, y, f"{d:+.3f}", va="center", ha="right", fontsize=7.5)
ax.set_xlim(-0.125, 0.004)
ax.set_xticks([-0.1, -0.05, 0]); ax.set_xticklabels(["−0.10", "−0.05", "0"])
ax.invert_yaxis()
ax.set_xlabel("Change in R² after removing the feature")
ax.tick_params(axis="y", length=0)
save(fig, "ablation")

# trends: popularity vs rating
games = pd.read_csv("data/games.csv")
mdl = joblib.load(ART / "model.joblib")
d = pd.read_csv("data/trends_popularity.csv").merge(games, left_on="game", right_on="name")
d["pred"] = mdl.predict(d[meta["features"]])
mass = d["game"].isin(["Exploding Kittens", "Cards Against Humanity", "Jenga", "Pictionary", "Twister",
                       "Connect Four", "Guess Who?", "Taboo", "Boggle"])
fig, ax = plt.subplots(figsize=(W, 2.5))
ax.vlines(d["popularity_vs_catan"], d["pred"], d["rating"], color=GRID, lw=1.2, zorder=1)
for m_, c, name in [(False, INK, "Hobby games"), (True, BRICK, "Mass-market games")]:
    sub = d[mass == m_]
    ax.scatter(sub["popularity_vs_catan"], sub["rating"], color=c, s=14, zorder=3, label=f"{name}, real")
    ax.scatter(sub["popularity_vs_catan"], sub["pred"], facecolor=PAPER, edgecolor=c, lw=1, s=13, zorder=3,
               label=f"{name}, predicted")
for n, xy, ha in [("Jenga", (0, -12), "center"), ("Gloomhaven", (-6, 0), "right"), ("Twister", (7, -3), "left")]:
    r = d[d["game"] == n].iloc[0]
    ax.annotate(n, (r.popularity_vs_catan, r.rating if n != "Twister" else r.pred), xytext=xy,
                textcoords="offset points", ha=ha, fontsize=7.5)
ax.set_xscale("log")
ticks = [2, 5, 10, 20, 50]
ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Google searches 2019–2026 (Catan = 100, log)")
ax.set_ylabel("BGG rating")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          handletextpad=0.2, columnspacing=0.8)
ax.set_ylim(4.2, 9.1)
save(fig, "trends")

# QR code to the repo
import qrcode
qr = qrcode.QRCode(border=1, box_size=12)
qr.add_data("https://github.com/surpiya19/tabletop-lens-xai"); qr.make(fit=True)
qr.make_image(fill_color=INK, back_color="white").save(FIG / "poster_qr.png")
print("ok")
