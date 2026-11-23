"""
What makes a board game great?
An explainable-AI webapp built on BCG X's FACET.

Run:  streamlit run app.py
(needs artifacts/ from analyze.py and validate.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shap
import streamlit as st

ART = Path("artifacts")
st.set_page_config(page_title="What makes a board game great?", page_icon="🎲", layout="wide")

# ------------------------------------------------------------------ look
INK, PAPER, CARD = "#1B2A41", "#EDF0F4", "#F7F9FB"
TEAL, BRICK, MEEPLE, MUTED, GRID = "#14807C", "#C24A32", "#E8A917", "#8692A6", "#D5DCE5"

pio.templates["lens"] = go.layout.Template(layout=dict(
    font=dict(family="Schibsted Grotesk, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[INK, TEAL, BRICK, MEEPLE, MUTED],
    xaxis=dict(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID),
    hoverlabel=dict(font_family="Schibsted Grotesk, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
))
pio.templates.default = "lens"

st.markdown(f"""
<style>
  h1, h2, h3 {{ letter-spacing: -0.015em; }}
  h1 {{ font-weight: 800 !important; font-size: 2.9rem !important; line-height: 1.05 !important; }}
  h2 {{ font-weight: 750 !important; font-size: 1.45rem !important; }}
  h3 {{ font-weight: 700 !important; font-size: 1.1rem !important; }}
  .block-container {{ padding-top: 3.4rem; max-width: 1280px; }}
  .byline {{ font-weight: 600; color: #0E625E; margin: -0.6rem 0 0.6rem; }}
  .lede {{ font-size: 1.05rem; color: #4A5872; max-width: 62ch; margin-top: -0.4rem; }}
  [data-testid="stSidebar"] h2 {{ font-size: 1.25rem !important; }}
  [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {{ color: #AEBBD0 !important; }}
  button[role="tab"] p {{ font-size: 0.98rem; font-weight: 600; }}
  [data-testid="stMetricValue"] {{ font-weight: 750; }}

  /* the score track: the one loud element */
  .track-wrap {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 6px;
                 padding: 1.2rem 1.4rem 3.3rem; margin: 0.4rem 0 1.4rem; }}
  .track-head {{ display: flex; align-items: baseline; gap: 1.2rem; flex-wrap: wrap; }}
  .track-score {{ font-weight: 800; font-size: 3.4rem; line-height: 1; color: {INK}; }}
  .track-score small {{ font-size: 1.1rem; font-weight: 600; color: {MUTED}; }}
  .track-note {{ color: #4A5872; font-size: 1rem; }}
  .track {{ position: relative; display: grid; grid-template-columns: repeat(10, 1fr);
            gap: 4px; margin-top: 3.6rem; }}
  .cell {{ height: 44px; border-radius: 3px; background: #DCE2EA; color: {MUTED};
           display: flex; align-items: flex-end; justify-content: flex-start;
           padding: 0 0 4px 7px; font-weight: 700; font-size: 0.85rem; }}
  .cell.lit {{ background: {INK}; color: #9FB0C9; }}
  .meeple {{ position: absolute; top: -50px; width: 42px; transform: translateX(-50%);
             filter: drop-shadow(0 2px 0 rgba(27,42,65,.35)); }}
  .pin {{ position: absolute; top: -12px; bottom: -12px; width: 2px; background: {INK}; }}
  .pin span {{ position: absolute; top: 100%; transform: translateX(-50%); white-space: nowrap;
               font-size: 0.8rem; color: #4A5872; padding-top: 3px; }}
  .pin.real {{ background: {TEAL}; }}
  .pin.real {{ bottom: -30px; }}
  .pin.real span {{ color: {TEAL}; font-weight: 700; }}
  .track-legend {{ margin-top: 2.6rem; font-size: 0.85rem; color: #4A5872; }}
  .callout {{ border-left: 3px solid {TEAL}; padding: 0.2rem 0 0.2rem 0.9rem; margin: 0.6rem 0; }}
  .callout.warn {{ border-color: {BRICK}; }}
</style>
""", unsafe_allow_html=True)

MEEPLE_SVG = (f"<svg class='meeple' viewBox='0 0 100 100' style='left:{{left}}%'>"
              f"<path fill='{MEEPLE}' stroke='{INK}' stroke-width='4' stroke-linejoin='round' "
              "d='M50 4c9 0 16 7 16 15 0 4-1 7-3 10l27 12c4 2 5 6 3 10l-2 3c-2 3-6 4-9 2l-14-6 "
              "13 38c1 4-1 8-5 8H64c-3 0-5-2-6-4l-8-17-8 17c-1 2-3 4-6 4H24c-4 0-6-4-5-8l13-38-14 6"
              "c-3 2-7 1-9-2l-2-3c-2-4-1-8 3-10l27-12c-2-3-3-6-3-10 0-8 7-15 16-15z'/></svg>")


# ------------------------------------------------------------------ data
@st.cache_resource
def load_model():
    model = joblib.load(ART / "model.joblib")
    return model, shap.TreeExplainer(model)


@st.cache_data
def load_artifacts():
    meta = json.loads((ART / "meta.json").read_text())
    sims = json.loads((ART / "simulations.json").read_text())
    imp = pd.read_csv(ART / "importance.csv", index_col=0)["importance"]
    syn = pd.read_csv(ART / "synergy.csv", index_col=0)
    red = pd.read_csv(ART / "redundancy.csv", index_col=0)
    board = pd.read_csv(ART / "leaderboard.csv")
    ho_path = ART / "heldout_predictions.csv"
    heldout = pd.read_csv(ho_path).set_index("name")["heldout_pred"] if ho_path.exists() else None
    val_path = ART / "validation.json"
    val = json.loads(val_path.read_text()) if val_path.exists() else None
    games = pd.read_csv("data/games.csv")
    return meta, sims, imp, syn, red, board, val, games, heldout


model, explainer = load_model()
meta, sims, importance, synergy, redundancy, leaderboard, val, games, heldout = load_artifacts()
FEATURES, LABELS = meta["features"], meta["labels"]
MECHS = [f for f in FEATURES if f.startswith("mech_")]
CATS = [f for f in FEATURES if f.startswith("cat_")]
BASE = meta["baseline_rating"]
PRESETS = ["Gloomhaven", "Wingspan", "Pandemic", "Catan", "Codenames", "Monopoly Cheaters Edition"]
PRESETS = [p for p in PRESETS if p in set(games["name"])]


def lab(f: str) -> str:
    return LABELS.get(f, f)


# ------------------------------------------------------------------ sidebar
DEFAULT = dict(year=2020, players=(2, 4), play_time=60, min_play=45, min_age=10,
               mechs=["mech_hand_management", "mech_set_collection"], cats=["cat_card_game"], n_mech=3)
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    name = st.session_state.preset
    if name == "Your own design":
        st.session_state.update(DEFAULT)
        return
    r = games.loc[games["name"] == name].iloc[0]
    mechs = [f for f in MECHS if r[f] == 1]
    pt = int(min(max(r["play_time"], 10), 480) // 5 * 5) or 10
    st.session_state.update(
        year=int(min(max(r["year"], 1980), 2021)),
        players=(int(min(r["min_players"], 12)), int(min(max(r["max_players"], r["min_players"]), 12))),
        play_time=pt, min_play=int(min(max(r["min_play_time"] // 5 * 5, 5), pt)),
        min_age=int(min(max(r["min_age"], 3), 18)), mechs=mechs,
        cats=[f for f in CATS if r[f] == 1], n_mech=int(max(r["n_mechanics"], len(mechs))))


with st.sidebar:
    st.markdown("## Your game")
    st.selectbox("Start from", ["Your own design"] + PRESETS, key="preset", on_change=apply_preset,
                 help="Load a real game's settings, then change them and watch the score move.")
    st.slider("Release year", 1980, 2021, key="year")
    st.slider("Player count", 1, 12, key="players")
    st.slider("Play time (minutes)", 10, 480, step=5, key="play_time")
    st.session_state.min_play = min(st.session_state.min_play, st.session_state.play_time)
    st.slider("Shortest play time (minutes)", 5, st.session_state.play_time, step=5, key="min_play")
    st.slider("Minimum age", 3, 18, key="min_age")
    st.multiselect("Mechanics", MECHS, format_func=lab, key="mechs")
    st.multiselect("Themes", CATS, format_func=lab, key="cats")
    st.session_state.n_mech = max(st.session_state.n_mech, len(st.session_state.mechs))
    st.slider("Total mechanics", len(st.session_state.mechs), 15, key="n_mech",
              help="Counts less common mechanics that aren't in the list above.")

ss = st.session_state
game = {f: 0.0 for f in FEATURES}
game.update(year=ss.year, min_players=ss.players[0], max_players=ss.players[1],
            play_time=ss.play_time, min_play_time=ss.min_play, min_age=ss.min_age,
            n_mechanics=ss.n_mech)
for f in ss.mechs + ss.cats:
    game[f] = 1.0
X = pd.DataFrame([game])[FEATURES]
pred = float(model.predict(X)[0])
sv = explainer(X)
contrib = pd.Series(sv.values[0], index=FEATURES)
expected = float(np.ravel(sv.base_values)[0])
real = None
if ss.preset != "Your own design":
    real = float(games.loc[games["name"] == ss.preset, "rating"].iloc[0])

# ------------------------------------------------------------------ header
st.title("What makes a board game great?")
st.markdown("<p class='byline'>Supriya Nannapaneni</p>", unsafe_allow_html=True)
st.markdown(f"<p class='lede'>Design a game on the left. A model trained on "
            f"{meta['n_games']:,} BoardGameGeek games predicts how players would rate it, "
            "and shows you why.</p>", unsafe_allow_html=True)

t_pred, t_what, t_inter, t_trust, t_reality, t_model = st.tabs(
    ["Your prediction", "What if", "Feature interactions", "Can we trust it?", "Reality check",
     "Model and data"])


def pos(r: float) -> float:
    """rating -> % across the track: cell i covers ratings i-1 to i, so cells 1..7 fill up to 7.0."""
    return float(np.clip(r * 10, 0, 100))


# ------------------------------------------------------------------ tab: prediction
with t_pred:
    pct = (games["rating"] < pred).mean() * 100
    cells = "".join(f"<div class='cell{' lit' if i <= pred else ''}'>{i}</div>" for i in range(1, 11))
    real_pin = (f"<div class='pin real' style='left:{pos(real)}%'><span>real rating {real:.2f}</span></div>"
                if real is not None else "")
    note = f"Higher than {pct:.0f}% of the games in the data."
    if real is not None:
        note += f" {ss.preset} is actually rated {real:.2f}."
        if heldout is not None and ss.preset in heldout.index:
            note += (f" It's in the training data, though: a model that never saw it predicts "
                     f"{heldout[ss.preset]:.2f}.")
    st.markdown(f"""
<div class='track-wrap'>
  <div class='track-head'>
    <div class='track-score'>{pred:.2f}<small> / 10</small></div>
    <div class='track-note'>{note}</div>
  </div>
  <div class='track'>
    {cells}
    {MEEPLE_SVG.format(left=pos(pred))}
    <div class='pin' style='left:{pos(BASE)}%'><span>average game {BASE:.2f}</span></div>
    {real_pin}
  </div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("### What moved the score")
        st.caption("Each bar is one design choice. Teal pushes the predicted rating up, red pulls it down. "
                   "Bars start from the average game and end at yours (SHAP values).")
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        shown, rest = top.iloc[:9], top.iloc[9:].sum()

        def describe(f):
            v = game[f]
            if f in MECHS or f in CATS:
                return f"{lab(f)}: {'yes' if v else 'no'}"
            return f"{lab(f)}: {v:g}"

        names = [describe(f) for f in shown.index] + ["Everything else"]
        vals = list(shown.values) + [rest]
        fig = go.Figure(go.Waterfall(
            orientation="h", y=names, x=vals, base=expected, measure=["relative"] * len(vals),
            increasing={"marker": {"color": TEAL}}, decreasing={"marker": {"color": BRICK}},
            connector={"line": {"color": GRID}}, text=[f"{v:+.2f}" for v in vals],
            textposition="outside", cliponaxis=False))
        fig.add_vline(x=expected, line_dash="dot", line_color=MUTED)
        fig.add_vline(x=pred, line_color=INK, line_width=2)
        for x, txt, anchor in [(expected, f"average {expected:.2f}", "right" if pred >= expected else "left"),
                               (pred, f"<b>your game {pred:.2f}</b>", "left" if pred >= expected else "right")]:
            fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom", text=txt, showarrow=False,
                               xanchor=anchor, xshift=4 if anchor == "left" else -4)
        cum = expected + np.cumsum(vals)
        lo, hi = min(expected, pred, cum.min()), max(expected, pred, cum.max())
        pad = (hi - lo) * 0.3 + 0.05
        fig.update_layout(height=430, margin=dict(t=30), showlegend=False,
                          xaxis=dict(title="Predicted rating", range=[lo - pad, hi + pad]),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        st.markdown("### Closest real games")
        z_mean, z_std = games[FEATURES].mean(), games[FEATURES].std().replace(0, 1)
        z, zx = (games[FEATURES] - z_mean) / z_std, (X.iloc[0] - z_mean) / z_std
        w = pd.Series(1.0, index=FEATURES)
        w[meta["numeric"]] = 2.0
        d = (((z - zx) ** 2) * w).sum(axis=1)
        if real is not None:
            d[games["name"] == ss.preset] = np.inf
        near = games.loc[d.nsmallest(6).index, ["name", "year", "rating"]]
        near.columns = ["Game", "Year", "Rating"]
        st.dataframe(near, hide_index=True, width="stretch")
        st.caption("Matched on player count, play time, age, year, mechanics and themes.")

# ------------------------------------------------------------------ tab: what if
with t_what:
    a, b = st.columns([1, 2.2])
    with a:
        st.markdown("### Change one thing")
        feat = st.radio("Feature", meta["numeric"], format_func=lab,
                        index=meta["numeric"].index("play_time"))
        s = sims[feat]
        xs = np.array(s["values"])
        grid = pd.DataFrame([game] * len(xs))[FEATURES]
        grid[feat] = xs
        if feat == "min_play_time":
            grid["play_time"] = np.maximum(grid["play_time"], xs)
        if feat == "play_time":
            grid["min_play_time"] = np.minimum(grid["min_play_time"], xs)
        mine = model.predict(grid)
        best_i = int(np.argmax(mine))
        st.markdown(f"<div class='callout'>For your game, the best {lab(feat).lower()} in this range is "
                    f"<b>{xs[best_i]:g}</b>, for a predicted <b>{mine[best_i]:.2f}</b> "
                    f"(now {pred:.2f}).</div>", unsafe_allow_html=True)
        st.caption("The grey line is FACET's uplift simulation: the feature is set to each value for "
                   "every game in the data and the predictions are averaged. Where your line and the grey "
                   "line differ in shape, your other choices are interacting with this one.")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.r_[xs, xs[::-1]], y=np.r_[BASE + np.array(s["upper"]), (BASE + np.array(s["lower"]))[::-1]],
            fill="toself", fillcolor="rgba(134,146,166,0.18)", line=dict(width=0),
            name="95% confidence band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=BASE + np.array(s["mean"]), mode="lines",
                                 line=dict(color=MUTED, width=2), name="Average game (FACET)"))
        fig.add_trace(go.Scatter(x=xs, y=mine, mode="lines+markers", line=dict(color=INK, width=3),
                                 marker=dict(size=6), name="Your game"))
        fig.add_trace(go.Scatter(x=[game[feat]], y=[pred], mode="markers", name="You are here",
                                 marker=dict(size=16, color=MEEPLE, line=dict(color=INK, width=2))))
        fig.update_layout(height=460, xaxis_title=lab(feat), yaxis_title="Predicted rating",
                          legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------------ tab: interactions
with t_inter:
    st.markdown("### Which features work together, and which overlap")
    c1, c2 = st.columns(2)
    c1.markdown("<div class='callout'><b>Synergy:</b> a feature's effect depends on another one. "
                "Neither tells the full story alone.</div>", unsafe_allow_html=True)
    c2.markdown("<div class='callout warn'><b>Redundancy:</b> two features carry the same information, "
                "so the model could use either.</div>", unsafe_allow_html=True)
    n = st.slider("Features shown (most important first)", 6, len(FEATURES), 12)
    keep = importance.sort_values(ascending=False).index[:n]

    def heat(m, title, colors):
        z = m.loc[keep, keep].values * 100
        np.fill_diagonal(z, np.nan)
        text = [["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(
            z=z, x=[lab(c) for c in keep], y=[lab(r) for r in keep], colorscale=[[0, CARD], [1, colors]],
            zmin=0, text=text, texttemplate="%{text}", xgap=2, ygap=2,
            hovertemplate="%{y} with %{x}: %{z:.1f}%<extra></extra>", colorbar=dict(title="%", thickness=10)))
        fig.update_layout(title=dict(text=title, font=dict(size=15)), height=600,
                          margin=dict(t=40, b=130), yaxis=dict(autorange="reversed"),
                          xaxis=dict(tickangle=-40, showgrid=False), plot_bgcolor=CARD)
        fig.update_yaxes(showgrid=False)
        return fig

    c1.plotly_chart(heat(synergy, "Synergy (% of row feature's effect)", TEAL), width="stretch",
                    config={"displayModeBar": False})
    c2.plotly_chart(heat(redundancy, "Redundancy (% of row feature's effect)", BRICK), width="stretch",
                    config={"displayModeBar": False})

    def top_pairs(m, k=4):
        st_ = m.loc[keep, keep].stack()
        return st_[[r != c for r, c in st_.index]].sort_values(ascending=False).head(k)

    with c1:
        for (r, c), v in top_pairs(synergy).items():
            st.write(f"{v:.0%} of **{lab(r)}**'s effect depends on **{lab(c)}**")
    with c2:
        for (r, c), v in top_pairs(redundancy).items():
            st.write(f"{v:.0%} of **{lab(r)}**'s information is also in **{lab(c)}**")

    st.markdown("### Redundancy clusters")
    a, b = st.columns([1.3, 1])
    a.image(str(ART / "redundancy_dendrogram.png"), width="stretch")
    b.markdown("Features that join further to the right share more information.\n\n"
               "**Wargame, the Simulation mechanic and play time** form one cluster: together they "
               "describe long, heavy strategy games. Shortest play time joins them, because it is "
               "partly a copy of play time.\n\n"
               "The *Can we trust it?* tab tests this by removing features from the cluster and "
               "retraining.")

# ------------------------------------------------------------------ tab: trust
with t_trust:
    if val is None:
        st.info("Run `python validate.py` to generate the validation results.")
    else:
        h = val["holdout"]
        st.markdown("### 1. Does it work on games it has never seen?")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test R²", f"{h['r2']:.2f}", f"{h['r2'] - h['cv_r2']:+.2f} vs. cross-validation",
                  delta_color="off")
        m2.metric("Typical error", f"±{h['mae']:.2f} stars")
        m3.metric("Guessing the average", f"±{h['baseline_mae']:.2f} stars")
        m4.metric("Error reduction", f"{1 - h['mae'] / h['baseline_mae']:.0%}")
        st.caption(f"Trained on {h['n_train']:,} games, tested on {h['n_test']:,} held-out games. "
                   "Test and cross-validation scores are close, so the model is not overfitting.")

        st.markdown("### 2. Is the redundancy real?")
        abl = pd.DataFrame(val["ablation"])
        a, b = st.columns([2, 1])
        with a:
            ab = abl.iloc[1:].copy()
            colors = [MUTED if k == "redundant" else INK for k in ab["kind"]]
            ylab = [f"{d}  ({s_:.0%} importance)" for d, s_ in zip(ab["dropped"], ab["importance_share"])]
            fig = go.Figure(go.Bar(
                y=ylab, x=ab["delta_r2"], orientation="h", marker_color=colors,
                text=[f"{d:+.3f}" for d in ab["delta_r2"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: R² change %{x:.4f}<extra></extra>"))
            fig.update_layout(height=360, xaxis=dict(title="Change in cross-validated R² after removing it",
                                                     range=[ab["delta_r2"].min() * 1.3, 0.008]),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            st.markdown(
                "If FACET is right that a feature is redundant, removing it and retraining should barely "
                "hurt, because another feature covers for it. Removing an important feature that is "
                "**not** redundant should hurt clearly.\n\n"
                "Grey bars are features FACET flagged as redundant. Dark bars are controls.")

        st.markdown("### 3. Do other explanation methods agree?")
        g, l, p = val["global_compare"], val["local_compare"], val["pdp_compare"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**SHAP vs permutation importance**")
            gs, gp = pd.Series(g["shap"]), pd.Series(g["permutation"])
            fig = go.Figure(go.Scatter(x=gs * 100, y=gp * 100, mode="markers", text=[lab(f) for f in gs.index],
                                       marker=dict(color=INK, size=8),
                                       hovertemplate="%{text}<br>SHAP %{x:.1f}%<br>permutation %{y:.1f}%<extra></extra>"))
            mx = max(gs.max(), gp.max()) * 105
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTED, dash="dot"),
                                     hoverinfo="skip"))
            for f in gs.nlargest(3).index:
                fig.add_annotation(x=gs[f] * 100, y=gp[f] * 100, text=lab(f), showarrow=False,
                                   xanchor="right", xshift=-8, font=dict(size=11))
            fig.update_layout(height=300, showlegend=False, xaxis_title="SHAP share (%)",
                              yaxis_title="Permutation share (%)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"Rank correlation {g['spearman']:.2f}; {g['top5_overlap']} of the top 5 features match.")
        with c2:
            st.markdown("**SHAP vs LIME, game by game**")
            hist = {int(k): v for k, v in l["top5_overlap_hist"].items()}
            ks = list(range(0, 6))
            fig = go.Figure(go.Bar(x=[f"{k}/5" for k in ks], y=[hist.get(k, 0) for k in ks], marker_color=INK))
            fig.update_layout(height=300, xaxis_title="Top-5 features shared", yaxis_title="Games")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"On {l['n_games']} test games: {l['top5_overlap_mean'] * 5:.1f} of 5 top features shared "
                       f"on average, and the direction of the top 3 SHAP features matches LIME "
                       f"{l['top3_sign_agreement']:.0%} of the time.")
        with c3:
            st.markdown("**FACET simulation vs partial dependence**")
            pp = p["play_time"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pp["values"], y=BASE + np.array(pp["pdp"]), mode="lines",
                                     line=dict(color=MUTED, width=8), name="Partial dependence (sklearn)"))
            fig.add_trace(go.Scatter(x=pp["values"], y=BASE + np.array(pp["facet"]), mode="lines+markers",
                                     line=dict(color=INK, width=2), name="FACET simulation"))
            fig.update_layout(height=300, xaxis_title="Play time (min)", yaxis_title="Predicted rating",
                              legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            diff = max(v["max_abs_diff"] for v in p.values())
            st.caption(f"The curves match exactly (largest gap {diff:.4f}): FACET's uplift simulation "
                       "is partial dependence, shifted to show change from the average rating.")

        st.markdown("### 4. Are the explanations stable?")
        s = val["stability"]
        a, b = st.columns([1.4, 1])
        with a:
            im, sd = pd.Series(s["importance_mean"]), pd.Series(s["importance_std"])
            order = im.sort_values(ascending=False).index[:10]
            fig = go.Figure(go.Bar(x=im[order] * 100, y=[lab(f) for f in order], orientation="h",
                                   marker_color=INK, error_x=dict(type="data", array=sd[order] * 100,
                                                                  color=MEEPLE, thickness=3, width=6)))
            fig.update_layout(height=340, yaxis=dict(autorange="reversed"),
                              xaxis_title=f"SHAP importance (%), mean and spread over {s['n_runs']} retrained models")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            st.metric("Importance ranking agreement", f"{s['importance_spearman_mean']:.2f}",
                      f"worst pair of runs {s['importance_spearman_min']:.2f}", delta_color="off")
            rows = []
            for kind, d in (("Redundancy", s["redundancy"]), ("Synergy", s["synergy"])):
                for pair, v in d.items():
                    a_, b_ = pair.split("|")
                    rows.append({"Pair": f"{lab(a_)} / {lab(b_)}", "Type": kind,
                                 "Mean": f"{v['mean']:.0%}", "Range": f"{v['min']:.0%} to {v['max']:.0%}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Each run retrains on a different 80% of the games and re-runs FACET on "
                       f"{s['n_inspect']:,} of them.")

# ------------------------------------------------------------------ tab: reality check (Google Trends)
MASS_MARKET = {"Exploding Kittens", "Cards Against Humanity", "Jenga", "Pictionary", "Twister",
               "Connect Four", "Guess Who?", "Taboo", "Boggle"}
CONTAMINATED = {"Pandemic": "2020 peak comes from COVID searches",
                "7 Wonders": "2007 spike is the Seven Wonders of the World vote, before the game existed"}
TERM_TO_FEATURE = {"party game": "cat_party_game", "wargame": "cat_wargame",
                   "cooperative board game": "mech_cooperative_game",
                   "deck building game": None, "worker placement": None}


@st.cache_data
def load_trends():
    pop_path, yr_path = Path("data/trends_popularity.csv"), Path("data/trends_yearly.csv")
    if not (pop_path.exists() and yr_path.exists()):
        return None
    pop = pd.read_csv(pop_path)
    yearly = pd.read_csv(yr_path, index_col=0)
    yearly.columns = yearly.columns.astype(int)
    d = pop.merge(games, left_on="game", right_on="name")
    d["pred"] = model.predict(d[FEATURES])
    d["mass"] = d["game"].isin(MASS_MARKET)
    sv_all = explainer(games[FEATURES]).values
    shap_on = {f: float(sv_all[(games[f] == 1).values, FEATURES.index(f)].mean())
               for f in TERM_TO_FEATURE.values() if f}
    return d, yearly, shap_on


with t_reality:
    tr = load_trends()
    if tr is None:
        st.info("Google Trends data not found. See trends/README.md for how to collect it.")
    else:
        from scipy.stats import spearmanr

        d, yearly, shap_on = tr
        st.markdown("### Does the model agree with the real world?")
        st.markdown(
            "<p class='lede'>Google search interest is an outside measure the model never saw. It tracks "
            "popularity, not quality, so it can't prove the model right. It can show what kind of "
            "\"great\" the model has learned.</p>", unsafe_allow_html=True)

        # ---- 1. popularity vs rating
        st.markdown("### 1. Popular isn't the same as well-rated")
        r_real = spearmanr(d["popularity_vs_catan"], d["rating"]).statistic
        r_pred = spearmanr(d["popularity_vs_catan"], d["pred"]).statistic
        r_rp = spearmanr(d["rating"], d["pred"]).statistic
        a, b = st.columns([2, 1])
        with a:
            fig = go.Figure()
            for _, r in d.iterrows():
                fig.add_trace(go.Scatter(x=[r.popularity_vs_catan] * 2, y=[r.pred, r.rating], mode="lines",
                                         line=dict(color=GRID, width=2), hoverinfo="skip", showlegend=False))
            for mass, name, color in [(False, "Hobby games", INK), (True, "Mass-market games", BRICK)]:
                sub = d[d["mass"] == mass]
                fig.add_trace(go.Scatter(
                    x=sub["popularity_vs_catan"], y=sub["rating"], mode="markers", name=f"{name}: real rating",
                    marker=dict(color=color, size=10), text=sub["game"],
                    hovertemplate="%{text}<br>searches %{x:.1f} (Catan = 100)<br>real rating %{y:.2f}<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=sub["popularity_vs_catan"], y=sub["pred"], mode="markers", name=f"{name}: model prediction",
                    marker=dict(color=PAPER, size=9, line=dict(color=color, width=2)), text=sub["game"],
                    hovertemplate="%{text}<br>predicted %{y:.2f}<extra></extra>"))
            for n in ["Jenga", "Gloomhaven", "Codenames", "Twister", "Kingdomino"]:
                r = d[d["game"] == n].iloc[0]
                fig.add_annotation(x=np.log10(r.popularity_vs_catan), y=r.rating, text=n, showarrow=False,
                                   yshift=14, font=dict(size=11, color="#4A5872"))
            fig.update_layout(height=460, xaxis=dict(type="log", title="Google search interest, 2019–2026 (Catan = 100, log scale)"),
                              yaxis_title="BoardGameGeek rating", legend=dict(orientation="h", y=-0.22, x=0))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            st.metric("Searches vs real rating (rank correlation)", f"{r_real:+.2f}")
            st.metric("Searches vs model prediction", f"{r_pred:+.2f}")
            st.metric("Model vs real rating", f"{r_rp:+.2f}")
            st.markdown(
                "The most-searched games tend to have **lower** BoardGameGeek ratings, and the model "
                "reproduces that. Jenga is the most searched game here and is rated 5.62.\n\n"
                "**What it means:** the model has learned what BoardGameGeek's hobbyist community values. "
                "\"Great\" in this app means great to enthusiasts, not popular with everyone.")
        st.caption(f"{len(d)} games, each measured as a Google Trends board-game topic in the same request as "
                   "Catan, so all share one scale. Filled dots are real ratings, hollow dots the model's "
                   "predictions. These games were in the training data.")

        # ---- 2. hype test
        st.markdown("### 2. Is the release-year effect just hype?")
        games_y = [n for n in yearly.index if not n.startswith("term:")]
        rows = []
        for n in games_y:
            rel = int(games.loc[games["name"] == n, "year"].iloc[0])
            s_ = yearly.loc[n, rel:2024]
            rows.append({"Game": n, "Released": rel, "Peak year": int(s_.idxmax()),
                         "Years to peak": int(s_.idxmax()) - rel, "2024 vs peak": s_.loc[2024] / s_.max(),
                         "Note": CONTAMINATED.get(n, "")})
        hype = pd.DataFrame(rows)
        clean = hype[~hype["Game"].isin(CONTAMINATED)]
        faded = clean[clean["2024 vs peak"] < 0.5]
        a, b = st.columns([2, 1])
        with a:
            pick = st.multiselect("Games", games_y, default=["Gloomhaven", "Cards Against Humanity",
                                                             "Spirit Island", "King of Tokyo"])
            fig = go.Figure()
            palette = [INK, TEAL, BRICK, MEEPLE, MUTED, "#5B6FA8"]
            for i, n in enumerate(pick):
                rel = int(games.loc[games["name"] == n, "year"].iloc[0])
                s_ = yearly.loc[n, rel:2024]
                fig.add_trace(go.Scatter(x=s_.index - rel, y=s_.values / s_.max() * 100, mode="lines+markers",
                                         name=n, line=dict(color=palette[i % len(palette)], width=2.5)))
            fig.update_layout(height=380, xaxis_title="Years since release", yaxis_title="Search interest (% of own peak)",
                              legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            st.metric("Games that faded below half their peak by 2024", f"{len(faded)} of {len(clean)}")
            st.metric("Typical years from release to peak", f"{clean['Years to peak'].median():.0f}")
            st.markdown(
                "If newer games were rated higher only because of launch hype, searches should spike at "
                "release and fade. They mostly don't: most games keep gaining interest for years, and many "
                "peaked in 2020 during the lockdown board game boom.\n\n"
                "**What it means:** the search data does **not** support hype as the explanation for the "
                "release-year effect. Its cause is still open.")
        with st.expander("Per-game numbers"):
            show = hype.copy()
            show["2024 vs peak"] = show["2024 vs peak"].map(lambda v: f"{v:.0%}")
            st.dataframe(show, hide_index=True, width="stretch")

        # ---- 3. public taste vs the model
        st.markdown("### 3. Public taste vs what the model rewards")
        a, b = st.columns([2, 1])
        terms = [t for t in yearly.index if t.startswith("term:") and t not in ("term:board game", "term:boardgamegeek")]
        with a:
            fig = go.Figure()
            colors = {"term:deck building game": MUTED, "term:worker placement": "#5B6FA8",
                      "term:cooperative board game": TEAL, "term:party game": MEEPLE, "term:wargame": BRICK}
            for t in terms:
                s_ = yearly.loc[t, 2004:2024]
                fig.add_trace(go.Scatter(x=s_.index, y=s_.values, mode="lines", name=t[5:],
                                         line=dict(color=colors.get(t, INK), width=2.5)))
            fig.update_layout(height=380, xaxis_title="Year", yaxis_title="Search interest (yearly mean, own scale)",
                              legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            rows = []
            for t, f in TERM_TO_FEATURE.items():
                s_ = yearly.loc["term:" + t]
                early, late = s_.loc[2005:2007].mean(), s_.loc[2022:2024].mean()
                trend = "rising" if late > early * 1.3 + 1 else ("falling" if late < early * 0.7 else "flat")
                rows.append({"Search term": t, "Searches": trend,
                             "Model": "not a feature" if f is None else f"{shap_on[f]:+.2f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.markdown(
                "Wargames are the clearest split: public searches have fallen since 2004, yet the model adds "
                f"{shap_on['cat_wargame']:+.2f} to a wargame's rating. Party games are searched steadily, and the "
                f"model gives them {shap_on['cat_party_game']:+.2f}. Again, the model follows hobbyist taste.")
        st.caption("Searches compares the 2005–07 average with 2022–24. Model = average SHAP value for "
                   "games that have the feature. The 2012–14 spike in "
                   "\"wargame\" is a video game series of the same name.")

        with st.expander("About the Google Trends data"):
            st.markdown(
                "- Collected on 23 Nov 2026 from trends.google.com, worldwide, monthly, Jan 2004 – Oct 2026.\n"
                "- Games use Google's board-game **topics**, not raw keywords, so \"Pandemic\" means the game. "
                "Six games (Azul, Dominion, Scythe, Root, Brass: Birmingham, Operation) had no clean topic "
                "and were left out.\n"
                "- Popularity uses Jan 2019 – Oct 2026, when every game in the set existed, with Catan in "
                "every request as a common reference.\n"
                "- Every search term roughly doubles in 2025–2026, which looks like a change in Google's "
                "measurement, so yearly charts stop at 2024.\n"
                "- Newer topics start recording only when Google creates them, so early years can read 0 "
                "even after a game's release.\n"
                "- Search interest measures attention, not enjoyment, and mixes in other meanings of a name.")

# ------------------------------------------------------------------ tab: model
with t_model:
    a, b = st.columns(2)
    with a:
        st.markdown("### Model selection")
        core = ["Rank", "Model", "CV R² (mean)", "CV R² (std)"]
        board = leaderboard[core].copy()
        board["Settings"] = leaderboard.drop(columns=core).apply(
            lambda r: ", ".join(f"{k}={v:g}" for k, v in r.items() if pd.notna(v)), axis=1)
        st.dataframe(board.round(3), hide_index=True, width="stretch")
        st.caption("FACET LearnerSelector: Random Forest vs LightGBM, 5-fold cross-validation repeated twice.")
    with b:
        st.markdown("### Overall feature importance")
        imp = importance.sort_values().tail(12)
        fig = go.Figure(go.Bar(x=imp.values * 100, y=[lab(i) for i in imp.index], orientation="h",
                               marker_color=[MUTED] * (len(imp) - 3) + [INK] * 3))
        fig.update_layout(height=400, xaxis_title="Share of total mean |SHAP| (%)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("### What this model can't tell you")
    st.markdown(
        "- **It finds patterns, not causes.** Newer games score higher partly because BoardGameGeek users "
        "rate the latest releases generously, not because 2021 is a better year to publish.\n"
        "- **Important things are missing.** Art, publisher, marketing, designer reputation and rules "
        "complexity aren't in the data. About half of the variation in ratings is unexplained.\n"
        "- **Only games that found an audience are included.** Every game here has 200+ ratings.\n"
        "- **Unusual designs are guesses.** A 12-player, 8-hour party game is unlike anything in the data.\n"
        "- **Explanations describe the model, not the world.** SHAP and FACET show what the model learned, "
        "biases included.")
    st.caption("By Supriya Nannapaneni. Data: TidyTuesday 2022-01-25 (BoardGameGeek) and Google Trends. Built with FACET (BCG X), SHAP, LightGBM and Streamlit.")
