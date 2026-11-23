# What makes a board game great? An explainable-AI webapp

An explainable-AI webapp that predicts how BoardGameGeek players would rate a board game you
design, explains why, and tests whether those explanations can be trusted. Built on
[FACET](https://github.com/BCG-X-Official/facet) by BCG X, with SHAP, LightGBM and Streamlit.

## The app

| Tab | What it shows | Method |
|---|---|---|
| Your prediction | Predicted rating on a 1–10 score track, what moved it, closest real games | SHAP waterfall |
| What if | Change one feature: your game's curve vs. the average game | FACET `UnivariateUpliftSimulator` |
| Feature interactions | Synergy and redundancy heatmaps, redundancy clusters | FACET `LearnerInspector` |
| Can we trust it? | Held-out test, redundancy ablation, LIME / permutation / partial dependence comparison, stability | `validate.py` |
| Reality check | Google search popularity vs rating, a hype test, genre trends | Google Trends (`trends/`) |
| Model and data | Model comparison, global importance, limitations | FACET `LearnerSelector` |

Use **Start from** in the sidebar to load a real game (Gloomhaven, Wingspan, Pandemic, Catan,
Codenames, Monopoly Cheaters Edition). The track then also shows its real rating.

## Run the app

Python 3.11–3.13 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

All results are pre-computed in `artifacts/`, so the app starts immediately.

## Re-run the analysis (optional, ~10 minutes)

```bash
python -m pip install -r requirements-analysis.txt
python prepare_data.py     # download BGG data, build features       -> data/games.csv
python analyze.py          # FACET model selection, inspection, simulation -> artifacts/
python validate.py         # the four validation checks                -> artifacts/validation.json
python make_figures.py     # styled figures (poster/report, 300 dpi)   -> figures/
```

## Key findings

- **Release year (25%), number of mechanics (16%) and play time (13%)** drive most predictions.
- **Wargame, the Simulation mechanic and play time are redundant**: they all describe long, heavy strategy games.
- **The redundancy is real.** Removing play time (13% importance) costs only 0.010 R²; removing
  number of mechanics (16%, not redundant) costs 0.022; removing release year costs 0.10.
- **Other methods agree.** SHAP vs permutation importance: rank correlation 0.89, same top 5.
  SHAP vs LIME: same direction for the top 3 features 94% of the time. FACET's simulation equals
  partial dependence exactly.
- **Stable.** Over 5 retrained models, importance rankings correlate at 0.96 on average.
- **Accuracy.** Held-out R² = 0.52; typical error ±0.40 points, 34% better than guessing the average.
- **Google Trends check.** More-searched games have *lower* BGG ratings (rank correlation −0.45), and the
  model reproduces this (−0.52): it has learned hobbyist taste, not mass popularity. Searches mostly keep
  rising for years after release (only 3 of 14 games faded below half their peak by 2024), so launch
  hype does not explain the release-year effect.

## Project structure

```
prepare_data.py   analyze.py   validate.py   make_figures.py   app.py
data/games.csv          cleaned modelling table (8,055 games x 29 features)
data/trends_*.csv       Google Trends data (see trends/README.md)
trends/                 browser script used to collect it, and its caveats
artifacts/              model + FACET + validation results used by the app
figures/                300 dpi PNGs for the poster and report
static/Archivo.ttf      the app's typeface (SIL Open Font License)
.streamlit/config.toml  theme
```

## Deploy for free

Push the folder to a public GitHub repo, then on share.streamlit.io choose "New app",
pick the repo and `app.py`. It installs `requirements.txt` and gives you a public link.

## Limitations

Correlation, not causation: the release-year effect most likely reflects BGG's rating audience,
not game quality. Rules complexity, art, publisher and marketing are missing, and about half the
variation is unexplained. Only games with 200+ ratings are included. Unusual designs are
extrapolations. The explanations describe the model, biases included.

Data: TidyTuesday 2022-01-25 (BoardGameGeek).
