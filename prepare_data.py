"""
Step 1: download the BoardGameGeek data and turn it into a clean modelling table.

Data: TidyTuesday 2022-01-25 (scraped from BoardGameGeek), ~21k games.
Output: data/games.csv  (one row per game, numeric features + target `rating`)

Run:  python prepare_data.py
"""
import ast
import json
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25"
DATA = Path("data")
DATA.mkdir(exist_ok=True)

MIN_RATINGS = 200      # only keep games with enough ratings for a reliable average
N_MECHANICS = 12       # most common mechanics become yes/no features
N_CATEGORIES = 10      # most common categories (themes) become yes/no features


def load(name: str) -> pd.DataFrame:
    local = DATA / f"{name}.csv"
    if not local.exists():
        print(f"Downloading {name}.csv ...")
        pd.read_csv(f"{BASE}/{name}.csv").to_csv(local, index=False)
    return pd.read_csv(local)


def parse_list(s) -> list[str]:
    if not isinstance(s, str):
        return []
    try:
        return list(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        return []


def slug(name: str, prefix: str) -> str:
    clean = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return f"{prefix}{clean}"


def main() -> None:
    ratings = load("ratings")
    details = load("details")

    df = details.merge(ratings[["id", "average", "users_rated"]], on="id", how="inner")
    df = df[df["users_rated"] >= MIN_RATINGS]
    df = df[(df["yearpublished"] >= 1960) & (df["yearpublished"] <= 2021)]
    df = df[(df["playingtime"] > 0) & (df["maxplayers"] > 0) & (df["minplayers"] > 0)]

    out = pd.DataFrame(index=df.index)
    out["year"] = df["yearpublished"]
    out["min_players"] = df["minplayers"].clip(1, 8)
    out["max_players"] = df["maxplayers"].clip(1, 12)
    out["play_time"] = df["playingtime"].clip(5, 480)
    out["min_play_time"] = df["minplaytime"].clip(5, 480)
    # (maxplaytime is an exact copy of playingtime in this dataset, so it is left out)
    out["min_age"] = df["minage"].clip(0, 18)

    mechanics = df["boardgamemechanic"].map(parse_list)
    categories = df["boardgamecategory"].map(parse_list)
    out["n_mechanics"] = mechanics.map(len).clip(0, 15)

    labels = {}
    top_mech = mechanics.explode().value_counts().head(N_MECHANICS).index
    for m in top_mech:
        col = slug(m, "mech_")
        out[col] = mechanics.map(lambda xs, m=m: int(m in xs))
        labels[col] = m
    top_cat = categories.explode().value_counts().head(N_CATEGORIES).index
    for c in top_cat:
        col = slug(c, "cat_")
        out[col] = categories.map(lambda xs, c=c: int(c in xs))
        labels[col] = c

    out["rating"] = df["average"]
    out.insert(0, "name", df["primary"])
    out = out.dropna().reset_index(drop=True)

    out.to_csv(DATA / "games.csv", index=False)
    (DATA / "binary_labels.json").write_text(json.dumps(labels, indent=2))
    print(f"Saved {len(out):,} games x {out.shape[1] - 2} features to data/games.csv")


if __name__ == "__main__":
    main()
