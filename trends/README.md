# Google Trends data

The "Reality check" tab compares the model with Google search interest. The data in
`data/trends_popularity.csv` and `data/trends_yearly.csv` was collected on 23 November 2026
with `fetch_trends.js`, run inside a browser on trends.google.com. Google Trends has no open
API (the official one is an invite-only alpha, and the pytrends library was archived in 2025),
so the script calls the same endpoints the Trends website uses.

To refresh the data, follow the instructions at the top of `fetch_trends.js`.

## What is in the files

**trends_popularity.csv**: average monthly search interest from January 2019 to October 2026
for 25 games, relative to Catan (= 100). Each request contained Catan plus 4 games, so all
games share Catan's scale.

**trends_yearly.csv**: yearly average search interest, 2004–2026, for 16 games and 7 search
terms. Each series is on its own 0–100 scale (100 = its busiest month).

## Choices and caveats

- Games are Google's board-game *topics* (ids in the CSV), not raw keywords, so "Pandemic"
  means the game, not the disease. Azul, Dominion, Scythe, Root, Brass: Birmingham and
  Operation had no clean topic and were left out.
- Topics still pick up some other meanings: the Pandemic game's 2020 peak follows COVID
  searches, and "7 Wonders" spikes in 2007, before the game existed.
- Newer topics only start recording when Google creates them, so early years can read 0.
- Every series roughly doubles in 2025–2026, including "board game", which looks like a change
  in Google's measurement. The yearly charts stop at 2024. The popularity numbers are ratios to
  Catan within the same request, so they are not affected.
- Search interest measures attention, not enjoyment.
