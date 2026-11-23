// Collect the Google Trends data used by the "Reality check" tab.
//
// How to run:
//   1. Open https://trends.google.com/trends/explore in a browser.
//   2. Open the developer console (Cmd+Option+J on Mac, Ctrl+Shift+J on Windows).
//   3. Paste this whole file and press Enter. It takes about 3 minutes.
//   4. When it finishes it downloads trends_popularity.csv and trends_yearly.csv.
//      Put both in the project's data/ folder.
//
// It uses the same internal endpoints as the Trends website, so Google may rate-limit it
// (HTTP 429). If that happens, wait a few minutes and run it again: finished parts are
// saved in localStorage and skipped.

(async () => {
  const CATAN = "/m/095hd";
  const GAMES = {
    "Gloomhaven": "/g/11ggr7s3mb", "Wingspan": "/g/11j0828t6w", "Pandemic": "/m/04ct4tq",
    "Codenames": "/g/11bwlqy0bn", "Ticket to Ride": "/m/050jzx", "Terraforming Mars": "/g/11fy_qgrsp",
    "Carcassonne": "/m/01kzfr", "7 Wonders": "/m/0gyx5lq", "Splendor": "/m/012nl24s",
    "Spirit Island": "/g/11hpmxkkhm", "Everdell": "/g/11g_vqt7fg", "Dixit": "/m/0cn_gq3",
    "King of Tokyo": "/m/0yt19r8", "Sushi Go!": "/g/11b7jn_khs", "Agricola": "/m/03d449b",
    "Kingdomino": "/g/11dd_tl_nk", "Exploding Kittens": "/m/012r3ljs",
    "Cards Against Humanity": "/m/0gx0llp", "Jenga": "/m/01cfcp", "Pictionary": "/m/0k1fz",
    "Twister": "/m/0p0tx", "Connect Four": "/m/01qzq9", "Guess Who?": "/m/03pq4f",
    "Taboo": "/m/02dn1f", "Boggle": "/m/0lz6p",
  };
  const HYPE = ["Gloomhaven", "Wingspan", "Pandemic", "Codenames", "Terraforming Mars", "7 Wonders",
                "Splendor", "Spirit Island", "Everdell", "King of Tokyo", "Sushi Go!", "Kingdomino",
                "Exploding Kittens", "Cards Against Humanity", "Dixit", "Agricola"];
  const TERMS = ["board game", "boardgamegeek", "deck building game", "cooperative board game",
                 "worker placement", "party game", "wargame"];
  const TIME = "2004-01-01 2026-10-31";
  const POP_FROM_YEAR = 2019;
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function series(keywords) {
    const req = {comparisonItem: keywords.map(k => ({keyword: k, geo: "", time: TIME})), category: 0, property: ""};
    const r = await fetch("/trends/api/explore?hl=en-US&tz=0&req=" + encodeURIComponent(JSON.stringify(req)));
    if (r.status !== 200) throw new Error("explore HTTP " + r.status);
    const t = await r.text();
    const w = JSON.parse(t.slice(t.indexOf("{"))).widgets.find(x => x.id === "TIMESERIES");
    await sleep(1500);
    const r2 = await fetch("/trends/api/widgetdata/multiline?hl=en-US&tz=0&req=" +
      encodeURIComponent(JSON.stringify(w.request)) + "&token=" + w.token);
    if (r2.status !== 200) throw new Error("data HTTP " + r2.status);
    const t2 = await r2.text();
    const d = JSON.parse(t2.slice(t2.indexOf("{"))).default;
    return {dates: d.timelineData.map(p => p.formattedAxisTime),
            values: keywords.map((_, i) => d.timelineData.map(p => p.value[i]))};
  }

  const KEY = "bgtrends_v2";
  const store = JSON.parse(localStorage.getItem(KEY) || '{"batches":[],"singles":{}}');
  const save = () => localStorage.setItem(KEY, JSON.stringify(store));

  // 1) popularity: 4 games + Catan per request, so every game shares Catan's scale
  const names = Object.keys(GAMES);
  const done = new Set(store.batches.flatMap(b => b.names));
  for (let i = 0; i < names.length; i += 4) {
    const chunk = names.slice(i, i + 4);
    if (chunk.every(n => done.has(n))) continue;
    console.log("popularity batch", chunk.join(", "));
    const res = await series([CATAN, ...chunk.map(n => GAMES[n])]);
    store.batches.push({names: chunk, dates: res.dates, catan: res.values[0], values: res.values.slice(1)});
    save(); await sleep(2500);
  }
  // 2) one request per game / term: each series on its own 0-100 scale
  for (const [name, kw] of [...HYPE.map(n => [n, GAMES[n]]), ...TERMS.map(t => ["term:" + t, t])]) {
    if (store.singles[name]) continue;
    console.log("series", name);
    const res = await series([kw]);
    store.singles[name] = res.values[0]; store.dates = res.dates;
    save(); await sleep(2500);
  }

  // 3) summarise and download
  const dates = store.batches[0].dates, yr = dates.map(d => +d.slice(-4));
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const idx = yr.map((y, i) => (y >= POP_FROM_YEAR ? i : -1)).filter(i => i >= 0);
  let pop = "game,topic_id,popularity_vs_catan\n";
  for (const b of store.batches) {
    const c = mean(idx.map(i => b.catan[i]));
    b.names.forEach((n, k) => { pop += `${n},${GAMES[n]},${(mean(idx.map(i => b.values[k][i])) / c * 100).toFixed(2)}\n`; });
  }
  const years = [...new Set(yr)];
  let yearly = "series," + years.join(",") + "\n";
  for (const [n, v] of Object.entries(store.singles)) {
    yearly += n + "," + years.map(y => mean(v.filter((_, i) => yr[i] === y)).toFixed(1)).join(",") + "\n";
  }
  for (const [fname, text] of [["trends_popularity.csv", pop], ["trends_yearly.csv", yearly]]) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], {type: "text/csv"}));
    a.download = fname; a.click();
  }
  console.log("Done. Move both CSVs into the project's data/ folder.");
})();
