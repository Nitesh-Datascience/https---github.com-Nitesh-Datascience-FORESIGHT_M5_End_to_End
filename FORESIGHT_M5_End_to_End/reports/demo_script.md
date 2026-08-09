# 4-Minute Demo Script

**0:00–0:30 — Problem**
“FORESIGHT solves a practical retail planning problem: forecast product demand accurately and turn the forecast into simple actions. I used the real M5 sales, calendar and price files, then compressed the data footprint so the project is easy to deploy.”

**0:30–1:10 — Data**
“The original files are very large, so the project selects 120 high-volume SKU/store series, keeps the latest 365 training days and the official 28-day future period, and filters prices and calendar data to only those records. This keeps the compact dataset around six megabytes.”

**1:10–2:00 — Machine learning**
“I created leakage-safe lag features for 1, 7, 14 and 28 days, rolling demand features, weekday and month features, events, price and price changes. I use a time-based 28-day backtest, never a random split. HistGradientBoosting is compared with a 7-day seasonal-naive baseline.”

**2:00–2:40 — Results**
“The model achieved 28.82% WAPE on the backtest compared with 36.19% for the baseline. On the official held-out 28-day future period, the compact 120-series test WAPE is 33.13%.”

**2:40–3:30 — Dashboard**
“The Streamlit dashboard shows model performance, sales at risk, decision mix, category impact, an individual 28-day forecast with an uncertainty band, and a downloadable action table.”

**3:30–4:00 — Business caveat**
“The supplied M5 files do not contain observed on-hand inventory, on-order units or lead times. I therefore do not fabricate those values. The action layer uses a clearly labelled 14-day stock-cover proxy from recent demand. With real inventory data, this layer can become a true reorder and overstock engine.”
