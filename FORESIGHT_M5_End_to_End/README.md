# FORESIGHT — End-to-End Demand Forecasting & Inventory Risk

## What changed
This version uses the **real M5 datasets uploaded by the user**, but reduces the project data footprint dramatically. Instead of shipping the original ~100–200 MB CSVs, the project creates a compact subset of **120 high-volume SKU/store series**, 365 training days + 28 official future days, and only the matching price/calendar rows.

The compact sales file is about 5.8 MB and the compact price file about 0.2 MB.

## Important source files
The source data are M5-style files supplied in the task:
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `calendar(1).csv`
- `sell_prices.csv`

`sample_submission(2).csv` is not needed for model training and is intentionally excluded from the project package.

## End-to-end workflow
1. Build compact real-data extracts.
2. Clean and merge sales, calendar and prices.
3. Build leakage-safe lag/rolling features.
4. Use a 28-day time-based backtest.
5. Compare HistGradientBoosting against a 7-day seasonal-naive baseline.
6. Refit the winning method on training history.
7. Forecast the official next 28 days.
8. Evaluate against the held-out `sales_train_evaluation.csv` future period.
9. Create a transparent business-risk layer.
10. Visualize decisions in Streamlit.
11. Expose SKU forecasts through FastAPI.

## Why the inventory risk is a proxy
The supplied M5 files contain sales, prices and calendar data, but **do not contain observed on-hand inventory, on-order quantity, lead time, or reorder point**. Therefore this project does not invent those fields. It uses a clearly labelled 14-day stock-cover proxy based on recent sales velocity for the business-risk demonstration.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/build_compact_data.py
python src/run_all.py
python -m streamlit run app/app.py
```

API:
```powershell
uvicorn service.api:app --reload
```

## Deploy
Streamlit Cloud main file: `app/app.py`.

For a lightweight deployment, commit the already-created compact files in `data/raw/` and do not upload the original large source CSVs.

## Outputs
- `outputs/summary.json`
- `outputs/forecast_28_days.csv`
- `outputs/risk_decisions.csv`
- `models/forecast_model.joblib`

## Portfolio title
**FORESIGHT: Real-World Demand Forecasting, 28-Day Prediction & Inventory Risk Intelligence**
