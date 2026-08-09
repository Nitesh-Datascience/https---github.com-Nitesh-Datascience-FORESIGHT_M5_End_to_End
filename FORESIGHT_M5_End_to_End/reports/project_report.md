# FORESIGHT Project Report

## Abstract
FORESIGHT is an end-to-end demand forecasting and inventory decision-support system built from the supplied M5 sales, calendar and price datasets. A compact real-data subset is used to make the repository practical for GitHub and cloud deployment. The system creates leakage-safe lag and rolling features, performs time-based backtesting, compares a machine-learning model with a seasonal-naive baseline, forecasts the official next 28 days, and converts forecast output into transparent planning signals.

## Dataset
The supplied data contain 30,490 series in the original M5 training files. This implementation selects 120 high-volume series with stratification by category/store segment. It retains 365 historical days and the official 28-day evaluation horizon. Prices and calendar rows are filtered to exactly those series/weeks.

## Methodology
- Daily sales as target.
- Lags: 1, 7, 14, 28 days.
- Rolling demand: 7 and 28 days.
- Calendar: weekday/month/event flag.
- Price and price change.
- Model: HistGradientBoostingRegressor.
- Baseline: 7-day seasonal naive.
- Validation: last 28 training days.
- Final evaluation: official future 28 days from `sales_train_evaluation.csv`.

## Business layer
Because no observed inventory file was supplied, the system uses a 14-day stock-cover proxy derived from recent demand. This keeps the analysis honest and makes the assumption explicit.

## Limitations
The compact subset is intentionally smaller than the complete M5 universe. The inventory action layer should be replaced by real on-hand/on-order/lead-time data when available.
