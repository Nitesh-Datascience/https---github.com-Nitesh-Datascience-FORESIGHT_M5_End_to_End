from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_compact_data_exists():
    assert (ROOT/'data/raw/sales_compact.csv').exists()
def test_forecast_output_exists():
    assert (ROOT/'outputs/forecast_28_days.csv').exists()
def test_forecast_columns():
    df=pd.read_csv(ROOT/'outputs/forecast_28_days.csv')
    assert {'id','date','forecast_units','lower_80','upper_80'}.issubset(df.columns)
