from pathlib import Path
import sys,pandas as pd
from fastapi import FastAPI,HTTPException
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.pipeline import main as run
app=FastAPI(title='FORESIGHT Forecast API',version='1.0')
P=ROOT/'outputs/forecast_28_days.csv'; R=ROOT/'outputs/risk_decisions.csv'
def load():
    if not P.exists() or not R.exists(): run()
    return pd.read_csv(P),pd.read_csv(R)
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/score/{sku_id}')
def score(sku_id:str):
    f,r=load(); rr=r[r.id.eq(sku_id)]; ff=f[f.id.eq(sku_id)]
    if rr.empty: raise HTTPException(404,'SKU not found')
    return {'risk':rr.iloc[0].to_dict(),'forecast':ff[['date','forecast_units','lower_80','upper_80','actual_units']].to_dict('records')}
