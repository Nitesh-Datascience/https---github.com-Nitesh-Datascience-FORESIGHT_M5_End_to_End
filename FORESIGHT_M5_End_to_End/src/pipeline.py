from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw'; OUT=ROOT/'outputs'; MODELS=ROOT/'models'
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
FEATURES=['lag_1','lag_7','lag_14','lag_28','roll_7','roll_28','weekday_num','month','price','price_change','event_flag']

def load():
    s=pd.read_csv(DATA/'sales_compact.csv',parse_dates=['date'])
    p=pd.read_csv(DATA/'prices_compact.csv')
    s=s.sort_values(['id','date']).reset_index(drop=True)
    s['units']=pd.to_numeric(s.units,errors='coerce').fillna(0).clip(lower=0)
    s=s.merge(p,on=['store_id','item_id','wm_yr_wk'],how='left')
    s['sell_price']=s['sell_price'].ffill().bfill().fillna(0)
    s['price']=s['sell_price']
    s['event_flag']=s['event_name_1'].notna().astype(int)
    s['weekday_num']=s['date'].dt.dayofweek
    s['month']=s['date'].dt.month
    s['price_change']=s.groupby('id')['sell_price'].pct_change().replace([np.inf,-np.inf],0).fillna(0).clip(-1,1)
    return s

def make_features(s):
    x=s.copy()
    g=x.groupby('id',sort=False)['units']
    for k in [1,7,14,28]: x[f'lag_{k}']=g.shift(k)
    x['roll_7']=g.shift(1).rolling(7).mean().reset_index(level=0,drop=True)
    x['roll_28']=g.shift(1).rolling(28).mean().reset_index(level=0,drop=True)
    return x

def wape(y,p):
    d=np.abs(y).sum(); return float(np.abs(y-p).sum()/d) if d else 0.0

def train_model(train):
    feat=make_features(train)
    # time split: last 28 training days as backtest; no random split
    dates=sorted(feat.date.unique())
    cut=dates[-28]
    tr=feat[feat.date<cut].dropna(subset=FEATURES)
    va=feat[feat.date>=cut].dropna(subset=FEATURES)
    model=HistGradientBoostingRegressor(max_iter=250,learning_rate=.08,max_leaf_nodes=31,l2_regularization=.5,random_state=42)
    model.fit(tr[FEATURES],tr.units)
    pred=np.clip(model.predict(va[FEATURES]),0,None)
    baseline=np.clip(va['lag_7'].fillna(va['lag_1']),0,None)
    metrics={'model_wape':wape(va.units.values,pred),'baseline_wape':wape(va.units.values,baseline.values),'model_mae':float(mean_absolute_error(va.units,pred)),'baseline_mae':float(mean_absolute_error(va.units,baseline)),'backtest_rows':int(len(va))}
    selected=metrics['model_wape']<metrics['baseline_wape']
    # refit selected model on all usable training history
    alltr=feat[feat.date<dates[-1]+np.timedelta64(1,'D')].dropna(subset=FEATURES)
    final=HistGradientBoostingRegressor(max_iter=250,learning_rate=.08,max_leaf_nodes=31,l2_regularization=.5,random_state=42)
    final.fit(alltr[FEATURES],alltr.units)
    joblib.dump({'model':final,'features':FEATURES,'metrics':metrics,'selected_model': 'HistGradientBoosting' if selected else 'SeasonalNaive'},MODELS/'forecast_model.joblib')
    return feat,metrics,selected,final

def forecast_28(train,model,selected):
    # recursive daily forecast using 7/14/28 lags and rolling windows
    hist=train[train.split.eq('train')].copy().sort_values(['id','date'])
    future=train[train.split.eq('test')][['id','item_id','dept_id','cat_id','store_id','state_id','date','wm_yr_wk','weekday','wday','month','year','event_name_1','event_type_1','snap_CA','snap_TX','snap_WI','sell_price']].drop_duplicates(['id','date']).sort_values(['id','date']).copy()
    work=hist[['id','item_id','dept_id','cat_id','store_id','state_id','date','wm_yr_wk','weekday','wday','month','year','event_name_1','event_type_1','snap_CA','snap_TX','snap_WI','sell_price','units']].copy()
    out=[]
    for d in sorted(future.date.unique()):
        day=future[future.date.eq(d)].copy()
        rows=[]
        for _,r in day.iterrows():
            h=work[work.id.eq(r.id)].sort_values('date')
            vals=h.units.to_numpy(dtype=float)
            price=float(r.sell_price) if pd.notna(r.sell_price) else float(h.sell_price.iloc[-1])
            prev_price=float(h.sell_price.iloc[-1]) if len(h) else price
            row=r.to_dict(); row.update({'lag_1':vals[-1],'lag_7':vals[-7] if len(vals)>=7 else vals[-1],'lag_14':vals[-14] if len(vals)>=14 else vals[-1],'lag_28':vals[-28] if len(vals)>=28 else vals[-1],'roll_7':vals[-7:].mean(),'roll_28':vals[-28:].mean(),'weekday_num':pd.Timestamp(d).dayofweek,'month':pd.Timestamp(d).month,'price':price,'price_change':np.clip((price-prev_price)/prev_price if prev_price else 0,-1,1),'event_flag':int(pd.notna(r.event_name_1))})
            rows.append(row)
        rf=pd.DataFrame(rows)
        if selected: pred=np.clip(model.predict(rf[FEATURES]),0,None)
        else: pred=rf.lag_7.to_numpy()
        rf['forecast_units']=pred
        # approximate 80% interval from backtest WAPE
        out.append(rf.copy())
        add=rf.copy()
        add['units']=rf['forecast_units']
        work=pd.concat([work,add[work.columns]],ignore_index=True)
    fc=pd.concat(out,ignore_index=True)
    return fc

def main():
    s=load()
    train=s[s.split.eq('train')].copy()
    feat,metrics,selected,model=train_model(train)
    fc=forecast_28(s,model,selected)
    err=max(metrics['model_wape'] if selected else metrics['baseline_wape'],.08)
    fc['lower_80']=np.clip(fc.forecast_units*(1-1.28*err),0,None)
    fc['upper_80']=fc.forecast_units*(1+1.28*err)
    # actual 28-day future is present in the compact test rows; evaluate without using it for training
    actual=s[s.split.eq('test')][['id','date','units']].rename(columns={'units':'actual_units'})
    fc=fc.merge(actual,on=['id','date'],how='left')
    test_wape=wape(fc.actual_units.values,fc.forecast_units.values)
    fc.to_csv(OUT/'forecast_28_days.csv',index=False)
    # inventory proxy: expected stock position from recent 28-day demand; transparent and reproducible.
    last=train.sort_values('date').groupby('id').tail(1)[['id','item_id','store_id','cat_id','sell_price','units']].rename(columns={'units':'last_day_units'})
    recent=train[train.date>=train.date.max()-pd.Timedelta(days=28)].groupby('id').units.sum().rename('recent_28d_units')
    fc_sum=fc.groupby('id').forecast_units.sum().rename('forecast_28d_units')
    risk=last.merge(recent,on='id').merge(fc_sum,on='id')
    risk['avg_daily_demand']=risk.recent_28d_units/28
    # proxy available stock uses recent sales velocity times a 14-day cover assumption; this is a planning proxy, not observed inventory.
    risk['proxy_stock_units']=np.maximum(risk.last_day_units*14,0)
    risk['shortage_units']=(risk.forecast_28d_units-risk.proxy_stock_units).clip(lower=0)
    risk['excess_units']=(risk.proxy_stock_units-risk.forecast_28d_units*1.5).clip(lower=0)
    risk['sales_at_risk_inr']=risk.shortage_units*risk.sell_price
    risk['locked_capital_inr']=risk.excess_units*risk.sell_price*.65
    risk['stockout_score']=(risk.shortage_units/(risk.forecast_28d_units+1)).clip(0,1)
    risk['overstock_score']=(risk.excess_units/(risk.proxy_stock_units+1)).clip(0,1)
    risk['action']=np.select([risk.stockout_score>=.25,risk.overstock_score>=.30],['REORDER NOW','MARKDOWN / CLEAR'],default='HEALTHY')
    risk['risk_level']=np.select([np.maximum(risk.stockout_score,risk.overstock_score)>=.6,np.maximum(risk.stockout_score,risk.overstock_score)>=.3],['High','Medium'],default='Low')
    risk=risk.sort_values('sales_at_risk_inr',ascending=False)
    risk.to_csv(OUT/'risk_decisions.csv',index=False)
    summary={'selected_method':'HistGradientBoosting' if selected else 'SeasonalNaive','backtest':metrics,'official_future_test_wape':test_wape,'series_count':int(s.id.nunique()),'sales_at_risk_inr':float(risk.sales_at_risk_inr.sum()),'locked_capital_inr':float(risk.locked_capital_inr.sum()),'reorder_now':int((risk.action=='REORDER NOW').sum()),'markdown_clear':int((risk.action=='MARKDOWN / CLEAR').sum())}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
