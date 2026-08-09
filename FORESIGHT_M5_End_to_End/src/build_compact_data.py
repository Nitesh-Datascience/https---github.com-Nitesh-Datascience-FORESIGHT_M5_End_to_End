from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UP=Path('/mnt/data')
VAL=UP/'sales_train_validation.csv'
EVAL=UP/'sales_train_evaluation.csv'
CAL=UP/'calendar(1).csv'
PRICE=UP/'sell_prices.csv'
OUT=ROOT/'data/raw'
OUT.mkdir(parents=True,exist_ok=True)

# Real M5 data supplied by the user. Keep a compact, representative subset.
val=pd.read_csv(VAL)
eval_=pd.read_csv(EVAL)
cal=pd.read_csv(CAL, parse_dates=['date'])
prices=pd.read_csv(PRICE)

# Select 120 highest-volume validation series across the catalog.
dcols=[c for c in val.columns if c.startswith('d_')]
val['total_units']=val[dcols].sum(axis=1)
# stratify by category/store so the compact sample is not dominated by one segment
val['segment']=val['cat_id'].astype(str)+'_'+val['store_id'].astype(str)
parts=[]
for seg,g in val.groupby('segment'):
    n=max(1,round(120*len(g)/len(val)))
    parts.append(g.nlargest(min(n,len(g)),'total_units'))
selected=pd.concat(parts).drop_duplicates('id').nlargest(120,'total_units').copy()
ids=selected['id'].tolist()
selected_val=val[val.id.isin(ids)].copy()
selected_eval=eval_[eval_.id.str.replace('_evaluation$','_validation',regex=True).isin(ids)].copy()
selected_eval['id']=selected_eval['id'].str.replace('_evaluation$','_validation',regex=True)

# 365-day training window immediately before validation cutoff + 28-day evaluation horizon.
train_days=dcols[-365:]
future_days=[f'd_{i}' for i in range(1914,1942)]
# Long-form compact sales table; ~45k rows rather than 100+ MB wide source.
meta_cols=['id','item_id','dept_id','cat_id','store_id','state_id']
train_long=selected_val[meta_cols+train_days].melt(id_vars=meta_cols,var_name='d',value_name='units')
future_long=selected_eval[meta_cols+future_days].melt(id_vars=meta_cols,var_name='d',value_name='units')
train_long['split']='train'
future_long['split']='test'
sales=pd.concat([train_long,future_long],ignore_index=True)
sales=sales.merge(cal[['d','date','wm_yr_wk','weekday','wday','month','year','event_name_1','event_type_1','snap_CA','snap_TX','snap_WI']],on='d',how='left')
sales.to_csv(OUT/'sales_compact.csv',index=False)

# Filter prices only to selected item/store pairs and relevant weeks.
pairs=selected[['item_id','store_id']].drop_duplicates()
weeks=sales['wm_yr_wk'].dropna().unique()
prices=prices.merge(pairs,on=['item_id','store_id'],how='inner')
prices=prices[prices.wm_yr_wk.isin(weeks)][['store_id','item_id','wm_yr_wk','sell_price']]
prices.to_csv(OUT/'prices_compact.csv',index=False)

# Calendar subset only.
cal[cal.d.isin(sales.d.unique())][['date','d','wm_yr_wk','weekday','wday','month','year','event_name_1','event_type_1','snap_CA','snap_TX','snap_WI']].to_csv(OUT/'calendar_compact.csv',index=False)

# Metadata and provenance.
selected[meta_cols].to_csv(OUT/'sku_master_compact.csv',index=False)
print('sales_compact MB', (OUT/'sales_compact.csv').stat().st_size/1024/1024)
print('prices_compact MB', (OUT/'prices_compact.csv').stat().st_size/1024/1024)
print('selected series',len(ids),'sales rows',len(sales))
