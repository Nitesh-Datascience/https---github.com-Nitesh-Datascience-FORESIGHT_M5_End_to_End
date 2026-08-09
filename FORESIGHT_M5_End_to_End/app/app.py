from pathlib import Path
import json,sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.pipeline import main as pipeline_main
st.set_page_config(page_title='FORESIGHT M5',layout='wide')
st.title('FORESIGHT — Demand Forecast & Inventory Risk')
st.caption('Real M5 data • compact subset • 28-day forecast • business decision dashboard')
summary_p=ROOT/'outputs/summary.json'
if not summary_p.exists():
    with st.spinner('Running pipeline...'): pipeline_main()
summary=json.loads(summary_p.read_text())
fc=pd.read_csv(ROOT/'outputs/forecast_28_days.csv',parse_dates=['date'])
risk=pd.read_csv(ROOT/'outputs/risk_decisions.csv')
with st.sidebar:
    cat=st.selectbox('Category',['All']+sorted(risk.cat_id.unique().tolist()))
    view=risk if cat=='All' else risk[risk.cat_id.eq(cat)]
    selected=st.selectbox('SKU',sorted(view.id.tolist()))
    acts=st.multiselect('Action',sorted(risk.action.unique()),default=sorted(risk.action.unique()))
    view=view[view.action.isin(acts)]
c1,c2,c3,c4=st.columns(4)
c1.metric('Series',summary['series_count']); c2.metric('Backtest WAPE',f"{summary['backtest']['model_wape']:.1%}"); c3.metric('Future WAPE',f"{summary['official_future_test_wape']:.1%}"); c4.metric('Sales at Risk',f"₹{view.sales_at_risk_inr.sum():,.0f}")
t1,t2,t3=st.tabs(['Executive','Forecast','Actions'])
with t1:
    a,b=st.columns(2)
    with a:
        counts=view.action.value_counts().reset_index(); counts.columns=['Action','SKUs']; st.plotly_chart(px.bar(counts,x='Action',y='SKUs',title='Decision mix'),use_container_width=True)
    with b:
        impact=view.groupby('cat_id',as_index=False)[['sales_at_risk_inr','locked_capital_inr']].sum().melt('cat_id'); st.plotly_chart(px.bar(impact,x='cat_id',y='value',color='variable',barmode='group',title='Business impact'),use_container_width=True)
with t2:
    f=fc[fc.id.eq(selected)].sort_values('date'); fig=go.Figure(); fig.add_trace(go.Scatter(x=f.date,y=f.forecast_units,mode='lines+markers',name='Forecast')); fig.add_trace(go.Scatter(x=f.date,y=f.upper_80,mode='lines',name='Upper 80%')); fig.add_trace(go.Scatter(x=f.date,y=f.lower_80,mode='lines',name='Lower 80%')); fig.update_layout(title=f'28-day forecast: {selected}',yaxis_title='Units'); st.plotly_chart(fig,use_container_width=True); st.dataframe(f[['date','forecast_units','lower_80','upper_80','actual_units']],use_container_width=True)
with t3:
    cols=['id','item_id','store_id','cat_id','action','risk_level','forecast_28d_units','shortage_units','sales_at_risk_inr','locked_capital_inr']; st.dataframe(view[cols],use_container_width=True); st.download_button('Download action list',view[cols].to_csv(index=False).encode(),file_name='foresight_actions.csv',mime='text/csv')
st.info('Inventory is not present in the supplied M5 files. The risk layer therefore uses a clearly labelled 14-day stock-cover proxy based on recent sales velocity; it is not presented as observed inventory.')
