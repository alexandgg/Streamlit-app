import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 


#Set Full Page Width
st.set_page_config(layout="wide")

#Title 
st.title("ETF Funnel")
#Load DataFrames
fundamental_df = pd.read_csv("fund_risk_cluster_reduced.csv",index_col="Ticker") #Fundamental Clustering Data
weekly_return = pd.read_csv("WeeklyReturns.csv",index_col="Date") #Weekly Return Data
fundamental_df = fundamental_df.loc[fundamental_df.index.intersection(weekly_return.columns)]


#Streamlit Code
col1, col2, col3 = st.beta_columns(3)
risk = col1.multiselect("Choose Risk Category", list(fundamental_df["Risk Cluster"].unique()))
if not risk:
    col1.error("Please choose an option")
funds = list(fundamental_df[(fundamental_df["Risk Cluster"].isin(risk))]["Fundamental Cluster"].unique())
funds_selected = col2.multiselect("Select Desired Fundamentals",funds) # Select Box of Fundamental Clusters
if not funds_selected:
    col2.error("Please choose an option")
category = list(fundamental_df[(fundamental_df["Risk Cluster"].isin(risk)) & (fundamental_df["Fundamental Cluster"].isin(funds_selected))]["Category"].unique())
category_selected = col3.multiselect("Select Desired Investment Categories",category) # Select Box of Fundamental Clusters
if not category_selected:
    col3.error("Please choose an option")

selected_tickers = fundamental_df[(fundamental_df["Risk Cluster"].isin(risk)) & (fundamental_df["Fundamental Cluster"].isin(funds_selected)) & (fundamental_df["Category"].isin(category_selected))  ].index # Tickers Selected

selected_df = fundamental_df[(fundamental_df["Risk Cluster"].isin(risk)) & (fundamental_df["Fundamental Cluster"].isin(funds_selected)) & (fundamental_df["Category"].isin(category_selected))  ]

if len(risk) > 0 and len(funds) > 0 and len(category)>0:
    col4, col5 = st.beta_columns(2)
    col4.dataframe(selected_df)
    st.write("Numer of ETf's: {}".format(len(selected_df)))

    fig = px.pie(selected_df, names = "Category", title="Asset Universe Allocation")
    col5.plotly_chart(fig)








st.sidebar.header("Test")

funds_extend = [[x] for x in funds_selected]




if len(selected_df) > 0:
    dfff = pd.DataFrame(index=weekly_return.index)
    dff_dict = {}
    n_assets = len(selected_tickers)
    portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
    port_weekly_return = weekly_return[list(selected_tickers)].mul(portfolio_weights_ew,axis=1).sum(axis=1)
    dfff["Selected Portfolio"] = port_weekly_return
    for i in funds_extend:
        tickers = list(fundamental_df[fundamental_df["Fundamental Cluster"].isin(i)].index)
        n_assets = len(tickers)
        portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
        port_weekly_return = weekly_return[tickers].mul(portfolio_weights_ew,axis=1).sum(axis=1)
        dfff[str(i)] = port_weekly_return
    cumsum = dfff.cumsum(axis=0)
    cumsum["SPY"]= weekly_return["SPY"].cumsum(axis=0)

    fig = px.line(cumsum, x = cumsum.index, y = cumsum.columns)
    fig.update_layout(
        title="Cluster Perforance",
        xaxis_title="Time",
        yaxis_title="Cumulative Performance",
        legend_title="Clusters",
        legend = dict(orientation = "v", y=-0.1, x=0 ,xanchor = 'left',
        yanchor ='top'))
    st.plotly_chart(fig,use_container_width=True)



