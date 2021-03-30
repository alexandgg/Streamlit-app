import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 

#Set Full Page Width
st.set_page_config(layout="wide")

#Title 
st.title("ETF Funnel")
#Load DataFrames
fundamental_df = pd.read_csv("Fundamental_Clustering.csv",index_col="Ticker") #Fundamental Clustering Data
weekly_return = pd.read_csv("WeeklyReturns.csv",index_col="Date") #Weekly Return Data
fundamental_df = fundamental_df.loc[fundamental_df.index.intersection(weekly_return.columns)]


#Streamlit Code

funds = st.multiselect("Select Desired Fundamentals",fundamental_df["Fundamental Cluster"].unique()) # Select Box of Fundamental Clusters
category = list(fundamental_df[fundamental_df["Fundamental Cluster"].isin(funds)]["Category"].unique())
category_selected = st.multiselect("Select Desired Fundamentals",category, default=category) # Select Box of Fundamental Clusters
selected_funda_tickers = fundamental_df[fundamental_df["Fundamental Cluster"].isin(funds)].index # Tickers Selected

funda_df_selected = fundamental_df[(fundamental_df["Fundamental Cluster"].isin(funds)) & (fundamental_df["Category"].isin(category_selected))]
st.dataframe(funda_df_selected)
tickers_selected = fundamental_df[(fundamental_df["Fundamental Cluster"].isin(funds)) & (fundamental_df["Category"].isin(category_selected))].index


st.sidebar.header("Test")

funds_extend = [[x] for x in funds]




if len(funda_df_selected) > 0:
    dfff = pd.DataFrame(index=weekly_return.index)
    dff_dict = {}
    n_assets = len(tickers_selected)
    portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
    port_weekly_return = weekly_return[list(tickers_selected)].mul(portfolio_weights_ew,axis=1).sum(axis=1)
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
        xaxis_title="Cumulative Performance",
        yaxis_title="Time",
        legend_title="Clusters",
        legend = dict(orientation = "v", y=-0.1, x=0 ,xanchor = 'left',
        yanchor ='top'))
    st.plotly_chart(fig,use_container_width=True)



