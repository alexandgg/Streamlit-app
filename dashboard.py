import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from tqdm import tqdm
from itertools import combinations
from sklearn.decomposition import PCA
import random


#Set Full Page Width
st.set_page_config(layout="wide")

#Title 
st.title("ETF Funnel")
#Load DataFrames
fundamental_df = pd.read_csv("fund_risk_cluster_reduced.csv",index_col="Ticker") #Fundamental Clustering Data
weekly_return = pd.read_csv("WeeklyReturns.csv",index_col="Date") #Weekly Return Data
fundamental_df = fundamental_df.loc[fundamental_df.index.intersection(weekly_return.columns)]


########## Functions ############
def meanRetAn(data):             
    Result = 1
    
    for i in data:
        Result *= (1+i)
        
    Result = Result**(1/float(len(data)/52))-1
     
    return(Result)

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


if len(risk) > 0 and len(funds) > 0 and len(category_selected)>0:
    col4, col5 = st.beta_columns(2)
    col4.dataframe(selected_df)
    col4.write("Numer of ETf's: {}".format(len(selected_df)))

    fig = px.pie(selected_df, names = "Category", title="Asset Universe Allocation")
    col5.plotly_chart(fig)


PDI_DF = pd.DataFrame()
PDI_dict = {}
number_of_assets = [0]
number_of_assets.extend([x for x in range(2,11)])
number_of_assets_selected = st.selectbox("Please pick number of assets desired to invest in", number_of_assets)
if number_of_assets_selected == 0:
    st.error("Please choose a number of assets")
if number_of_assets_selected > 2:
    ############ PDI calculation ############

    samples = [["SPY"]]
    for number in range(2,number_of_assets_selected, 1):
        for i in range(1,2000):
            #samples.extend([list(x) for x in combinations(selected_tickers, number_of_assets)])
            samples.append(random.sample(list(selected_tickers),number))




    pca = PCA()
    PDI_dict = {}
    for i,y in zip(samples,range(len(samples))):
        n_assets = len(i)
        portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
        port_weekly_return = weekly_return[i].mul(portfolio_weights_ew,axis=1).sum(axis=1)
        ann_ret = meanRetAn(list(port_weekly_return))
        an_cov = weekly_return[i].cov()
        port_std = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(an_cov, portfolio_weights_ew)))*np.sqrt(52)
        corr_matrix = np.array(weekly_return[i].corr())
        principalComponents = pca.fit(corr_matrix)
        PDI = 2*sum(principalComponents.explained_variance_ratio_*range(1,len(principalComponents.explained_variance_ratio_)+1,1))-1
        PDI_dict[y] = {}
        PDI_dict[y]["PDI_INDEX"] = PDI
        PDI_dict[y]["Number of Assets"] = len(i)
        PDI_dict[y]["Assets"] = i
        PDI_dict[y]["Sharpe Ratio"] = ann_ret/port_std
        PDI_dict[y]["Annual Return"] = ann_ret
        PDI_dict[y]["Annual STD"] = port_std

    PDI_DF = pd.DataFrame(PDI_dict).T
    PDI_DF["Assets"] = PDI_DF["Assets"].astype(str)
    PDI_DF["Number of Assets"] = PDI_DF["Number of Assets"].astype(int)
    PDI_DF["Sharpe Ratio"] = PDI_DF["Sharpe Ratio"].astype(float)
    PDI_DF["Annual STD"] = PDI_DF["Annual STD"].astype(float)
    PDI_DF["PDI_INDEX"] = PDI_DF["PDI_INDEX"].astype(float)
    PDI_DF["Annual Return"] = PDI_DF["Annual Return"].astype(float)

SPY_DF = PDI_DF.iloc[0,:]
PDI_DF_2 = PDI_DF.copy()
min_df = float(PDI_DF_2["PDI_INDEX"].min())
max_df = float(PDI_DF_2["PDI_INDEX"].max())

# while min_df != max_df and min_df < max_df:
a = st.slider("Diversification Slider", min_value=float(PDI_DF_2["PDI_INDEX"].min()), max_value=float(PDI_DF_2["PDI_INDEX"].max()))
PDI_DF_2 = PDI_DF_2[PDI_DF_2["PDI_INDEX"] >= a]
# st.write(a)

st.dataframe(PDI_DF_2)
if len(PDI_DF_2) > 0:
    fig = px.scatter(PDI_DF_2, x ="PDI_INDEX" , y = "Sharpe Ratio", hover_data=["Assets"], color = "Annual STD")
    fig.update_layout(
                title="Portfolio Diversificaton",
                xaxis_title="Diversification",
                yaxis_title="Sharpe Ratio",
                legend_title="Clusters",
                legend = dict(orientation = "v", y=-0.1, x=0 ,xanchor = 'left',
                yanchor ='top'))
    fig.add_hline(y=SPY_DF["Sharpe Ratio"], line_color= "orange", annotation_text=SPY_DF["Assets"], line_dash="dot",annotation_position="bottom right")
    st.plotly_chart(fig,use_container_width=True)



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



