import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from tqdm import tqdm
from itertools import combinations
from sklearn.decomposition import PCA
import random
from collections import OrderedDict


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

@st.cache(suppress_st_warning=True)
def make_selected_df(x_risk,x_funds, x_cat):
    selected_tickers = fundamental_df[(fundamental_df["Risk Cluster"].isin(x_risk)) & (fundamental_df["Fundamental Cluster"].isin(x_funds)) & (fundamental_df["Category"].isin(x_cat))  ].index # Tickers Selected

    selected_df = fundamental_df[(fundamental_df["Risk Cluster"].isin(x_risk)) & (fundamental_df["Fundamental Cluster"].isin(x_funds)) & (fundamental_df["Category"].isin(x_cat))  ]
    return selected_tickers, selected_df

selected_tickers, selected_df = make_selected_df(risk,funds_selected,category_selected)



if len(risk) > 0 and len(funds) > 0 and len(category_selected)>0:
    col4, col5 = st.beta_columns([1.5,1])
    col4.header("ETF's in Universe")
    col4.write("Number of ETf's in universe: {}".format(len(selected_df)))
    col4.dataframe(selected_df)
    col5.header("Category Distribution in Universe")
    

    fig = px.pie(selected_df, names = "Category")
    col5.plotly_chart(fig)


######################################################################## Portfolio Divercification ######################################################################## 
st.header("Diversification Calculation")
col1,col2 = st.beta_columns(2)
PDI_DF = pd.DataFrame()
PDI_dict = {}
number_of_assets = [0]
number_of_assets.extend([x for x in range(2,12)])
number_of_assets_selected = col1.selectbox("Please pick number of assets desired to invest in", number_of_assets)
col2.write("The diversification index desscribes how broad a investment is, withnin the selected universe. The large the index number, the more diversified the portfolio is.")


@st.cache(show_spinner=False)
def calculate_pdi(num_assets, tickers):
    pca = PCA()
    samples = [["SPY"]]
    for number in range(2,num_assets, 1):
        for i in range(1,2000):
            #samples.extend([list(x) for x in combinations(selected_tickers, number_of_assets)])
            samples.append(random.sample(list(tickers),number))
    samples_mini = []
    for i in samples:
        if i not in samples_mini:
            samples_mini.append(i)


    
    for i,y in zip(samples_mini,range(1,len(samples_mini)+1)):
        prog = int(y/len(samples_mini)*100)
        progress_bar.progress(prog)
        status_text.text("{}% Complete".format(prog))
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
        PDI_dict[y]["# of Assets"] = len(i)
        PDI_dict[y]["Assets"] = i
        PDI_dict[y]["Sharpe Ratio"] = ann_ret/port_std
        PDI_dict[y]["Annual Return"] = ann_ret
        PDI_dict[y]["Annual STD"] = port_std
    

        


    PDI_DF = pd.DataFrame(PDI_dict).T
    PDI_DF["Assets"] = PDI_DF["Assets"].astype(str)
    PDI_DF["# of Assets"] = PDI_DF["# of Assets"].astype(int)
    PDI_DF["Sharpe Ratio"] = PDI_DF["Sharpe Ratio"].astype(float)
    PDI_DF["Annual STD"] = PDI_DF["Annual STD"].astype(float)
    PDI_DF["PDI_INDEX"] = PDI_DF["PDI_INDEX"].astype(float)
    PDI_DF["Annual Return"] = PDI_DF["Annual Return"].astype(float)
    SPY_DF = PDI_DF.iloc[0,:]
    return PDI_DF,SPY_DF

if number_of_assets_selected == 0:
    st.error("Please choose a number of assets")
else:
    progress_bar = st.progress(0)
    status_text = st.empty()
    PDI_DF,SPY_DF = calculate_pdi(number_of_assets_selected, selected_tickers)
    progress_bar.empty()

changing_pdi_df = PDI_DF.copy()
min = float(x["PDI_INDEX"].min())
max = float(x["PDI_INDEX"].max())

if min < max and min!= max:
    div_choice = st.slider("Diversification Slider", min_value=min, max_value=max)
    changing_pdi_df = PDI_DF[PDI_DF["PDI_INDEX"].astype(float) >= div_choice]

col1, col2 = st.beta_columns([2,1])
col1.subheader("Different Portfolio combinations")
col1.dataframe(changing_pdi_df)
col2.subheader("Performance of World Index")
col2.write("Name of ETF: {}".format(SPY_DF["Assets"]))
col2.write("Sharpe Ratio: {}".format(SPY_DF["Sharpe Ratio"].round(3)))
col2.write("Annual Mean Return: {}".format(SPY_DF["Annual Return"].round(3)))
col2.write("Annual Standard Deviation: {}".format(SPY_DF["Annual STD"].round(3)))

fig = px.scatter(changing_pdi_df, x ="PDI_INDEX" , y = "Sharpe Ratio", hover_data=["Assets", changing_pdi_df.index], color = "Annual STD")
fig.update_layout(
            title="Portfolio Diversificaton",
            xaxis_title="Diversification",
            yaxis_title="Sharpe Ratio",
            legend_title="Volatility",
            legend = dict(orientation = "v", y=-0.1, x=0 ,xanchor = 'left',
            yanchor ='top'))
fig.add_hline(y=SPY_DF["Sharpe Ratio"], line_color= "orange", annotation_text=SPY_DF["Assets"], line_dash="dot",annotation_position="bottom right")
st.plotly_chart(fig,use_container_width=True)


plotting_pdi(PDI_DF,SPY_DF)
################################################################################################################################################################



# funds_extend = [[x] for x in funds_selected]


# if len(selected_df) > 0:
#     dfff = pd.DataFrame(index=weekly_return.index)
#     dff_dict = {}
#     n_assets = len(selected_tickers)
#     portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
#     port_weekly_return = weekly_return[list(selected_tickers)].mul(portfolio_weights_ew,axis=1).sum(axis=1)
#     dfff["Selected Portfolio"] = port_weekly_return
#     for i in funds_extend:
#         tickers = list(fundamental_df[fundamental_df["Fundamental Cluster"].isin(i)].index)
#         n_assets = len(tickers)
#         portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
#         port_weekly_return = weekly_return[tickers].mul(portfolio_weights_ew,axis=1).sum(axis=1)
#         dfff[str(i)] = port_weekly_return
#     cumsum = dfff.cumsum(axis=0)
#     cumsum["SPY"]= weekly_return["SPY"].cumsum(axis=0)

#     fig = px.line(cumsum, x = cumsum.index, y = cumsum.columns)
#     fig.update_layout(
#         title="Cluster Perforance",
#         xaxis_title="Time",
#         yaxis_title="Cumulative Performance",
#         legend_title="Clusters",
#         legend = dict(orientation = "v", y=-0.1, x=0 ,xanchor = 'left',
#         yanchor ='top'))
#     st.plotly_chart(fig,use_container_width=True)



