import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from tqdm import tqdm
from itertools import combinations
from sklearn.decomposition import PCA
import random



########## Functions ############
def meanRetAn(data):             
    Result = 1
    
    for i in data:
        Result *= (1+i)
        
    Result = Result**(1/float(len(data)/52))-1
     
    return(Result)

def calculate_pdi(num_assets, tickers, weekly_returns, status, prog_bar): 
    
    def meanRetAn(data):             
        Result = 1
        
        for i in data:
            Result *= (1+i)
            
        Result = Result**(1/float(len(data)/52))-1
        
        return(Result)

    pca = PCA()
    PDI_dict = {}
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
        prog_bar.progress(prog)
        status.text("{}% Complete".format(prog))
        n_assets = len(i)
        portfolio_weights_ew = np.repeat(1/n_assets, n_assets)
        port_weekly_return = weekly_returns[i].mul(portfolio_weights_ew,axis=1).sum(axis=1)
        ann_ret = meanRetAn(list(port_weekly_return))
        an_cov = weekly_returns[i].cov()
        port_std = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(an_cov, portfolio_weights_ew)))*np.sqrt(52)
        corr_matrix = np.array(weekly_returns[i].corr())
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