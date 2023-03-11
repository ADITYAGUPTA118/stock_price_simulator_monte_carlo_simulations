import streamlit as st
from datetime import date

import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import random 
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader as data
from scipy.stats import norm
from scipy.stats import skewnorm
from plotly import graph_objs as go

st.title("Stock Price Prediction using Monte Carlo Simulations")
st.subheader("Description of Simulator")
st.write("The Simulator works on the principle of monte carlo simulation. We analyse historical data and based on the analysis we generate a probablity distribution. Using the generated Probablity distribution some random walks(simulations) are generated")
desired_ticker=st.text_input('Enter Stock Ticker','INFY.NS')


default_start=date(2020,1,1)
default_end=date(2020,12,31)
start=st.date_input("Enter starting date",default_start)
end=st.date_input("Enter ending date",default_end)

df=yf.download(desired_ticker,start,end)

st.subheader("Analysis of Data used to used to calculate distributions")
st.write(df.describe())

price=df['Close']
historical_returns=price.pct_change()
historical_returns=historical_returns.dropna()

lower_bound=norm.ppf(0.25)
upper_bound=norm.ppf(0.70)

mean=historical_returns.mean()
stdev=historical_returns.std()

np.random.seed()
n=np.random.normal(size=(10,30))
rows=n.shape[0]
cols=n.shape[1]

for i in range(0,rows):
    for j in range(0,cols):
        if n[i][j]>upper_bound:
            n[i][j]=upper_bound
        elif n[i][j]<lower_bound:
            n[i][j]=lower_bound
        else:
            n[i][j]=n[i][j]
        n[i][j]=(stdev*(n[i][j]))+mean

p=np.zeros([rows,cols+1])

for i in range(0,rows):
    p[i][0]=price[-1]

for i in range (1,cols+1):
    for j in range(0,rows):
        p[j][i]=(p[j][i-1])*(1+n[j][i-1])
    
x=np.arange(0,cols+1)
fig, ax = plt.subplots(nrows=1,ncols=1)
for d in range(0,rows):
    ax.plot(x, p[d])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
st.write("Here we generate 10 simulations for the next 30 days")
st.pyplot(fig)
