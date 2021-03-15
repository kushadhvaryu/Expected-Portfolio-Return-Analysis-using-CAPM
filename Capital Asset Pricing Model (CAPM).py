#!/usr/bin/env python
# coding: utf-8

# # Understanding the Theory and Intuition behind Capital Asset Pricing Model (CAPM)
# 
# ## CAPM
# 
# - Capital Assets Pricing Model (CAPM) is one of the most important models in Finance.
# - CAPM is a model that describes the relationship between the expected return and risk of securities.
# - CAPM indicates that the expected return on a security is equal to the risk-free return plus a risk premium.
# 
# ## Risk Free Asset
# 
# - CAPM assumes that there exist a risk free asset with zero standard deviation.
# - Investors who are extremely risk averse would prefer to buy the risk free asset to protect their money and earn a low return.
# - If investors are interested in gaining more return, they have to bear more risk compared to the risk free asset.
# - A risk free asset could be a U. S. government 10 year Treasury bill. This is technically a risk free asset since it's backed by the US Government.
# 
# ## Market Portfolio
# 
# - Market portfolio includes all securities in the market. A good representation of the market portfolio is the S&P500 (Standard & Poor's 500 Index).
# - The S&P500 is a market-capitalization-weighted index of the 500 largest U. S. publicly traded companies.
# - The index is viewed as a guage of large-cap U. S. equities.
# 
# ## Beta
# 
# - Beta represents the slope of the regression line (market return vs. stock return)
# - Beta is a measure of the volatility or systematic risk of a security or portfolio compared to the entire market (S&P500)
# - Beta is used in the CAPM and describes the relationship between systematic risk and expected return for assets.
# + Tech stocks generally have higher betas than S&P500 but they also have excess returns
#     - Beta = 1, this indicates that its price activity is strongly correlated with the market.
#     - Beta < 1 (defensive): indicates that the security is theoretically less volatile than the market. (Ex: Utility and consumer goods (P&G). If the stock is included, this will make the portfolio less risky compared to the same portfolio without the stock.
#     - Beta > 1 (aggressive), indicates that the security's price is more volatile than the market. For instance, Tesla stock beta is 1.26 indicating that it's 26% more volatile than the market. It will do better if the economy is booming and worse in cases of recession.

# # Import Libraries/Datasets and Visualized Stocks Data

# In[1]:


import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go


# In[2]:


# Read the stock data file
stocks_df = pd.read_csv("D:\Python and Machine Learning for Financial Analysis\stock.csv")
stocks_df


# In[3]:


# Sorted the data based on Date
stocks_df = stocks_df.sort_values(by = ['Date'])
stocks_df


# In[4]:


# Function to normalize the prices based on the initial price
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x


# In[5]:


# Function to plot interactive plot
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()


# In[6]:


# Plotted interactive chart
interactive_plot(stocks_df, 'Prices')


# In[7]:


# Plotted normalized interactive chart
interactive_plot(normalize(stocks_df), 'Normalized Prices')


# # Calculated Daily Returns

# In[8]:


# Function to calculate the daily returns 
def daily_return(df):

  df_daily_return = df.copy()
  
  # Looped through each stock
  for i in df.columns[1:]:
    
    # Looped through each row belonging to the stock
    for j in range(1, len(df)):
      
      # Calculated the percentage of change from the previous day
      df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
    
    # set the value of first row to zero, as previous value is not available
    df_daily_return[i][0] = 0
  return df_daily_return


# In[9]:


# Got the daily returns 
stocks_daily_return = daily_return(stocks_df)
stocks_daily_return


# # Higher Average Daily Return by Amazon and Google when compared to S&P500

# In[10]:


stocks_daily_return.mean()
# S&P500 average daily return is 0.049%
# Amazon average daily return is 0.15%
# Google average daily return is 0.084%


# # Calculated Beta for a Single Stock

# In[11]:


# Selected any stock, let's say Apple 
stocks_daily_return['AAPL']


# In[12]:


# Selected the S&P500 (Market)
stocks_daily_return['sp500']


# In[13]:


# plotted a scatter plot between the selected stock and the S&P500 (Market)
stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'AAPL')


# In[14]:


# Fitted a polynomial between the selected stock and the S&P500 (Poly with order = 1 is a straight line)

# beta represents the slope of the line regression line (market return vs. stock return). 
# Beta is a measure of the volatility or systematic risk of a security or portfolio compared to the entire market (S&P500) 
# Beta is used in the CAPM and describes the relationship between systematic risk and expected return for assets 

# Beta = 1.0, this indicates that its price activity is strongly correlated with the market. 
# Beta < 1, indicates that the security is theoretically less volatile than the market (Ex: Utility stocks). If the stock is included, this will make the portfolio less risky compared to the same portfolio without the stock.
# Beta > 1, indicates that the security's price is more volatile than the market. For instance, Tesla stock beta is 1.26 indicating that it's 26% more volatile than the market. 
# Tech stocks generally have higher betas than S&P500 but they also have excess returns
# MGM is 65% more volatile than the S&P500!


beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['AAPL'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('AAPL', beta, alpha))  


# In[15]:


# Now let's plot the scatter plot and the straight line on one plot
stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'AAPL')

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
plt.plot(stocks_daily_return['sp500'], beta * stocks_daily_return['sp500'] + alpha, '-', color = 'r')


# # Calculated Beta for Tesla Inc. and Compared it with Apple

# In[16]:


# Fitted a polynomial between the selected stock and the S&P500 (Poly with order = 1 is a straight line)
beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['TSLA'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('TSLA', beta, alpha))
# Now let's plot the scatter plot and the straight line on one plot
stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'TSLA')

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
plt.plot(stocks_daily_return['sp500'], beta * stocks_daily_return['sp500'] + alpha, '-', color = 'r')


# # Apply the CAPM formula to an Individual Stock

# In[17]:


beta


# In[18]:


# Let's calculate the average daily rate of return for S&P500
stocks_daily_return['sp500'].mean()


# In[19]:


# Let's calculate the annualized rate of return for S&P500 
# Noted that out of 365 days/year, stock exchanges are closed for 104 days during weekend days (Saturday and Sunday) 
# Checked my answers with: https://dqydj.com/sp-500-return-calculator/
rm = stocks_daily_return['sp500'].mean() * 252
rm


# In[20]:


# Assumed risk free rate is zero
# Also you can use the yield of a 10-years U.S. Government bond as a risk free rate
rf = 0 

# Calculated return for any security (TSLA) using CAPM  
ER_TSLA = rf + (beta * (rm-rf)) 


# In[21]:


ER_TSLA


# # Applied CAPM formula to calculate the Return for AT&T

# In[22]:


# Calculated Beta for AT&T first
beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['T'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('T', beta, alpha)) 


# In[23]:


# Calculated return for AT&T using CAPM  
ER_T = rf + (beta * (rm - rf)) 
print(ER_T)


# # Calculated Beta for All Stocks

# In[24]:


# Let's create a placeholder for all betas and alphas (empty dictionaries)
beta = {}
alpha = {}

# Looped on every stock daily return
for i in stocks_daily_return.columns:

  # Ignored the date and S&P500 Columns 
  if i != 'Date' and i != 'sp500':
    # plotted a scatter plot between each individual stock and the S&P500 (Market)
    stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = i)
    
    # Fit a polynomial between each stock and the S&P500 (Poly with order = 1 is a straight line)
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
    
    plt.plot(stocks_daily_return['sp500'], b * stocks_daily_return['sp500'] + a, '-', color = 'r')
    
    beta[i] = b
    
    alpha[i] = a
    
    plt.show()


# In[25]:


# Let's view Beta for every stock 
beta


# In[26]:


# Let's view alpha for each of the stocks
# Alpha describes the strategy's ability to beat the market (S&P500)
# Alpha indicates the “excess return” or “abnormal rate of return,” 
# A positive 0.175 alpha for Tesla means that the portfolio’s return exceeded the benchmark S&P500 index by 17%.

alpha


# # Interactive Plot showing S&P500 Daily Returns Vs. Every Stock

# In[27]:


# Let's do the same plots but in an interactive way
# Explored some wierd points in the dataset: Tesla stock return was at 24% when the S&P500 return was -0.3%!

for i in stocks_daily_return.columns:
  
  if i != 'Date' and i != 'sp500':
    
    # Used plotly express to plot the scatter plot for every stock vs. the S&P500
    fig = px.scatter(stocks_daily_return, x = 'sp500', y = i, title = i)

    # Fit a straight line to the data and obtain beta and alpha
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
    
    # Plotted the straight line 
    fig.add_scatter(x = stocks_daily_return['sp500'], y = b*stocks_daily_return['sp500'] + a)
    fig.show()


# # Applied CAPM formula to Calculate the Return for the Portfolio

# In[28]:


# Obtained a list of all stock names
keys = list(beta.keys())
keys


# In[29]:


# Defined the expected return dictionary
ER = {}

rf = 0 # assumed risk free rate is zero in this case
rm = stocks_daily_return['sp500'].mean() * 252 # this is the expected return of the market 
rm


# In[30]:


for i in keys:
  # Calculated return for every security using CAPM  
  ER[i] = rf + (beta[i] * (rm-rf)) 


# In[31]:


for i in keys:
  print('Expected Return Based on CAPM for {} is {}%'.format(i, ER[i]))


# In[32]:


# Assumed equal weights in the portfolio
portfolio_weights = 1/8 * np.ones(8) 
portfolio_weights


# In[33]:


# Calculated the portfolio return 
ER_portfolio = sum(list(ER.values()) * portfolio_weights)
ER_portfolio


# In[34]:


print('Expected Return Based on CAPM for the portfolio is {}%\n'.format(ER_portfolio))


# # Calculated the Expected Return for the Portfolio assuming we only have 50% allocation in Apple and 50% in Amazon

# In[35]:


ER['AMZN']


# In[36]:


# Calculate the portfolio return 
ER_portfolio_2 = 0.5 * ER['AAPL'] +  0.5 * ER['AMZN']
ER_portfolio_2

