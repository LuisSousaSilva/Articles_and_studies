#%%
# importing libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import cufflinks as cf
import seaborn as sns
import pandas as pd
import numpy as np
import quandl
import plotly
import time

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Markdown, display
from matplotlib.ticker import FuncFormatter
from pandas.core.base import PandasObject
from datetime import datetime

# Setting pandas dataframe display options
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 800)
pd.set_option('max_colwidth', 800)

# Set plotly offline
init_notebook_mode(connected=True)

# Set matplotlib style
plt.style.use('seaborn')

# Set cufflinks offline
cf.go_offline()

# Defining today's Date
from datetime import date
today = date.today()

from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

import PortfolioLab as pl

#%%
investments = pd.DataFrame(pd.date_range('1999', periods=41, freq='A'))
investments.columns = ['Date']
investments

#%%
investments['Age'] = 0

for i in range(25, 66):
    investments['Age'].iloc[i-25] = i

investments.set_index('Age', inplace=True)

#%%
investments
# %%
poupança = 2500
rentabilidade = 0.06

investments['25A'] = 0
investments = investments[['25A']]
# %%
for i in range(1, len(investments)):
    investments['25A'].iloc[i] = (investments['25A'].iloc[i - 1] * (1 + rentabilidade)) + poupança

# %%
investments['30A'] = 0

for i in range(6, len(investments)):
    investments['30A'].iloc[i] = (investments['30A'].iloc[i - 1] * (1 + rentabilidade)) + poupança
# %%
investments['40A'] = 0

for i in range(16, len(investments)):
    investments['40A'].iloc[i] = (investments['40A'].iloc[i - 1] * (1 + rentabilidade)) + poupança

# %%
investments['50A'] = 0

for i in range(26, len(investments)):
    investments['50A'].iloc[i] = (investments['50A'].iloc[i - 1] * (1 + rentabilidade)) + poupança
    
# %%
investments[['25A', '30A', '40A', '50A']].iplot()
# %%
#########################################################################################
################################## Com inflação #########################################
#########################################################################################
# %%
investments = pd.DataFrame(pd.date_range('1999', periods=41, freq='A'))
investments.columns = ['Date']
investments['Age'] = 0

for i in range(25, 66):
    investments['Age'].iloc[i-25] = i

investments.set_index('Age', inplace=True)

investments['Poupança'] = 2500

for i in range(26, 66):
    investments['Poupança'].iloc[i-25] = investments['Poupança'].iloc[i-1-25] * 1.02

investments['25A'] = 0

for i in range(1, len(investments)):
    investments['25A'].iloc[i] = (((investments['25A'].iloc[i - 1] * (1 + rentabilidade)) + investments['Poupança'].iloc[i])) * 0.98

investments['30A'] = 0

for i in range(6, len(investments)):
    investments['30A'].iloc[i] = ((investments['30A'].iloc[i - 1] * (1 + rentabilidade)) + investments['Poupança'].iloc[i]) * 0.98

investments['40A'] = 0

for i in range(16, len(investments)):
    investments['40A'].iloc[i] = ((investments['40A'].iloc[i - 1] * (1 + rentabilidade)) + investments['Poupança'].iloc[i]) * 0.98

investments['50A'] = 0

for i in range(26, len(investments)):
    investments['50A'].iloc[i] = ((investments['50A'].iloc[i - 1] * (1 + rentabilidade)) + investments['Poupança'].iloc[i]) * 0.98

investments[['25A', '30A', '40A', '50A']].iplot()
# %%
#########################################################################################
################################ With Deflator ##########################################
#########################################################################################
investments = pd.DataFrame(pd.date_range('1999', periods=41, freq='A'))
investments.columns = ['Date']
investments['Age'] = 0

for i in range(25, 66):
    investments['Age'].iloc[i-25] = i

investments.set_index('Age', inplace=True)

investments['Deflator'] = 100

for i in range(1, len(investments)):
    investments['Deflator'].iloc[i] = investments['Deflator'].iloc[i - 1] * 1.02

investments['Poupança'] = 2500

investments['Poupança'] = investments['Poupança'] * (investments['Deflator']) / 100

investments['25A'] = 0

for i in range(1, len(investments)):
    investments['25A'].iloc[i] = (((investments['25A'].iloc[i - 1] * (1 + rentabilidade)) + investments['Poupança'].iloc[i]))

investments['25A_adjusted'] = (investments['25A'] /  investments['Deflator']) * 100

investments
# %%
