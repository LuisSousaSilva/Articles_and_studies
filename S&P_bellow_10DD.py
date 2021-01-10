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

# %%
# download dataframe using pandas_datareader
data = pdr.get_data_yahoo("^GSPC", start="1970-12-21", end="2021-04-30")[['Close']]

# %%
data.columns = ['S&P 500']

# %%
data['DD'] = pl.compute_drawdowns(data)
# %%
data['<10'] = data['DD'].apply(lambda x: 1 if x < -10 else 0)

# %%
data['Shade'] = True
# %%
data
# %%
for i in range(len(data) - 1):
    if data['<10'].iloc[i-1] != data['<10'].iloc[i+1]:
        data['Shade'].iloc[i] = 'Turn'
    else:
        data['Shade'].iloc[i] = 'Same'

for i in range(len(data) - 1):
    if data['<10'].iloc[i] == 0:
        data['Shade'].iloc[i] = 'Same'

#%%
data.to_csv('DATA.csv')
# %%
dates_below_10 = data[data['Shade'] == 'Turn'].index
dates_below_10
# %%
fig = data[['S&P 500']].iplot(color='royalblue',
                            xTitle='Valores sombreados quando o S&P está mais de 10% abaixo dos máximos históricos',
                            asFigure=True, dimensions=(1000, 500))

#%%
for i in range(2, int(len(dates_below_10) / 2)):
    end = i * 2
    start = end + 1

    # Add shaded days (below 10% drawdown)
    fig.add_vrect(
        x0=dates_below_10[start], x1=dates_below_10[end],
        fillcolor="grey", opacity=0.5,
        line_width=0),

fig.update_layout(title_text='S&P 500',
                            title_x=0.5)
                            
# %%
len(data[data['<10'] == 1]) / len(data)
# %%
len(data) - len(data[data['<10'] == 1])
# %%
len(data[data['<10'] == 1])
# %%
len(data)
# %%
pl.compute_drawdowns_table(data[['S&P 500']])