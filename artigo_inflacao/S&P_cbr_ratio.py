# %%
import pandas as pd
import numpy as np
import PortfolioLab as pl

# %%
CBR = pd.read_csv('CRB.csv', index_col='Date', parse_dates=True)[['Close']]

CBR.columns = ['CBR']

# %%
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe using pandas_datareader
data = round(pdr.get_data_yahoo("^GSPC", start="1994-01-01", end="2022-04-30")[['Close']], 2)

# %%
data = pl.merge_time_series(data, CBR)
# %%
data.columns=['S&P', 'CBR']
data = pl.normalize(data)

pl.ichart(data) 
# %%
data['Ratio'] = data['S&P'] / data['CBR']
# %%
pl.ichart(data['Ratio']) 
# %%
data['Ratio'].iplot()
# %%
data.to_csv('SP500_CBR.csv')
# %%
data.to_csv('SP500_CBR_values.csv')
# %%
data.resample('M').last().pct_change().corr()
# %%
from numba import jit
import random
#%%
@jit(nopython=True)
def monte_carlo_pi_numba(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples
# %%
%%time
monte_carlo_pi(10000000)
# %%
def monte_carlo_pi_normal(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples
# %%
%%time
monte_carlo_pi_normal(10000000)
# %%
