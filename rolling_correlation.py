# %%
# Import libraries
import pandas as pd
import cufflinks as cf

import PortfolioLab as pl

# Set cufflinks offline
cf.go_offline()

#%%
# Pesquisa dos ETFs por ticker
pl.search_investing_etf(tickers=['DBXN'], visual='jupyter')

#%%
pl.search_investing_etf(isins=['LU0290355717'], visual='jupyter')

#%%
# # Fazer o download das cotações
names = ['iShares Core MSCI World UCITS', 'iShares Global Aggregate Bond Hedged Acc']
countries = ['netherlands', 'germany']
colnames = ['IWDA', 'AGGH']

etfs = pl.get_quotes_investing_etf(names=names, countries=countries, colnames=colnames)

returns = etfs.pct_change().dropna()

#%%
etfs
# %%
returns
#%%
IWDA_ret = returns['IWDA']
LMTC_ret = returns['AGGH']
# %%
corr = IWDA_ret.rolling('90D').corr(LMTC_ret)
corr
# %%
start = returns.index[0] + pd.DateOffset(days=90)

import numpy as np

corr[:start] = np.nan
# %%
corr.iplot(color='royalblue',
        title='90 day Correlation between IWDA.AS and AGGH.MI<br> (with average)',
        hline=dict(y=corr.mean(),color='indigo',width=1, dash="dash"))
# %%
corr.mean()
# %%
type(pl.search_investing_etf(isins=['LU1287023003'], visual='jupyter')
