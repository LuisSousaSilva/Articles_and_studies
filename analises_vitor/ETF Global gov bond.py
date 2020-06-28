# %%
import pandas as pd
import numpy as np

# %%
# Downloading funds and creating quotes and returns dataframes
Begin = '2000-03-10'
# End = '2017-08-20' # Só activas se quiseres que acabe num dia especifíco 

Tickers = ['GAAA.LSE', 'EUN3.XETRA', 'DBZB.XETRA']

ETFs = pd.DataFrame()

# Download
for ticker in Tickers:
    url = "https://eodhistoricaldata.com/api/eod/" + str(ticker) + "?api_token=5c982bff80deb2.22427269&period=d."
    ETF = pd.read_csv(url, index_col = 'Date', parse_dates = True)[['Adjusted_close']].iloc[:-1, :]
    ETFs = ETFs.merge(ETF, left_index = True, right_index = True, how='outer')
    
ETFs.columns = Tickers
ETFs = ETFs.fillna(method='ffill').dropna()
ETFs = ETFs.replace(to_replace=0, method='ffill')

ETFs = ETFs/ETFs.iloc[0]

# %%
ETFs.plot(figsize=(12, 6));

# %%
