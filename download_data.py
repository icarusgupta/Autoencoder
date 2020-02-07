import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

from yahoo_finance import Share


def pull_historical_data(ticker_symbol, sd, ed):
    ticker = pdr.get_data_yahoo(symbols=ticker_symbol, start=datetime.strptime(sd, '%Y-%m-%d'),
                                end=datetime.strptime(ed, '%Y-%m-%d'))
    ser = ticker['Adj Close']
    df = pd.DataFrame({ticker_symbol: ser})
    return df


def get_tickers():
    df = pd.read_csv('data/tickers.txt', sep='\t', header=None, encoding='utf_8')
    return df[0].tolist()


def create_data_obj(tickers, sd, ed):
    df = None
    for ticker in tickers[0:5]:
        try:
            df_tmp = pull_historical_data(ticker, sd, ed)
            if df is None:
                df = df_tmp
            else:
                df = df.merge(df_tmp, how='inner', left_index=True, right_index=True)
        except:
            pass
    return df


if __name__ == "__main__":

    tickers = get_tickers()
    sd = '2015-01-01'
    ed = '2019-01-01'
    create_data_obj(tickers, sd, ed)