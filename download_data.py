import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
from sklearn import preprocessing

from yahoo_finance import Share
from keras.models import Model
from keras.layers import Input, Dense, Activation
import matplotlib.pyplot as plt

def get_model(model_config):
    inputs = Input(shape=(model_config['input'],))
    first = Dense(units=model_config['first'], activation='relu')(inputs)
    middle = Dense(units=model_config['middle'], activation='linear')(first)
    second = Dense(units=model_config['second'], activation='linear')(middle)
    _model = Model(inputs=inputs, outputs=second)
    _model.compile('adam', loss='mean_squared_error', metrics=['accuracy'])
    return _model


def calc_daily_returns(df):
    return np.subtract(np.log(df.iloc[1:]), np.log(df.iloc[:-1]))


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df.loc[:, :] = x_scaled


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
    for ticker in tickers:
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
    _size = 20
    tickers = tickers[:_size]

    model_config = {
        'input': _size,
        'first': 5,
        'middle': 10,
        'second': _size
    }

    sd = '2015-01-01'
    ed = '2019-01-01'
    df = create_data_obj(tickers, sd, ed)
    df = calc_daily_returns(df)
    _shape = df.shape
    x_train_bound = int(_shape[0]*3.0/4)
    X_train = df.iloc[x_train_bound:]
    X_test = df.iloc[:x_train_bound]
    # normalize(df)
    model = get_model(model_config)
    model.fit(X_train, X_train, batch_size=32, epochs=1000)
    res = model.evaluate(X_train, X_train, batch_size=32)
    print(res)
    Y = model.predict(X_train)

    plt.plot(range(X_train.shape[0]), X_train['MMM'], 'b-', label='input')
    plt.plot(range(X_train.shape[0]), Y[:, 0], 'r-', label='output')
    plt.xlabel('Date')
    plt.ylabel('Daily returns')
    plt.title('Daily returns: Input vs Reconstructed')
    plt.legend()
    plt.show()
