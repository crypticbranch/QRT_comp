import numpy as np
import pandas as pd
from scipy.stats import rankdata

def rank(df):
    return df.rank(axis=1, pct=True)

def delay(df, period=1):
    return df.shift(period)

def correlation(x, y, window=10):
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    return x.rolling(window).cov(y)

def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum(axis=1), axis=0)

def delta(df, period=1):
    return df.diff(period)

def signedpower(x, a):
    return x ** a

def decay_linear(df, period=10):
    if df.isnull().values.any():
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    w = np.arange(period) + 1
    w = w / w.sum()
    return df.rolling(period).apply(lambda x: np.dot(x, w))

def ts_min(df, window=10):
    return df.rolling(window).min()

def ts_max(df, window=10):
    return df.rolling(window).max()

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin) + 1

def ts_rank(df, window=10):
    def _rolling_rank(na):
        return rankdata(na)[-1]
    return df.rolling(window).apply(_rolling_rank)

def ts_sum(df, window=10):
    return df.rolling(window).sum()

def ts_product(df, window=10):
    return df.rolling(window).apply(np.prod)

def ts_std(df, window=10):
    return df.rolling(window).std()

def sma(df, window=10):
    return df.rolling(window).mean()
