import numpy as np
import pandas as pd
from scripts.alpha_helpers import *

class Alpha101:
    def __init__(self, yf_data_df, returns_df):
        self.open = yf_data_df.xs('Open', level='Price', axis=1)
        self.high = yf_data_df.xs('High', level='Price', axis=1)
        self.low = yf_data_df.xs('Low', level='Price', axis=1)
        self.close = yf_data_df.xs('Close', level='Price', axis=1)
        self.volume = yf_data_df.xs('Volume', level='Price', axis=1)
        self.returns = returns_df

        # Remove duplicate ticker columns (e.g. multiple 'NAN' tickers from yf data)
        # This must happen before alignment, otherwise pandas operations like
        # rolling().corr() will fail with "'arg1' columns are not unique"
        for attr in ['open', 'high', 'low', 'close', 'volume']:
            df = getattr(self, attr)
            if df.columns.has_duplicates:
                setattr(self, attr, df.loc[:, ~df.columns.duplicated()])

        # Using typical price as an approximation for VWAP since it's missing in yf_data
        self.vwap = (self.high + self.low + self.close) / 3
        
        # Align indexes
        common_tickers = self.close.columns.intersection(self.returns.columns)
        self.open = self.open[common_tickers]
        self.high = self.high[common_tickers]
        self.low = self.low[common_tickers]
        self.close = self.close[common_tickers]
        self.volume = self.volume[common_tickers]
        self.vwap = self.vwap[common_tickers]
        self.returns = self.returns[common_tickers]
            
    def compute_all_alphas(self):
        alphas = {}
        # Get all methods that start with alpha_
        methods = [method for method in dir(self) if method.startswith('alpha_')]
        for method in methods:
            try:
                print(f"Computing {method}...")
                alphas[method] = getattr(self, method)()
            except Exception as e:
                print(f"Error computing {method}: {e}")
        return alphas

    # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha_001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = ts_std(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5)) - 0.5

    # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha_002(self):
        return -1 * correlation(rank(delta(np.log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        
    # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    def alpha_003(self):
        return -1 * correlation(rank(self.open), rank(self.volume), 10)

    # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    def alpha_004(self):
        return -1 * ts_rank(rank(self.low), 9)

    # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha_005(self):
        return rank((self.open - (ts_sum(self.vwap, 10) / 10))) * (-1 * np.abs(rank((self.close - self.vwap))))

    # Alpha#6: (-1 * correlation(open, volume, 10))
    def alpha_006(self):
        return -1 * correlation(self.open, self.volume, 10)

    # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    def alpha_007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * np.sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha_008(self):
        return -1 * rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)))

    # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha_009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close[cond_1 | cond_2]
        return alpha

    # Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha_010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close[cond_1 | cond_2]
        return rank(alpha)

    # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    def alpha_011(self):
        return (rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) * rank(delta(self.volume, 3))

    # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha_012(self):
        return np.sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha_013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha_014(self):
        return -1 * rank(delta(self.returns, 3)) * correlation(self.open, self.volume, 10)

    # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha_015(self):
        return -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)

    # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha_016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha_017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) * rank(delta(delta(self.close, 1), 1)) * rank(ts_rank((self.volume / adv20), 5)))

    # Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha_018(self):
        return -1 * rank((ts_std(np.abs(self.close - self.open), 5) + (self.close - self.open)) + correlation(self.close, self.open, 10))

    # Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha_019(self):
        return (-1 * np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) * (1 + rank(1 + ts_sum(self.returns, 250)))

    # Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha_020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) * rank(self.open - delay(self.close, 1)) * rank(self.open - delay(self.low, 1)))

    # Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha_021(self):
        cond_1 = sma(self.close, 8) + ts_std(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha

    # Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha_022(self):
        return -1 * (delta(correlation(self.high, self.volume, 5), 5) * rank(ts_std(self.close, 20)))

    # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha_023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2)[cond]
        return alpha

    # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha_024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))[cond]
        return alpha

    # Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha_025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))

    # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha_026(self):
        return -1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    def alpha_027(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        res = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        res[alpha > 0.5] = -1
        return res

    # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha_028(self):
        adv20 = sma(self.volume, 20)
        return scale(((correlation(adv20, self.low, 5) + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha_029(self):
        return (ts_min(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) + ts_rank(delay((-1 * self.returns), 6), 5))

    # Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha_030(self):
        delta_close = delta(self.close, 1)
        inner = np.sign(delta_close) + np.sign(delay(delta_close, 1)) + np.sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha_031(self):
        adv20 = sma(self.volume, 20)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10))))
        p2 = rank((-1 * delta(self.close, 3)))
        p3 = np.sign(scale(correlation(adv20, self.low, 12)))
        return p1 + p2 + p3

    # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha_032(self):
        return scale(((sma(self.close, 7) / 7) - self.close)) + (20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))

    # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
    def alpha_033(self):
        return rank(-1 + (self.open / self.close))

    # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha_034(self):
        inner = ts_std(self.returns, 2) / ts_std(self.returns, 5)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha_035(self):
        return ((ts_rank(self.volume, 32) * (1 - ts_rank(self.close + self.high - self.low, 16))) * (1 - ts_rank(self.returns, 32)))

    # Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha_036(self):
        adv20 = sma(self.volume, 20)
        return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (0.7 * rank((self.open - self.close)))) + (0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(np.abs(correlation(self.vwap, adv20, 6)))) + (0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open)))))

    # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha_037(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha_038(self):
        inner = self.close / self.open
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

    # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha_039(self):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20), 9))))) * (1 + rank(sma(self.returns, 250))))

    # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha_040(self):
        return -1 * rank(ts_std(self.high, 10)) * correlation(self.high, self.volume, 10)

    # Alpha#41: (((high * low)^0.5) - vwap)
    def alpha_041(self):
        return pow((self.high * self.low), 0.5) - self.vwap

    # Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    def alpha_042(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))

    # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha_043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    # Alpha#44: (-1 * correlation(high, rank(volume), 5))
    def alpha_044(self):
        return -1 * correlation(self.high, rank(self.volume), 5)

    # Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha_045(self):
        return -1 * (rank(sma(delay(self.close, 5), 20)) * correlation(self.close, self.volume, 2) * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha_046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha_047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) / 5))) - rank((self.vwap - delay(self.vwap, 5))))

    # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha_049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha

    # Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha_050(self):
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))

    # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha_051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha

    # Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha_052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) * rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))
        
    # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha_053(self):
        inner = (self.close - self.low)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha_054(self):
        inner = (self.low - self.high)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha_055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12))
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        return -1 * correlation(rank(inner), rank(self.volume), 6)

    # Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    def alpha_057(self):
        return (0 - (1 * ((self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2))))

    # Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    def alpha_060(self):
        divisor = (self.high - self.low)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
    
    # Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha_061(self):
        adv180 = sma(self.volume, 180)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))).astype(float)
    
    # Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha_062(self):
        adv20 = sma(self.volume, 20)
        return ((rank(correlation(self.vwap, sma(adv20, 22), 10)) < rank(((rank(self.open) + rank(self.open)) < (rank(((self.high + self.low) / 2)) + rank(self.high))).astype(float))) * -1).astype(float)
    # Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha_064(self):
        adv120 = sma(self.volume, 120)
        return ((rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13), sma(adv120, 13), 17)) < rank(delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 4))).astype(float) * -1)
    
    # Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    def alpha_065(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60, 9), 6)) < rank((self.open - ts_min(self.open, 14)))).astype(float) * -1)
      
    # Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha_066(self):
        return ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
    
    # Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    def alpha_068(self):
        adv15 = sma(self.volume, 15)
        return ((ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) < rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))).astype(float) * -1)
    
    # Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    def alpha_071(self):
        adv180 = sma(self.volume, 180)
        p1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18), 4), 16)
        p2 = ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))) ** 2), 16), 4)
        return pd.DataFrame(np.maximum(p1.values, p2.values), index=self.close.index, columns=self.close.columns)
    
    # Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    def alpha_072(self):
        adv40 = sma(self.volume, 40)
        return (rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) / rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3)))
    
    # Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    def alpha_073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        p2 = ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=self.close.index, columns=self.close.columns)
    
    # Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    def alpha_074(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, sma(adv30, 37), 15)) < rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11))).astype(float) * -1)
    
    # Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
    def alpha_075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50), 12))).astype(float)

    # Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    def alpha_077(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
        p2 = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3), 6))
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=self.close.index, columns=self.close.columns)
    
    # Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    def alpha_078(self):
        adv40 = sma(self.volume, 40)
        return (rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7)) ** rank(correlation(rank(self.vwap), rank(self.volume), 6)))

    # Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    def alpha_081(self):
        adv10 = sma(self.volume, 10)
        return ((rank(np.log(ts_product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)) ** 4)), 15))) < rank(correlation(rank(self.vwap), rank(self.volume), 5))).astype(float) * -1)
    
    # Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    def alpha_083(self):
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))
    
    # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    def alpha_084(self):
        return signedpower(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))
    
    # Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    def alpha_085(self):
        adv30 = sma(self.volume, 30)
        return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10)) ** rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7)))
    
    # Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    def alpha_086(self):
        adv20 = sma(self.volume, 20)
        return ((ts_rank(correlation(self.close, sma(adv20, 15), 6), 20) < rank(((self.open + self.close) - (self.vwap + self.open)))).astype(float) * -1)
    
    # Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    def alpha_088(self):
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8), 7), 3)
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=self.close.index, columns=self.close.columns)

    # Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    def alpha_092(self):
        adv30 = sma(self.volume, 30)
        p1 = ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).astype(float), 15), 19)
        p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8), 7), 7)
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=self.close.index, columns=self.close.columns)
    
    # Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    def alpha_094(self):
        adv60 = sma(self.volume, 60)
        return ((rank((self.vwap - ts_min(self.vwap, 12))) ** ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)) * -1)
    
    # Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    def alpha_095(self):
        adv40 = sma(self.volume, 40)
        return (rank((self.open - ts_min(self.open, 12))) < ts_rank((rank(correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13)) ** 5), 12)).astype(float)
    
    # Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    def alpha_096(self):
        adv60 = sma(self.volume, 60)
        p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8)
        p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=self.close.index, columns=self.close.columns)

    # Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    def alpha_098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        return (rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9), 7), 8)))
    
    # Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    def alpha_099(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) < rank(correlation(self.low, self.volume, 6))).astype(float) * -1)
    
    # Alpha#101: ((close - open) / ((high - low) + .001))
    def alpha_101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

# Function to run the alphas just like technical_indicators.py
def calculate_all_alphas(yf_data_df, returns_df):
    """
    Given the full YF data and returns DataFrames, compute all 101 Alphas
    where applicable (some are skipped if they require industry classification).
    """
    alphas_obj = Alpha101(yf_data_df, returns_df)
    alphas_dict = alphas_obj.compute_all_alphas()
    
    # Flatten the dict of DataFrames into a single MultiIndex DataFrame
    df_list = []
    for name, df in alphas_dict.items():
        if df is not None:
            # Drop NaN rows efficiently
            df = df.dropna(how='all')
            # Add Alpha name as top level column
            df.columns = pd.MultiIndex.from_product([[name], df.columns], names=['Alpha', 'Ticker'])
            df_list.append(df)
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, axis=1)

