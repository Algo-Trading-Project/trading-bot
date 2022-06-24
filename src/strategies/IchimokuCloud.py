from backtesting import Strategy
from backtesting.lib import crossover, cross

import pandas as pd
import numpy as np

class IchimokuCloud(Strategy):
    ########### INDICATOR PARAMETERS ##################

    # DEFAULT: 9, OPTIMAL: 18
    tenkan_window = 18

    # DEFAULT: 26, OPTIMAL: 24
    kijun_window = 24

    # DEFAULT: 52, OPTIMAL: 104
    senkou_b_window = 104

    # DEFAULT: 26, OPTIMAL: 24
    senkou_ab_period = 24
    
    #### TRAILING STOP LOSS (ATR-based) PARAMETERS ####

    # DEFAULT: 26, OPTIMAL: 14
    atr_period = 14

    # DEFAULT: 1, OPTIMAL: ?
    atr_stop_loss_multiplier = 1.5

    ############ TRADING PARAMETERS ###################

    # The minimum number of entry indicators to be true
    # before a buy signal is generated

    # DEFAULT: 1, OPTIMAL: 6
    en_bar = 8

    # The minimum number of exit indicators to be true
    # before a sell signal is generated

    # DEFAULT: 1, OPTIMAL: 7
    ex_bar = 4

    ###################################################

    # Tenkan-sen (Conversion Line)
    def TENKAN_SEN(d_high, d_low, tenkan_window):
        # DEFAULT:  (9-period high + 9-period low)/2))
        tenkan_period_high=pd.Series(d_high).rolling(window=tenkan_window).max()
        tenkan_period_low=pd.Series(d_low).rolling(window=tenkan_window).min()
        d_tenkan_sen=(tenkan_period_high+tenkan_period_low)/2
        return d_tenkan_sen

    # Kijun-sen (Base Line):
    def KIJUN_SEN(d_high, d_low, kijun_period):
        # DEFAULT:  (26-period high + 26-period low)/2))
        ichimoku_period_high=pd.Series(d_high).rolling(window=kijun_period).max()
        ichimoku_period_low=pd.Series(d_low).rolling(window=kijun_period).min()
        d_kijun_sen=(ichimoku_period_high+ichimoku_period_low)/2
        return d_kijun_sen

    # Senkou Span A (Leading Span A)
    def SENKOU_SPAN_A(d_high, d_low, tenkan_window, kijun_window, senkou_ab_period):
        # DEFAULT:  (9-period high + 9-period low)/2))
        tenkan_period_high=pd.Series(d_high).rolling(window=tenkan_window).max()
        tenkan_period_low=pd.Series(d_low).rolling(window=tenkan_window).min()
        d_tenkan_sen=(tenkan_period_high+tenkan_period_low)/2
        # DEFAULT:  (26-period high + 26-period low)/2))
        ichimoku_period_high=pd.Series(d_high).rolling(window=kijun_window).max()
        ichimoku_period_low=pd.Series(d_low).rolling(window=kijun_window).min()
        d_kijun_sen=(ichimoku_period_high+ichimoku_period_low)/2
        # DEFAULT:  (Conversion Line + Base Line)/2))
        d_senkou_a=pd.Series((d_tenkan_sen+d_kijun_sen)/2).shift(senkou_ab_period)
        return d_senkou_a

    # Senkou Span B (Leading Span B)
    def SENKOU_SPAN_B(d_high, d_low, senoku_b_window, senkou_ab_period):
        # DEFAULT:  (52-period high + 52-period low)/2))
        senkou_period_high = pd.Series(d_high).rolling(window=senoku_b_window).max()
        senkou_period_low = pd.Series(d_low).rolling(window=senoku_b_window).min()
        d_senkou_b = pd.Series((senkou_period_high + senkou_period_low)/2).shift(senkou_ab_period)
        return d_senkou_b

    # ATR: average true range over atr_period (DEFAULT: 14)
    def ATR(d_high, d_low, d_close, atr_period):
        d_high = pd.Series(d_high)
        d_low = pd.Series(d_low)
        d_close = pd.Series(d_close)

        high_low = d_high - d_low
        high_close = np.abs(d_high - d_close)
        low_close = np.abs(d_low - d_close)

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        # atr_period DEFAULT: is 14
        atr = true_range.rolling(atr_period).sum()/atr_period
        return atr

    def name():
        return 'IchimokuCloud'

    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(IchimokuCloud, param_name, optimal_param_value)

    def init(self):
        # Initiate parent classes
        super().init()
        
        # Define indicator value arrays
        self.atr = self.I(IchimokuCloud.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.tenkan_sen = self.I(IchimokuCloud.TENKAN_SEN, self.data.High, self.data.Low, self.tenkan_window)
        self.kijun_sen = self.I(IchimokuCloud.KIJUN_SEN, self.data.High, self.data.Low, self.kijun_window)
        self.senkou_span_a = self.I(IchimokuCloud.SENKOU_SPAN_A, self.data.High, self.data.Low, self.tenkan_window, self.kijun_window, self.senkou_ab_period)
        self.senkou_span_b = self.I(IchimokuCloud.SENKOU_SPAN_B, self.data.High, self.data.Low, self.senkou_b_window, self.senkou_ab_period)

    # Define how the backtest acts at each step of iterating through the dataset
    def next(self):
        close = self.data.Close[-1]
        atr = self.atr[-1]
        tenkan_sen = self.tenkan_sen[-1]
        kijun_sen = self.kijun_sen[-1]
        senkou_span_a = self.senkou_span_a[-1]
        senkou_span_b = self.senkou_span_b[-1]

        # List of position entry indicators
        entry_indicators = np.array([
            crossover(tenkan_sen, kijun_sen), # [1]: MACD crossover
            (tenkan_sen > (senkou_span_a if senkou_span_a > senkou_span_b else senkou_span_b)), # [2]: conversion line over the high part of the cloud
            (tenkan_sen > (senkou_span_b if senkou_span_a > senkou_span_b else senkou_span_a)), # [3]: conversion line over the low part of the cloud
            (senkou_span_a > senkou_span_b), # [4]: green cloud
            (close > (senkou_span_a if senkou_span_a > senkou_span_b else senkou_span_b)), # [5]: price over the high part of the cloud
            (close > (senkou_span_b if senkou_span_a > senkou_span_b else senkou_span_a)), # [6]: price over the low part of the cloud
            (close > tenkan_sen), # [7]: price is above the conversion line -- big signal of potentially triggered bullish run
            (close > kijun_sen) # [8]: price is above the base line -- more general signal of potential big run up
        ]).astype(int)

        entry_indicators_sum = entry_indicators.sum()
        
        # List of position exit indicators
        exit_indicators = np.array([
            (cross(tenkan_sen, kijun_sen) and not crossover(tenkan_sen, kijun_sen)) ,
            (tenkan_sen < senkou_span_a),
            (senkou_span_a < senkou_span_b),
            (close < senkou_span_a),
            (close < senkou_span_b),
            (close < tenkan_sen),
            (close < kijun_sen)
        ]).astype(int)

        exit_indicators_sum = exit_indicators.sum()

        # Entry and exit signals
        entry_signal = (entry_indicators_sum >= self.en_bar)
        exit_signal = (exit_indicators_sum >= self.ex_bar)

        # Close position
        if (self.position and exit_signal):
            self.position.close()

        # Open position--buy asset (this does not do shorting yet)
        elif (not self.position) and entry_signal:
            # Trailing stop adjustment
            stop_loss = close - (atr * self.atr_stop_loss_multiplier)
            self.buy(sl = stop_loss)