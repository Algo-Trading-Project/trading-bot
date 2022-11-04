from backtesting import Strategy
from backtesting.lib import crossover, cross

import numpy as np
import pandas as pd

from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands
from ta.trend import IchimokuIndicator
from ta.trend import EMAIndicator


class BlueMoonBull_0(Strategy):
    entry_bar = 18
    exit_bar = 19

    ############## INDICATOR PARAMETERS ########################
     ##################### ICHIMOKU CLOUD ########################
    # [P1] DEFAULT: 9, OPTIMAL: 18
    tenkan_window = 9
    # [P2] DEFAULT: 26, OPTIMAL: 24
    kijun_window = 26
    # [P3] DEFAULT: 52, OPTIMAL: 104
    senkou_b_window = 52

      ##################### FOUR EMA ##############################
    # [P5] DEFAULT: 7, OPTIMAL: ?
    n1 = 7
    # [P6] DEFAULT: 12, OPTIMAL: ?
    n2 = 12
    # [P7] DEFAULT: 26, OPTIMAL: ?
    n3 = 26
    # [P8] DEFAULT: 108, OPTIMAL: ?
    n4 = 108

      ############### BOLLINGER BANDS ####################
    # [P12] DEFAULT: 20, OPTIMAL: ?
    bollinger_sma_window = 24
    # [P13] DEFAULT: 2, OPTIMAL: ?
    
    bollinger_window_std = 3

      ########### ATR (TRAILING STOP LOSS) #############
    # [P14] DEFAULT: 26, OPTIMAL: 14
    atr_window = 26
    # [P15] DEFAULT: 1, OPTIMAL: ?
    atr_stop_loss_multiplier = 1.5

    ############################## FUNCTIONS ###################################
    ######################### ICHIMOKU CLOUD #########################
    # Tenkan-sen (Conversion Line)
    def TENKAN_SEN(high, low, tenkan_window, kijun_window, senkou_b_window):
        high, low = pd.Series(high), pd.Series(low)

        ichimoku = IchimokuIndicator(
            high = high,
            low = low,
            window1 = tenkan_window,
            window2 = kijun_window,
            window3 = senkou_b_window,
            visual = False,
            fillna = True
        )

        return ichimoku.ichimoku_conversion_line()

    # Kijun-sen (Base Line):
    def KIJUN_SEN(high, low, tenkan_window, kijun_window, senkou_b_window):
        high, low = pd.Series(high), pd.Series(low)

        ichimoku = IchimokuIndicator(
            high = high,
            low = low,
            window1 = tenkan_window,
            window2 = kijun_window,
            window3 = senkou_b_window,
            visual = False,
            fillna = True
        )

        return ichimoku.ichimoku_base_line()

    # Senkou Span A (Leading Span A)
    def SENKOU_SPAN_A(high, low, tenkan_window, kijun_window, senkou_b_window):
        high, low = pd.Series(high), pd.Series(low)

        ichimoku = IchimokuIndicator(
            high = high,
            low = low,
            window1 = tenkan_window,
            window2 = kijun_window,
            window3 = senkou_b_window,
            visual = False,
            fillna = True
        )

        return ichimoku.ichimoku_a()

    # Senkou Span B (Leading Span B)
    def SENKOU_SPAN_B(high, low, tenkan_window, kijun_window, senkou_b_window):
        high, low = pd.Series(high), pd.Series(low)

        ichimoku = IchimokuIndicator(
            high = high,
            low = low,
            window1 = tenkan_window,
            window2 = kijun_window,
            window3 = senkou_b_window,
            visual = False,
            fillna = True
        )

        return ichimoku.ichimoku_b()

    ################################ FOUR EMA ################################
    # N_EMA (an N size grouping of multiple EMA's a.k.a. exponential moving averages)
    # data is the 'Close' column of the price feed of the token with the timestamp as the index
    # ema_days is the number of days before from which the weighted sum starts
    # ema_smoothings is the smoothing factor ALPHA (ranges from 0.0 <= ALPHA <= 1.0)
    def N_EMA(close, ema_window):
        close = pd.Series(close)

        ema = EMAIndicator(
            close = close,
            window = ema_window,
            fillna = True
        )

        return ema.ema_indicator()

    ######################### BOLLINGER BANDS #########################
    # Values for upper Bollinger Band
    def BOLLINGER_H(close, bollinger_sma_window, bollinger_window_std):
        close = pd.Series(close)

        # Calculate Simple Moving Average with 20 days window
        bolllinger = BollingerBands(
            close = close,
            window = bollinger_sma_window,
            window_dev = bollinger_window_std,
            fillna = True
        )

        return bolllinger.bollinger_hband()

    # Values for lower Bollinger Band
    def BOLLINGER_L(close, bollinger_sma_window, bollinger_window_std):
        close = pd.Series(close)

        # Calculate Simple Moving Average with 20 days window
        bolllinger = BollingerBands(
            close = close,
            window = bollinger_sma_window,
            window_dev = bollinger_window_std,
            fillna = True
        )

        return bolllinger.bollinger_lband()

    ################################## ATR ##################################
    # ATR: average true range over atr_period (DEFAULT: 14)
    def ATR(high, low, close, atr_window):
        high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
        atr = AverageTrueRange(
            high = high,
            low = low,
            close = close, 
            window = atr_window,
            fillna = True
        )

        return atr.average_true_range()

    ################ BACKTESTER OBJECT FUNCTIONS ################
    # Define the name of the strategy object
    def name():
        return 'BlueMoonBull_0'

    ###################### STRATEGY INIT() ######################
    # Initialize using backtesting.py's init() method
    def init(self):
        # Initiate parent classes
        super().init()

        ################ INDICATOR I() DEFINITONS ################
        self.atr = self.I(BlueMoonBull_0.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_window)
        self.tenkan_sen = self.I(BlueMoonBull_0.TENKAN_SEN, self.data.High, self.data.Low, self.tenkan_window, self.kijun_window, self.senkou_b_window)
        self.kijun_sen = self.I(BlueMoonBull_0.KIJUN_SEN, self.data.High, self.data.Low, self.tenkan_window, self.kijun_window, self.senkou_b_window)
        self.senkou_span_a = self.I(BlueMoonBull_0.SENKOU_SPAN_A, self.data.High, self.data.Low, self.tenkan_window, self.kijun_window, self.senkou_b_window)
        self.senkou_span_b = self.I(BlueMoonBull_0.SENKOU_SPAN_B, self.data.High, self.data.Low, self.tenkan_window, self.kijun_window, self.senkou_b_window)
        
        self.ema_1 = self.I(BlueMoonBull_0.N_EMA, self.data.Close, self.n1)
        self.ema_2 = self.I(BlueMoonBull_0.N_EMA, self.data.Close, self.n2)
        self.ema_3 = self.I(BlueMoonBull_0.N_EMA, self.data.Close, self.n3)
        self.ema_4 = self.I(BlueMoonBull_0.N_EMA, self.data.Close, self.n4)

        self.bollinger_h = self.I(BlueMoonBull_0.BOLLINGER_H, self.data.Close, self.bollinger_sma_window, self.bollinger_window_std)
        self.bollinger_l = self.I(BlueMoonBull_0.BOLLINGER_L, self.data.Close, self.bollinger_sma_window, self.bollinger_window_std)
   
    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(BlueMoonBull_0, param_name, optimal_param_value)

    ########################## STRATEGY NEXT() ##########################
    # Define how the backtest acts at each step of iterating through the dataset
    def next(self):
        ############# DEFINING PARAMETERS PER INSTANCE #############
        # ICHIMOKU
        tenkan_sen = self.tenkan_sen[-1]
        kijun_sen = self.kijun_sen[-1]
        senkou_span_a = self.senkou_span_a[-1]
        senkou_span_b = self.senkou_span_b[-1]

        high_cloud_series = (self.senkou_span_a if self.senkou_span_a > self.senkou_span_b else self.senkou_span_b)
        low_cloud_series = (self.senkou_span_b if self.senkou_span_b > self.senkou_span_a else self.senkou_span_a)

        high_cloud = high_cloud_series[-1]
        low_cloud = low_cloud_series[-1]

        cloud_color = ('GREEN' if (senkou_span_a > senkou_span_b) else 'RED')
        # FOUR EMA
        ema_1 = self.ema_1[-1]
        ema_2 = self.ema_2[-1]
        ema_3 = self.ema_3[-1]
        ema_4 = self.ema_4[-1]
        
        # BOLLINGER BANDS
        bollinger_h = self.bollinger_h[-1]
        bollinger_l = self.bollinger_l[-1]

        # ATR
        atr = self.atr[-1]

        # Price value (set to the closing value of each candle)
        close = self.data.Close[-1]

        # Trailing stop adjustment
        current_stop_loss = close - (atr * self.atr_stop_loss_multiplier)

        ################ ENTRY AND EXIT DECISIONING ################
        # Update training stop loss if the current_stop_loss value is above the set self.stop_loss value
        # AND the current_stop_loss value is large enough that profit above commission costs is guaranteed
            # * The second clause exists so that we don't get stopped out of good opportunities by
            #  being riskier with a dynamic stop loss when we have not yet guaranteed at least break-even

        # BELOW LIKELY HAS A BUG

        # if (self.position and (current_stop_loss > self.stop_loss) and (current_stop_loss > (self.trade_entry_price * self.commission))):
        #     self.stop_loss = current_stop_loss

        # List of position entry indicators
        entry_indicators = np.array([
            crossover(self.tenkan_sen, self.kijun_sen), # [EN1]: MACD crossover
            (tenkan_sen > kijun_sen),

            (cloud_color == 'GREEN'), # [EN2]: green cloud

            crossover(self.tenkan_sen, high_cloud_series),
            (tenkan_sen > high_cloud), # [EN3]: conversion line over the high part of the cloud

            crossover(self.tenkan_sen, low_cloud_series),
            (tenkan_sen > low_cloud), # [EN4]: conversion line over the low part of the cloud

            crossover(self.data.Close, high_cloud_series),
            (close > high_cloud), # [EN5]: price over the high part of the cloud

            crossover(self.data.Close, low_cloud_series),
            (close > low_cloud), # [EN6]: price over the low part of the cloud

            crossover(self.data.Close, self.kijun_sen),
            (close > kijun_sen), # [EN7]: price is above the base line -- more general signal of potential big run up

            crossover(self.data.Close, self.tenkan_sen),
            (close > tenkan_sen), # [EN8]: price is above the conversion line -- big signal of potentially triggered bullish run

            (ema_1 > ema_2 > ema_3 > ema_4), # [EN9]: the 4EMA aligns
            (close > ema_1 > ema_2), # [EN10]: the price is above the first two EMA's, when they are aligned. This signals that everything is in place
            (close < (close + ((bollinger_h - close) / 3))) # [E12]: closing price is not above the middle of the upper the bollinger band
        ]).astype(int)

        # Calculate the initial sum for entry decisioning
        entry_indicators_sum = entry_indicators.sum()
        
        # List of position exit indicators
        exit_indicators = np.array([
            (cross(self.tenkan_sen, self.kijun_sen) and not crossover(self.tenkan_sen, self.kijun_sen)), # [EX]: Conversion line dips below base line
            (tenkan_sen < kijun_sen),

            (cross(self.tenkan_sen, high_cloud_series) and not crossover(self.tenkan_sen, high_cloud_series)),
            (tenkan_sen < high_cloud), # [EX]: Conversion line dips into cloud

            (cloud_color == 'RED'), # [EX]: Cloud turns red from green

            (cross(self.data.Close, self.tenkan_sen) and not crossover(self.data.Close, self.tenkan_sen)),
            (close < tenkan_sen), # [EX]: Price dips below conversion

            (cross(self.data.Close, self.kijun_sen) and not crossover(self.data.Close, self.kijun_sen)),
            (close < kijun_sen), # [EX]: Price dips below base

            (cross(self.data.Close, high_cloud_series) and not crossover(self.data.Close, high_cloud_series)),
            (close < high_cloud), # [EX]: Price dips below high cloud

            (cross(self.data.Close, low_cloud_series) and not crossover(self.data.Close, low_cloud_series)),
            (close < low_cloud), # [EX]: Price dips below low cloud

            (not (ema_1 > ema_2 > ema_3 > ema_4)), # [EX]: the 4EMA un-aligns
            (close < ema_1 < ema_2), # [EX]: Sizable trend downward confirmation

            (cross(self.ema_2, self.ema_3) and not crossover(self.ema_2, self.ema_3)),
            (ema_2 < ema_3), # [EX]: Another sizable trend downward confirmation

            crossover(self.data.Close, self.bollinger_h),
            (close > bollinger_h) # [EX]: Price has gone over the high bollinger band <<LIKELY MOST USEFUL>>
        ]).astype(int)
        
        # Calculate the intiial sum for exit decisioning
        exit_indicators_sum = exit_indicators.sum()

        # Entry and exit signals
        entry_signal = (entry_indicators_sum >= self.entry_bar)
        exit_signal = (exit_indicators_sum >= self.exit_bar)

        ######################## EXECUTING TRADES ########################
        # Close position if exit signal is triggered or the close price is below our set stop loss
        if (self.position and exit_signal):
            self.position.close()

        # Open position--buy asset (this does not do shorting yet)
        elif ((not self.position) and (entry_signal)):
            self.buy(sl = current_stop_loss) 
