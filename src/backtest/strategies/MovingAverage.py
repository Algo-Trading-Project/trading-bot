from backtesting import Strategy
from backtesting.lib import crossover, cross
import statsmodels.api as sm

from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
import pandas as pd
import numpy as np

class MovingAverage(Strategy):
    ########### INDICATOR PARAMETERS ##################
    ema_window_small = 24
    ema_window_large = 24 * 7
    lookback_window_trend = 24 * 3
    trend_threshold = 20
    ###################################################
    def ema(close, ema_window):
        return EMAIndicator(close = close, window = ema_window).ema_indicator()

    def calculate_trend(ema_large, lookback_period, thresh):
        to_return = []
        for i in range(len(ema_large)):
            if lookback_period != len(ema_large.iloc[i-lookback_period+1:i+1]):
                to_return.append(np.nan)
                continue
            
            period = ema_large.iloc[i-lookback_period+1:i+1]
            r = np.corrcoef(period, np.arange(1, len(period) + 1))[0][1]
        
            trend = np.sign(r)
            trend_strength = abs(r)

            if trend_strength <= thresh / 100:
                to_return.append(0)
            else:
                to_return.append(trend)

        return pd.Series(to_return)

    def name():
        return 'MovingAverage'

    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(MovingAverage, param_name, optimal_param_value)

    def init(self):
        # Initiate parent classes
        super().init()
        
        # Define indicator value arrays
        self.ema_small = self.I(MovingAverage.ema, pd.Series(self.data.Close), self.ema_window_small)
        self.ema_large = self.I(MovingAverage.ema, pd.Series(self.data.Close), self.ema_window_large)
        self.trend = self.I(MovingAverage.calculate_trend, pd.Series(self.ema_large), self.lookback_window_trend, self.trend_threshold)

    # Define how the backtest acts at each step of iterating through the dataset
    def next(self):
        trend = self.trend[-1]

        entry_signal = ((trend != -1 and crossover(self.ema_small, self.ema_large)) or 
                        (self.position and self.position.is_short and crossover(self.data.Close, self.ema_large))
                        )
        exit_signal = ((not crossover(self.data.Close, self.ema_large) and cross(self.data.Close, self.ema_large)) or
                       (trend != 1 and (not crossover(self.ema_small, self.ema_large) and cross(self.ema_small, self.ema_large)))
                       )

        if exit_signal:
            if self.position and self.position.is_long:
                self.position.close()
            elif not self.position:
                self.sell(sl = self.data.Close[-1] * 1.05)
            return
        
        # Open position--buy asset (this does not do shorting yet)
        if entry_signal:
            if self.position and self.position.is_short:
                self.position.close()
            elif not self.position:
                self.buy(sl = self.data.Close[-1] * 0.95)
            
        