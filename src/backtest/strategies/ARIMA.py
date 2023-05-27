from backtesting import Strategy
from backtesting.lib import crossover

from statsmodels.tsa.arima.model import ARIMA

from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

import pandas as pd
import numpy as np

class ARIMAStrategy(Strategy):
   
    # ARIMA hyperparameters
    p = 1
    d = 0
    q = 1
    ema_window = 24
                
    def ema(open, ema_window):
        ema = EMAIndicator(close = open, window = ema_window)
        return ema.ema_indicator()

    def name():
        return 'ARIMA'
    
    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(ARIMAStrategy, param_name, optimal_param_value)

    def init(self):
        super().init()

        self.ema = self.I(ARIMAStrategy.ema, pd.Series(self.data.Close), self.ema_window)

    def next(self):
        if len(self.data.Close) >= 50:
            model = ARIMA(self.data.Close, order = (self.p, self.d, self.q))
            fit_model = model.fit()          
            model_forecast = fit_model.forecast()[0]

            price_slope = self.data.Close[-1] - self.data.Close[-2]
            
            entry_signal = ((model_forecast > self.data.Close[-1]) and 
                            (self.data.Close[-1] < self.ema[-1]) and 
                            (price_slope > 0))
            
            exit_signal = ((model_forecast < self.data.Close[-1]) and
                          (self.data.Close[-1] > self.ema[-1]) and 
                          (price_slope < 0))

            if self.position and exit_signal:
                self.position.close()

            elif not self.position and entry_signal:
                self.buy(sl = self.data.Close[-1] * 0.95, tp = self.data.Close[-1] * 1.1)
