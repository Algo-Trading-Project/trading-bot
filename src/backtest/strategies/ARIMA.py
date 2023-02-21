from backtesting import Strategy
from pmdarima.arima import ARIMA
from ta.trend import EMAIndicator
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
        if len(self.trades) == 1 and self.data.index[-1] - self.trades[0].entry_time > pd.Timedelta(12, 'hours'):
            self.position.close()

        print('{}.) (p = {}, d = {}, q = {}) ema_window = {}'.format(len(self.data), self.p, self.d, self.q, self.ema_window))
        
        if len(self.data.Close) >= 50:
            model = ARIMA(order = (self.p, self.d, self.q))
            fit_model = model.fit(self.data.Close)                
            model_forecasts = fit_model.predict(n_periods = 1)
            
            entry_signal = (np.mean(model_forecasts) > self.data.Close[-1]) and (self.data.Close[-1] < self.ema[-1])
            exit_signal = (np.mean(model_forecasts) < self.data.Close[-1]) and (self.data.Close[-1] > self.ema[-1])

            if self.position and exit_signal:
                self.position.close()

            elif not self.position and entry_signal:
                self.buy(sl = self.data.Close[-1] * 0.95)
