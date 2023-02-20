from backtesting import Strategy
from statsmodels.tsa.arima.model import ARIMA

class ARIMAStrategy(Strategy):
   
    # ARIMA hyperparameters
    p = 1
    d = 0
    q = 1

    def name():
        return 'ARIMA'
    
    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(ARIMAStrategy, param_name, optimal_param_value)

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Open) >= 24:
            model = ARIMA(self.data.Open, order = (self.p, self.d, self.q))
            model_fit = model.fit()
            model_forecast = model_fit.forecast()[0]

            entry_signal = model_forecast > self.data.Open[-1]
            exit_signal = model_forecast < self.data.Open[-1]

            if self.position and exit_signal:
                self.position.close()

            elif not self.position and entry_signal:
                self.buy(sl = .1)

