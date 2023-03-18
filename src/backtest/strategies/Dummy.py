import pandas as pd
from backtesting import Strategy

class Dummy(Strategy):   

    def name():
        return 'Dummy'

    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(Dummy, param_name, optimal_param_value)

    def init(self):
        # Initiate parent classes
        super().init()
        
    # Define how the backtest acts at each step of iterating through the dataset
    def next(self):        
        if len(self.data) == 0 or len(self.data) == 1:
            self.buy()
            