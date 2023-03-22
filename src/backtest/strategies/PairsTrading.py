from backtesting import Strategy
from backtesting.lib import crossover, cross
from statsmodels.tools import add_constant
from statsmodels.regression.rolling import RollingOLS

import pandas as pd
import numpy as np

class PairsTrading(Strategy):
    ########### INDICATOR PARAMETERS ##################
    z_window = 72
    hedge_ratio_window = 72
    
    z_thresh_upper = 3
    z_thresh_lower = -2
    
    X = 'ETH_USD_COINBASE'
    Y = 'AAVE_ETH_BINANCE'
    ###################################################
        
    def z(data, window):
        data = pd.Series(data)
        return (data - data.rolling(window = window).mean()) / data.rolling(window = window).std()
    
    def h(self, window):
        
        X = pd.DataFrame(add_constant(self.data[PairsTrading.X]))
        X = X.rename({0:'intercept', 1:PairsTrading.X}, axis = 1)
        Y = pd.DataFrame(self.data[PairsTrading.Y])
        
        rolling_hedge_ratio = RollingOLS(
            endog = Y, 
            exog = X, 
            window = self.hedge_ratio_window).fit().params[PairsTrading.X]
        
        return rolling_hedge_ratio
    
    def s(self, h):
        return self.data[PairsTrading.Y] - h * self.data[PairsTrading.X]

    def name():
        return 'PairsTrading'

    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(PairsTrading, param_name, optimal_param_value)

    def init(self):
        # Initiate parent classes
        super().init()
        
        # Define indicator value arrays
        
        self.rolling_hedge_ratio = self.I(
            self.h,
            self.hedge_ratio_window
        )
        
        self.spread = self.I(
            self.s,
            self.rolling_hedge_ratio.flatten()
        )

        self.rolling_spread_z_score = self.I(
            PairsTrading.z,
            self.spread.flatten(),
            self.z_window
        )
        
    # Define how the backtest acts at each step of iterating through the dataset
    def next(self): 
        if self.rolling_hedge_ratio[-1] < 0:
            return
        
        
        entry_signal = (self.rolling_spread_z_score[-2] > self.z_thresh_lower) and (self.rolling_spread_z_score[-1] < self.z_thresh_lower)
        exit_signal = (self.rolling_spread_z_score[-2] < self.z_thresh_upper) and (self.rolling_spread_z_score[-1] > self.z_thresh_upper)
                
        # Close position
        if (self.position and exit_signal):
            self.position.close()

        # Open position
        elif (not self.position) and entry_signal:
            self.buy(size = 0.5, sl = self.data.Close[-1] * 0.9)