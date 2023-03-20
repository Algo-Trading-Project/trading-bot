import pandas as pd

from statsmodels.regression.rolling import RollingOLS
from backtesting import Strategy
from backtesting.lib import crossover, cross


class PairsTrading(Strategy):
    ########### INDICATOR PARAMETERS ##################
    z_window = 24
    hedge_ratio_window = 24
    
    z_thresh_upper = 1
    z_thresh_lower = -1
    
    X = 'QNT_ETH_GATEIO'
    Y = 'AAVE_ETH_BINANCE'
    ###################################################
    
    def rolling_hedge_ratios(X, Y, window):
        return RollingOLS(endog = Y, exog = X, window = window).fit().params
    
    def rolling_spread(X, Y, rolling_hedge_ratios):
        return Y - rolling_hedge_ratios * X
    
    def z(data, window):
        data = pd.Series(data.flatten())
        return (data - data.rolling(window = window).mean()) / data.rolling(window = window).std()
    
    def name():
        return 'PairsTrading'

    def update_hyperparameters(params_dict):
        for param_name, optimal_param_value in params_dict.items():
            setattr(PairsTrading, param_name, optimal_param_value)

    def init(self):
        # Initiate parent classes
        super().init()
        
        # Define indicator value arrays
        self.hedge_ratios = self.I(
            PairsTrading.rolling_hedge_ratios,
            pd.DataFrame(self.data[PairsTrading.X]),
            pd.DataFrame(self.data[PairsTrading.Y]),
            self.hedge_ratio_window
        )
        
        self.spread = self.I(
            PairsTrading.rolling_spread,
            self.data[PairsTrading.X],
            self.data[PairsTrading.Y],
            self.hedge_ratios
        )
        
        self.rolling_spread_z_score = self.I(
            PairsTrading.z,
            self.spread,
            self.z_window
        )

    # Define how the backtest acts at each step of iterating through the dataset
    def next(self):        
        entry_signal = (
            not crossover(self.rolling_spread_z_score, self.z_thresh_lower) and 
            cross(self.rolling_spread_z_score, self.z_thresh_lower)
        )
        exit_signal = crossover(self.rolling_spread_z_score, self.z_thresh_upper)

        # Close position
        if (self.position and exit_signal):
            self.position.close()

        # Open position--buy asset (this does not do shorting yet)
        elif (not self.position) and entry_signal:
            # Trailing stop adjustment
            self.buy(size = 0.1, sl = self.data.Close[-1] * 0.9)
            