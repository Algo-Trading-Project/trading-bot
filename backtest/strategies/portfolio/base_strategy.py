import numpy as np
import pandas as pd
import vectorbtpro as vbt

class BasePortfolioStrategy:

    # Set backtest parameters
    
    # init_cash - Initial cash
    # fees      - Comission percent
    # sl_stop   - Stop-loss percent
    # sl_trail  - Indicate whether or not want a trailing stop-loss
    # tp_stop   - Take-profit percent
    # size      - Percentage of capital to use for each trade
    # size_type - Indicates the 'size' parameter represents a percent
    # cash_sharing - Indicate whether or not want to share cash across assets

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0029,
        'sl_stop': 0.1,
        'sl_trail': True,
        'tp_stop': 0.1,
        'size': 0.05,
        'size_type': 2, # e.g. 2 if 'size' is 'Fixed Percent' and 0 otherwise,
        'cash_sharing': True
    }

    optimize_dict = {}

    indicator_factory_dict = {
        'class_name':'BaseStrategy',
        'short_name':'base_strategy',
        'input_names':['universe'],
        'param_names':[],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    # TODO: make calculate_sl based on returns instead of close price
    def calculate_sl(self, universe, backtest_params, window):
        if type(backtest_params['sl_stop']) == float:
            sl = np.abs(np.full(universe.shape, backtest_params['sl_stop']))
            return sl

        elif backtest_params['sl_stop'] == 'std':
            # Calculate the rolling standard deviation over the entire universe
            rolling_std = universe.pct_change().rolling(window, min_periods = 1).std()
            # Calculate the stop-loss price as 2 times the rolling standard deviation of returns
            # below the close price
            sl = 2 * rolling_std
            return sl

        else:
            raise ValueError('Invalid stop-loss method')

    # TODO: make calculate_tp based on returns instead of close price
    def calculate_tp(self, universe, backtest_params, window):
        if type(backtest_params['tp_stop']) == float:
            tp = np.abs(np.full(universe.shape, backtest_params['tp_stop']))
            return tp

        elif backtest_params['tp_stop'] == 'std':
            # Calculate the rolling standard deviation over the entire universe
            rolling_std = universe.pct_change().rolling(window, min_periods = 1).std()
            # Calculate the take-profit price as 2 times the rolling standard deviation of returns
            # above the close price
            tp = 2 * rolling_std
            return tp

        else:
            raise ValueError('Invalid take-profit method')

    # TODO: make calculate_size based on returns instead of close price
    def calculate_size(self, universe, backtest_params, window):
        if type(backtest_params['size']) == float:
            size = np.abs(np.full(universe.shape, backtest_params['size']))
            return size

        elif backtest_params['size'] == 'std':
            # Calculate the rolling standard deviation over the entire universe
            rolling_std = universe.pct_change().rolling(window, min_periods = 1).std().values * 100

            # Calculate the position size as the percentage of the account
            # that should be risked on each trade divided by the standard deviation
            size = 0.1 / rolling_std
            return size
            
        else:
            raise ValueError('Invalid position sizing method')

    # Must be overridden by the child class
    def run_strategy_with_parameter_combination(self, universe, **params):
        pass
    
    @vbt.parameterized(merge_func = 'concat')
    def indicator_func(self, universe, **params):
        # Run the strategy with the given parameter combination
        portfolio = self.run_strategy_with_parameter_combination(
            universe = universe,
            **params
        )
        # Return Portfolio Sortino Ratio
        return portfolio.sortino_ratio