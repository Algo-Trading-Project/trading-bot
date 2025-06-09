import numpy as np
import vectorbtpro as vbt

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

class BasePortfolioStrategy:

    # Set backtest parameters

    # init_cash - Initial cash
    # fees      - Commission percent
    # sl_stop   - Stop-loss percent
    # sl_trail  - Indicate whether you want a trailing stop-loss
    # tp_stop   - Take-profit percent
    # size      - Percentage of capital to use for each trade
    # size_type - Indicates the 'size' parameter represents a percent
    # cash_sharing - Indicate whether you want to share cash across assets

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

    metric_map = {
        'Total Return':'total_return',
        'Max Drawdown':'drawdowns.max_drawdown',
        'Win Rate':'trades.win_rate',
        'Sharpe Ratio':'sharpe_ratio',
        'Calmar Ratio':'calmar_ratio',
        'Omega Ratio':'omega_ratio',
        'Sortino Ratio':'sortino_ratio'
    }

    def __init__(self, *args, **kwargs):
        if kwargs.get('optimization_metric') is not None:
            self.optimization_metric = self.metric_map[kwargs.get('optimization_metric')]

        self.start_date = None

    @staticmethod
    def calculate_sl(universe, backtest_params, window):
        if type(backtest_params['sl_stop']) == float:
            sl = np.abs(np.full(universe.shape, backtest_params['sl_stop']))
            return sl

        elif backtest_params['sl_stop'] == 'std':
            # Calculate the rolling standard deviation over the entire universe
            # Resample to daily frequency
            rolling_std = universe.close.pct_change().rolling(window).std()
            # Calculate the stop-loss price as 2 times the rolling standard deviation of returns
            # below the close price
            sl = rolling_std
            return sl

        else:
            raise ValueError('Invalid stop-loss method')

    @staticmethod
    def calculate_tp(universe, backtest_params, window):
        if type(backtest_params['tp_stop']) == float:
            tp = np.abs(np.full(universe.shape, backtest_params['tp_stop']))
            return tp

        elif backtest_params['tp_stop'] == 'std':
            # Calculate the rolling standard deviation over the entire universe
            rolling_std = universe.close.pct_change().rolling(window).std()
            # Calculate the take-profit price as 2 times the rolling standard deviation of returns
            # above the close price
            tp = rolling_std
            return tp

        else:
            raise ValueError('Invalid take-profit method')

    # Make calculate_size a static method
    @staticmethod
    def calculate_size(universe, backtest_params, window):
        if type(backtest_params['size']) == float:
            size = np.abs(np.full(universe.shape, backtest_params['size']))
            return size

        elif backtest_params['size'] == 'std':
            # Calculate the rolling standard deviation over the entire universe
            rolling_std = universe.pct_change().rolling(window).std().values * 100

            # Calculate the position size as the percentage of the account
            # that should be risked on each trade divided by the standard deviation
            size = 0.1 / rolling_std
            return size

        else:
            raise ValueError('Invalid position sizing method')

    # Must be overridden by the child class with potentially a different signature
    # to allow for arbitrary parameters
    @staticmethod
    def run_strategy_with_parameter_combination(universe, **params):
        pass
    
    @vbt.parameterized(merge_func = 'concat')
    def indicator_func(self, universe, **params):
        # Run the strategy with the given parameter combination
        portfolio = self.run_strategy_with_parameter_combination(
            universe = universe,
            **params
        )
        optimization_metric = getattr(portfolio, self.optimization_metric)

        rename_dict = {
            'Entry Index':'entry_date', 'Exit Index':'exit_date',
            'Column':'symbol_id', 'Size':'size', 'Entry Fees':'entry_fees',
            'Exit Fees':'exit_fees', 'PnL':'pnl', 'Return':'pnl_pct', 'Direction':'is_long',
            'Status':'status'
        }
        cols = ['Entry Index', 'Exit Index', 'Column', 'Size', 'Entry Fees', 'Exit Fees', 'PnL', 'Return', 'Direction', 'Status']

        positions = portfolio.positions.records_readable
        positions = positions[cols]
        positions = positions.rename(rename_dict, axis = 1)

        # replace nan or inf values with 0
        if np.isnan(optimization_metric) or np.isinf(optimization_metric):
            optimization_metric = 0.0

        print(f'Optimization Metric: {optimization_metric}')
        print()

        # Return Portfolio Performance Metric based on self.optimization_metric
        return optimization_metric
