from quantstats.stats import expected_return

from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from vectorbtpro import *

import pandas as pd
import numpy as np
import vectorbtpro as vbt

class MACrossoverStrategy(BasePortfolioStrategy):
    indicator_factory_dict = {
        'class_name': 'MACrossoverStrategy'
    }

    optimize_dict = {
        # Size of slow moving average window
        'slow_window': vbt.Param(np.array([
            60, 90, 120
        ])),

        # Size of fast moving average window
        'fast_window': vbt.Param(np.array([
            7, 14, 21, 30
        ])),
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0035,  # 0.1% maker/taker fees (Binance) + 0.25% spread each way
        'slippage': 0.0025,  # 0.25% slippage each way
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.1,
        'size_type': "valuepercent",  # for example, 2 if 'size' is 'Fixed Percent' and 0 otherwise,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_size(self, volatilities, betas):
        """
        Calculate position sizes from the tokens' volatilities and betas

        :param volatilities: DataFrame of volatilities for each token (columns) and date (index)
        :param betas: DataFrame of betas for each token (columns) and date (index)
        :return: DataFrame of position sizes for each token (columns) and date (index)
        """
        # Volatility targeting
        token_ann_vol = np.sqrt(365) * volatilities

        # Position size is calculated as the ratio of the target annual volatility to the token's annual volatility
        ann_vol_target = 0.6
        size = ann_vol_target / token_ann_vol

        # Replace NaNs or infinite values with 0
        size = size.replace([np.inf, -np.inf, np.nan], 0)

        # Ensure that the position sizes are between 0 and 0.05
        size = size.clip(0, 0.05)

        return size

    def run_strategy_with_parameter_combination(
            self,
            universe: pd.DataFrame,
            slow_window: int,
            fast_window: int,
    ):
        universe.index = pd.to_datetime(universe.index)

        # Exponentially weighted moving average of close prices (slow and fast)
        slow_ewma = universe.open.ewm(span=slow_window, min_periods=30).mean()
        fast_ewma = universe.open.ewm(span=fast_window, min_periods=7).mean()

        # Long entries
        long_entries = (
            (fast_ewma > slow_ewma) &
            (fast_ewma.shift(1) < slow_ewma.shift(1))
        )

        # Short entries
        short_entries = (
            (fast_ewma < slow_ewma) &
            (fast_ewma.shift(1) > slow_ewma.shift(1))
        )

        default_size = self.backtest_params['size']
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        open = universe.open
        high = universe.high
        low = universe.low
        close = universe.close

        # Make indices all the same datatype
        open.index = open.index.astype('datetime64[ns]')
        high.index = high.index.astype('datetime64[ns]')
        low.index = low.index.astype('datetime64[ns]')
        close.index = close.index.astype('datetime64[ns]')
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        short_entries.index = short_entries.index.astype('datetime64[ns]')

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            long_entries=long_entries,
            short_entries=short_entries,
            open=open,
            high=high,
            low=low,
            close=close,
            price = -np.inf,  
            size=0.05,
            cash_sharing=True,
            accumulate=False,
            freq='D',
            upon_opposite_entry = 'reverse',
            **self.backtest_params
        )

        # Restore the parameters
        self.backtest_params['sl_stop'] = temp_sl_stop
        self.backtest_params['tp_stop'] = temp_tp_stop
        self.backtest_params['size'] = temp_size

        return portfolio