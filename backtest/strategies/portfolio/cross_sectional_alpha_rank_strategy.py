from quantstats.stats import expected_return

from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from vectorbtpro import *

import pandas as pd
import numpy as np
import vectorbtpro as vbt

class CrossSectionalAlphaRankStrategy(BasePortfolioStrategy):
    indicator_factory_dict = {
        'class_name': 'CrossSectionalAlphaRankStrategy'
    }

    optimize_dict = {
        # Lookback period for sharpe ratio
        'lookback_period': vbt.Param(np.array([
            30, 90, 180
        ])),

        # Upper threshold for sharpe ratio
        'upper_threshold': vbt.Param(np.array([
            0.1, 0.15, 0.2
        ])),

        # Lower threshold for sharpe ratio
        'lower_threshold': vbt.Param(np.array([
            -0.2, -0.15, -0.1
        ])),
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.00,  # 0.1% maker/taker fees (Binance) + 0.25% spread each way
        'slippage': 0.00,  # 0.25% slippage each way
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': "valuepercent",  # for example, 2 if 'size' is 'Fixed Percent' and 0 otherwise,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ml_features = QUERY(
            """
            SELECT * FROM market_data.ml_features
            """
        )
        ml_features['symbol_id'] = ml_features['symbol_id'].astype('category')
        ml_features['time_period_end'] = pd.to_datetime(ml_features['time_period_end'], unit='ns')
        ml_features = ml_features.set_index('time_period_end').sort_index()
        self.ml_features = ml_features

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
        ann_vol_target = 0.4
        size = ann_vol_target / (token_ann_vol * betas.abs())

        # Replace NaNs or infinite values with 0
        size = size.replace([np.inf, -np.inf, np.nan], 0)

        # Ensure that the position sizes are between 0 and 0.05
        size = size.clip(0, 0.05)

        return size

    def generate_signals(self, lookback_period, upper_threshold, lower_threshold):
        sharpe_pivot = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', f'sharpe_ratio_1_{lookback_period}']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=f'sharpe_ratio_1_{lookback_period}', dropna=False)
        ).sort_index()
        prev_sharpe_pivot = sharpe_pivot.shift(1)

        # long entry when sharpe crosses below the lower threshold
        long_entries = (
            (prev_sharpe_pivot > lower_threshold) & (sharpe_pivot < lower_threshold)
        )
        # short entry when sharpe crosses above the upper threshold
        short_entries = (
            (prev_sharpe_pivot < upper_threshold) & (sharpe_pivot > upper_threshold)
        )

        return long_entries, short_entries

    def run_strategy_with_parameter_combination(
            self,
            universe: pd.DataFrame,
            lookback_period: int,
            upper_threshold: float,
            lower_threshold: float,
        ):
        universe.index = pd.to_datetime(universe.index)

        # Ensure that the signals are aligned with the universe
        long_entries, short_entries = self.generate_signals(lookback_period, upper_threshold, lower_threshold)

        long_entries = long_entries[long_entries.index.isin(universe.index)]
        short_entries = short_entries[short_entries.index.isin(universe.index)]

        # Pivot volatilities and betas for calculating position sizes
        volatilities = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'std_returns_1_180']]
            .pivot_table(index='time_period_end', columns='symbol_id', values='std_returns_1_180', dropna=False)
        ).sort_index()
        betas = (
            self.ml_features
            .reset_index()[['time_period_end', 'symbol_id', 'beta_1d_180d']]
            .pivot_table(index='time_period_end', columns='symbol_id', values='beta_1d_180d', dropna=False)
        ).sort_index()

        # Position sizes
        size = self.calculate_size(volatilities, betas)
        # Ensure that the signals are aligned with the universe
        size = size[size.index.isin(universe.index)]

        # Temporary variables to store the default values
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        # Intersection of columns between the universe and the signals
        common_cols = pd.Index(universe.close.columns.intersection(size.columns))
        open = universe.open[common_cols]
        high = universe.high[common_cols]
        low = universe.low[common_cols]
        close = universe.close[common_cols]
        long_entries = long_entries[common_cols]
        short_entries = short_entries[common_cols]
        size = size[common_cols]

        open.columns = high.columns = low.columns = close.columns = long_entries.columns = short_entries.columns = size.columns = common_cols

        # Make indices all the same datatype
        open.index = pd.to_datetime(open.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        close.index = pd.to_datetime(close.index)
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        short_entries.index = short_entries.index.astype('datetime64[ns]')
        size.index = size.index.astype('datetime64[ns]')

        # All signals are after start_date
        start_date = self.start_date
        long_entries = long_entries[long_entries.index >= start_date]
        short_entries = short_entries[short_entries.index >= start_date]
        open = open[open.index >= start_date]
        high = high[high.index >= start_date]
        low = low[low.index >= start_date]
        close = close[close.index >= start_date]
        size = size[size.index >= start_date]

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            long_entries=long_entries,
            short_entries=short_entries,
            open=open,
            high=high,
            low=low,
            close=close,
            size=size,
            cash_sharing=True,
            accumulate=False,
            upon_opposite_entry='reverse',
            freq='D',
            **self.backtest_params
        )

        # Restore the parameters
        self.backtest_params['sl_stop'] = temp_sl_stop
        self.backtest_params['tp_stop'] = temp_tp_stop
        self.backtest_params['size'] = temp_size

        return portfolio