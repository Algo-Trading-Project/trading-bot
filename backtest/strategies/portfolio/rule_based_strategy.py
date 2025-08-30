from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY
from numba import njit
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import joblib
import numpy as np
import vectorbtpro as vbt
import numba

class RuleBasedStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'MLStrategyRuleBased',
    }

    optimize_dict = {
        # Maximum portfolio loss before exiting all positions and
        # preventing new positions from being entered for the rest of backtest period
        'max_loss': vbt.Param(np.array([
            0.1
        ])),

        # Expected value threshold for entering long positions
        'long_threshold': vbt.Param(np.array([
            0.0,
        ])),

        # Expected value threshold for entering short positions
        'short_threshold': vbt.Param(np.array([
            0.0,
        ])),

        # Number of positions per side
        'max_positions': vbt.Param(np.array([
            5
        ])),
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.00,  # 0.5% fees
        'slippage': 0.00,
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': "valuepercent",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load the ML features from the database
        cols = [
            'time_period_end', 'asset_id_base', 'symbol_id', 'open_spot', 'high_spot', 
            'low_spot', 'close_spot', 'open_futures', 'high_futures', 'low_futures', 
            'close_futures', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7', 
            'rolling_ema_30_cs_median_basis_pct_rz_30', 'dollar_volume_spot'
        ]
        ml_features = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet', columns=cols)
        ml_features.rename(columns={'time_period_end': 'time_period_open'}, inplace=True)
        ml_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        ml_features.set_index('time_period_open', inplace=True)
        self.ml_features = ml_features

        self.entries = []

    def calculate_size(self, volatilities):
        # Volatility targeting
        token_ann_vol = np.sqrt(365) * volatilities

        # Position size is volatility weighted
        ann_vol_target = 0.15  # Target annualized volatility of the portfolio
        max_positions = 20  # Maximum number of positions in the portfolio
        size = ann_vol_target / token_ann_vol / max_positions

        # Replace NaNs or infinite values with 0
        size = size.replace([np.inf, -np.inf, np.nan], 0)
        size = size.clip(lower = 0, upper = 0.05)
        return size

    def run_strategy_with_parameter_combination(
            self,
            universe: pd.DataFrame,
            max_loss: float,
            long_threshold: float,
            short_threshold: float,
            max_positions: int,
    ):
        universe.index = pd.to_datetime(universe.index)
        min_date = universe.index.min()
        max_date = universe.index.max()

        # Get features for this month
        feature_filter = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_spot'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD'))
        )

        spot_data = self.ml_features.loc[feature_filter]

        # Pivot futures model's predictions for ranking the models' signals by date
        long_model_probs = (
            spot_data
            .reset_index()[['time_period_open', 'symbol_id', 'rolling_ema_30_cs_median_basis_pct_rz_30', 'dollar_volume_spot']]
            .pivot_table(index='time_period_open', columns='symbol_id', values=['rolling_ema_30_cs_median_basis_pct_rz_30', 'dollar_volume_spot'], dropna=False)
        )
        long_model_probs = long_model_probs[long_model_probs.index.isin(universe.index)]
        long_entry_symbols = pd.Series(index = long_model_probs.index, dtype=object)

        for date in long_model_probs.index: 
            long_universe_ema = long_model_probs.loc[date, 'rolling_ema_30_cs_median_basis_pct_rz_30']
            long_universe_dollar_vol = long_model_probs.loc[date, 'dollar_volume_spot']
            top_symbols = long_universe_ema[long_universe_ema >= 3].index.tolist()
            if len(top_symbols) == 0:
                long_entry_symbols.loc[date] = []
                continue
            top_symbols = long_universe_dollar_vol.loc[top_symbols].sort_values(ascending=False).head(10).index.tolist()
            long_entry_symbols.loc[date] = list(top_symbols)

        print(long_entry_symbols)
        print()
        self.entries.append(long_entry_symbols.reset_index())

        long_entries = long_model_probs['rolling_ema_30_cs_median_basis_pct_rz_30'].copy() * 0
        long_entries = long_entries.astype(bool)

        for date in long_entry_symbols.index:
            long_symbols = long_entry_symbols.loc[date]
            if len(long_symbols) == 0:
                continue
            long_entries.loc[date, long_symbols] = True

        # Make the last exit in the long and short exits DataFrame True
        # to ensure exiting all positions at the end of the backtest
        long_exits = (long_entries.copy() * 0).astype(bool)
        long_exits.iloc[-1] = True
        long_exits = long_exits[long_exits.index.isin(universe.index)]

        # Decrement the time_period_open to the previous day for spot and futures data
        # (adjust right-labelled index to left-labelled index)
        # spot_data.index -= pd.Timedelta(days=1)
        # futures_data.index -= pd.Timedelta(days=1)
        # long_entries.index -= pd.Timedelta(days=1)
        # long_exits.index -= pd.Timedelta(days=1)
        long_entries = long_entries.shift(1).fillna(False)

        # Intersection of columns between the universe and the signals
        open_spot = spot_data.pivot_table(
            index='time_period_open',
            columns='symbol_id',
            values='open_spot',
            dropna=False
        )

        high_spot = spot_data.pivot_table(
            index='time_period_open',
            columns='symbol_id',
            values='high_spot',
            dropna=False
        )
        low_spot = spot_data.pivot_table(
            index='time_period_open',
            columns='symbol_id',
            values='low_spot',
            dropna=False
        )
        close_spot = spot_data.pivot_table(
            index='time_period_open',
            columns='symbol_id',
            values='close_spot',
            dropna=False
        )

        # Save these parameters temporarily
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        open = open_spot
        high = high_spot
        low = low_spot
        close = close_spot
        
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        long_exits.index = long_exits.index.astype('datetime64[ns]')
        open.index = open.index.astype('datetime64[ns]')
        high.index = high.index.astype('datetime64[ns]')
        low.index = low.index.astype('datetime64[ns]')
        close.index = close.index.astype('datetime64[ns]')

        long_entries = long_entries.astype('bool')

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            long_entries=long_entries,
            open=open,
            high=high,
            low=low, 
            close=close,
            price=-np.inf,
            size=0.05,
            sl_stop=1, 
            td_stop=pd.Timedelta(days=7),
            cash_sharing=True,
            accumulate=True,
            freq='D',
            **self.backtest_params
        ) 

        # Restore the parameters
        self.backtest_params['sl_stop'] = temp_sl_stop
        self.backtest_params['tp_stop'] = temp_tp_stop
        self.backtest_params['size'] = temp_size

        return portfolio