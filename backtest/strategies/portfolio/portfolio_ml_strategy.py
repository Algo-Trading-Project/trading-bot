from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
import pandas as pd
import joblib
import numpy as np
import vectorbtpro as vbt

class PortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategy',
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
        'fees': 0.005,  # 0.5% fees
        'slippage': 0.001,  # 0.1% slippage
        'sl_stop': 'std',
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': "valuepercent",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load the ML features from the database
        ml_features = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet')
        ml_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        ml_features.sort_values(by=['symbol_id', 'time_period_end'], inplace=True)
        ml_features.set_index('time_period_end', inplace=True)
        # Cast columns to float
        cols = [col for col in ml_features.columns if col not in [
            'symbol_id', 'asset_id_base', 'asset_id_base_x', 'asset_id_base_y',
            'asset_id_quote', 'asset_id_quote_x', 'asset_id_quote_y', 
            'exchange_id', 'exchange_id_x', 'exchange_id_y',
            'month', 'day_of_week', 'time_period_end']
        ]
        ml_features[cols] = ml_features[cols].astype('float32')
        self.ml_features = ml_features

    def __get_model(self, min_date: pd.Timestamp): 
        year = min_date.year
        month = min_date.month
        model_path_long = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_long_model_{year}_{month}_7d.pkl'
        model_path_short = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_short_model_{year}_{month}_7d.pkl'
        model_long = joblib.load(model_path_long)
        model_long.set_params(verbosity=-1)
        try:
            model_short = joblib.load(model_path_short)
            model_short.set_params(verbosity=-1)
        except FileNotFoundError:
            model_short = None
        return model_long, model_short

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

        # Get models
        model_long, model_short = self.__get_model(min_date)

        # Get features for this month
        feature_filter = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_spot'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD')) &
            (self.ml_features['dollar_volume_spot'] >= 5_000_000)
        )

        spot_data = self.ml_features.loc[feature_filter]

        # Get LightGBM model features from each model
        features_long = model_long.feature_names_in_

        # Predictions on this month
        y_pred_long = model_long.predict(spot_data[features_long])
        
        # Add the predictions to the DataFrame
        spot_data['expected_value_long'] = y_pred_long

        # Pivot futures model's predictions for ranking the models' signals by date
        long_model_probs = (
            spot_data
            .reset_index()[['time_period_end', 'symbol_id', 'expected_value_long', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=['expected_value_long', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7'], dropna=False)
        )
        long_model_probs = long_model_probs[long_model_probs.index.isin(universe.index)]

        short_entry_symbols = pd.Series(index = long_model_probs.index, dtype=object)
        long_entry_symbols = pd.Series(index = long_model_probs.index, dtype=object)

        for date in long_model_probs.index:
            long_universe = long_model_probs.loc[date, 'expected_value_long']
            regime = long_model_probs.loc[date, 'rolling_ema_30_cs_spot_returns_mean_7'].mean()
            if regime > 0:
                bottom_symbols = []
                top_symbols = long_universe.nlargest(10).index.to_list()
            else:
                bottom_symbols = long_universe.nsmallest(5).index.to_list()
                top_symbols = long_universe.nlargest(5).index.to_list()

            long_entry_symbols.loc[date] = list(top_symbols)
            short_entry_symbols.loc[date] = list(bottom_symbols)

        short_entries = long_model_probs['expected_value_long'].copy() * 0
        short_entries = short_entries.astype(bool)
        long_entries = long_model_probs['expected_value_long'].copy() * 0
        long_entries = long_entries.astype(bool)

        for date in long_entry_symbols.index:
            long_symbols = long_entry_symbols.loc[date]
            long_entries.loc[date, long_symbols] = True

            short_symbols = short_entry_symbols.loc[date]
            short_entries.loc[date, short_symbols] = True

        # Trade signals on open of next candle, so shift by 1
        long_entries = long_entries.shift(1).fillna(False)
        short_entries = short_entries.shift(1).fillna(False)

        # Intersection of columns between the universe and the signals
        open = spot_data.pivot_table(
            index='time_period_end',
            columns='symbol_id',
            values='open_spot',
            dropna=False
        ).astype('float32')
        high = spot_data.pivot_table(
            index='time_period_end',
            columns='symbol_id',
            values='high_spot',
            dropna=False
        ).astype('float32')
        low = spot_data.pivot_table(
            index='time_period_end',
            columns='symbol_id',
            values='low_spot',
            dropna=False
        ).astype('float32')     
        close = spot_data.pivot_table(
            index='time_period_end',
            columns='symbol_id',
            values='close_spot',
            dropna=False
        ).astype('float32')

        # Save these parameters temporarily
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']
        
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        short_entries.index = short_entries.index.astype('datetime64[ns]')
        open.index = open.index.astype('datetime64[ns]')
        high.index = high.index.astype('datetime64[ns]')
        low.index = low.index.astype('datetime64[ns]')
        close.index = close.index.astype('datetime64[ns]')

        long_entries = long_entries.astype('bool')
        short_entries = short_entries.astype('bool')
        
        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            long_entries=long_entries,
            short_entries=short_entries,
            open=open,
            high=high,
            low=low,
            close=close,
            price = -np.inf, # Trade at open price
            size=0.1,
            sl_stop=1,
            td_stop=pd.Timedelta(days=7),
            cash_sharing=True,
            accumulate=False,
            freq='D',
            **self.backtest_params
        ) 

        # Restore the parameters
        self.backtest_params['sl_stop'] = temp_sl_stop
        self.backtest_params['tp_stop'] = temp_tp_stop
        self.backtest_params['size'] = temp_size

        return portfolio