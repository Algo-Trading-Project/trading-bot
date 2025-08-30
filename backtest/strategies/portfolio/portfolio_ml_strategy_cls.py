from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
import pandas as pd
import joblib
import numpy as np
import vectorbtpro as vbt

class PortfolioMLStrategyCls(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategyCls',
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
        ml_features = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet')
        ml_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        ml_features.sort_values(by=['symbol_id', 'time_period_end'], inplace=True)
        ml_features.set_index('time_period_end', inplace=True)
        self.ml_features = ml_features
        self.long_entries = []
        self.short_entries = []

    def __get_model(self, min_date: pd.Timestamp): 
        year = min_date.year
        month = min_date.month
        model_path_long_reg = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_long_model_{year}_{month}_7d_sign.pkl'
        model_path_long_cls = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/lgbm_long_model_{year}_{month}_7d.pkl'
        try:
            model_path_short_reg = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/regression/lgbm_short_model_{year}_{month}_7d_sign.pkl'
            model_path_short_cls = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/lgbm_short_model_{year}_{month}_7d.pkl'
            model_short_reg = joblib.load(model_path_short_reg)
            model_short_cls = joblib.load(model_path_short_cls)
            model_short_reg.set_params(verbosity=-1)
            model_short_cls.set_params(verbosity=-1)
        except FileNotFoundError:
            model_short_reg = None
            model_short_cls = None

        model_long_reg = joblib.load(model_path_long_reg)
        model_long_cls = joblib.load(model_path_long_cls)
        model_long_reg.set_params(verbosity=-1)
        model_long_cls.set_params(verbosity=-1)

        return model_long_reg, model_long_cls, model_short_reg, model_short_cls

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
        model_long_reg, model_long_cls, model_short_reg, model_short_cls = self.__get_model(min_date)

        # Get features for this month
        feature_filter = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_spot'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD'))
        )
        feature_filter_futures = (
            (self.ml_features.index >= min_date) &
            (self.ml_features.index <= max_date) &
            (~self.ml_features['close_futures'].isna()) &
            (~self.ml_features['asset_id_base'].str.contains('USD')) 
        )

        spot_data = self.ml_features.loc[feature_filter]
        futures_data = self.ml_features.loc[feature_filter_futures]

        features_long = model_long_cls.feature_names_in_
        y_pred_long_reg = model_long_reg.predict(spot_data[features_long])
        y_pred_long_cls = model_long_cls.predict_proba(spot_data[features_long])[:, 1]
        expected_value_long = y_pred_long_reg * y_pred_long_cls

        if model_short_reg is not None and model_short_cls is not None:
            features_short = model_short_cls.feature_names_in_
            y_pred_short_reg = model_short_reg.predict(futures_data[features_short])
            y_pred_short_cls = model_short_cls.predict_proba(futures_data[features_short])[:, 1]
            expected_value_short = y_pred_short_reg * y_pred_short_cls
        else:
            y_pred_short_cls = np.nan * np.ones(futures_data.shape[0])
            expected_value_short = np.nan * np.ones(futures_data.shape[0])

        # Add the predictions to the DataFrame
        spot_data['expected_value_long'] = expected_value_long
        spot_data['pred_prob_long'] = y_pred_long_cls
        futures_data['expected_value_short'] = expected_value_short
        futures_data['pred_prob_short'] = y_pred_short_cls

        # Pivot futures model's predictions for ranking the models' signals by date
        long_model_probs = (
            spot_data
            .reset_index()[['time_period_end', 'symbol_id', 'pred_prob_long', 'expected_value_long', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=['pred_prob_long', 'expected_value_long', 'rolling_ema_30_cs_spot_returns_mean_7', 'rolling_ema_30_cs_futures_returns_mean_7'], dropna=False)
        )
        short_model_probs = (
            futures_data
            .reset_index()[['time_period_end', 'symbol_id', 'pred_prob_short', 'expected_value_short', 'rolling_ema_30_cs_spot_returns_mean_7']]
            .pivot_table(index='time_period_end', columns='symbol_id', values=['pred_prob_short', 'expected_value_short', 'rolling_ema_30_cs_spot_returns_mean_7'], dropna=False)
        )

        short_entry_symbols = pd.Series(index = short_model_probs.index, dtype=object)
        long_entry_symbols = pd.Series(index = long_model_probs.index, dtype=object)
        for date in long_model_probs.index:
            long_universe_ev = long_model_probs.loc[date, 'expected_value_long']
            long_universe_prob = long_model_probs.loc[date, 'pred_prob_long']
            long_universe_ev = long_universe_ev[long_universe_prob >= 0.5]
            
            try:
                short_universe_ev = short_model_probs.loc[date, 'expected_value_short']
                short_universe_prob = short_model_probs.loc[date, 'pred_prob_short']
                short_universe_ev = short_universe_ev[short_universe_prob >= 0.5]
            except KeyError:
                continue
            
            regime = long_model_probs.loc[date, 'rolling_ema_30_cs_spot_returns_mean_7'].mean()
            is_futures_available = not np.isnan(long_model_probs.loc[date, 'rolling_ema_30_cs_futures_returns_mean_7'].mean())    

            if regime > 0:
                # Bullish regime - prioritize longs
                top_symbols = long_universe_ev.nlargest(10).index.to_list()
                bottom_symbols = []
            else:
                if is_futures_available:
                    # Bearish regime - prioritize shorts
                    top_symbols = long_universe_ev.nlargest(5).index.to_list()
                    bottom_symbols = short_universe_ev.nsmallest(5).index.to_list() 
                else:
                    bottom_symbols = []
                    top_symbols = long_universe_ev.nlargest(10).index.to_list()

            long_entry_symbols.loc[date] = list(top_symbols)
            short_entry_symbols.loc[date] = list(bottom_symbols)

        if not futures_data.empty:
            short_entries = short_model_probs['expected_value_short'].copy() * 0
            short_entries = short_entries.astype(bool)
        else:
            short_entries = long_model_probs['expected_value_long'].copy() * 0
            short_entries = short_entries.astype(bool)

        long_entries = long_model_probs['expected_value_long'].copy() * 0
        long_entries = long_entries.astype(bool)

        for date in long_entry_symbols.index:
            long_symbols = long_entry_symbols.loc[date]
            long_entries.loc[date, long_symbols] = True
            try:
                short_symbols = short_entry_symbols.loc[date]
                short_entries.loc[date, short_symbols] = True
            except KeyError:
                continue

        self.long_entries.append(long_entry_symbols.reset_index())
        self.short_entries.append(short_entry_symbols.reset_index())

        empty_long = pd.DataFrame(
            np.zeros((long_entries.shape[0], long_entries.shape[1])),
            index=long_entries.index,
            columns=[col for col in long_entries.columns]
        )
        empty_short = pd.DataFrame(
            np.zeros((short_entries.shape[0], short_entries.shape[1])),
            index=short_entries.index,
            columns=[col for col in short_entries.columns]
        )

        # Merge the long and short entries w/ the empty DataFrames 
        # to have a consistent universe structure for VBT
        # long_entries = pd.merge(
        #     long_entries,
        #     empty_short,
        #     left_index=True,
        #     right_index=True,
        #     how='outer'
        # ).fillna(False)
        # short_entries = pd.merge(
        #     empty_long,
        #     short_entries,
        #     left_index=True,
        #     right_index=True,
        #     how='outer'
        # ).fillna(False)

        # Make the last exit in the long and short exits DataFrame True
        # to ensure exiting all positions at the end of the backtest
        long_exits = (long_entries.copy() * 0).astype(bool)
        long_exits.iloc[-1] = True
        short_exits = (short_entries.copy() * 0).astype(bool)
        short_exits.iloc[-1] = True

        # decrement 1 day from spot_data and futures_data time_period_end to have left-label alignment instead of right-label alignment
        spot_data.index = spot_data.index - pd.Timedelta(days=1)
        futures_data.index = futures_data.index - pd.Timedelta(days=1)

        # Intersection of columns between the universe and the signals
        open_spot = spot_data.pivot_table(index='time_period_end', columns='symbol_id', values='open_spot', dropna=False)
        open_futures = futures_data.pivot_table(index='time_period_end', columns='symbol_id', values='open_futures', dropna=False)

        high_spot = spot_data.pivot_table(index='time_period_end', columns='symbol_id', values='high_spot', dropna=False)
        high_futures = futures_data.pivot_table(index='time_period_end', columns='symbol_id', values='high_futures', dropna=False)

        low_spot = spot_data.pivot_table(index='time_period_end', columns='symbol_id', values='low_spot', dropna=False)
        low_futures = futures_data.pivot_table(index='time_period_end', columns='symbol_id', values='low_futures', dropna=False)
        
        close_spot = spot_data.pivot_table(index='time_period_end', columns='symbol_id', values='close_spot', dropna=False)
        close_futures = futures_data.pivot_table(index='time_period_end', columns='symbol_id', values='close_futures', dropna=False)

        # Save these parameters temporarily
        temp_sl_stop = self.backtest_params['sl_stop']
        temp_tp_stop = self.backtest_params['tp_stop']
        temp_size = self.backtest_params['size']

        # Remove these parameters from the backtest_params dictionary so that they aren't passed to the Portfolio
        del self.backtest_params['sl_stop']
        del self.backtest_params['tp_stop']
        del self.backtest_params['size']

        open = pd.merge(
            open_spot,
            open_futures,
            left_index=True,
            right_index=True,
            how='outer',
        ).astype('float32')
        high = pd.merge(
            high_spot,
            high_futures,
            left_index=True,
            right_index=True,
            how='outer',
        ).astype('float32')
        low = pd.merge(
            low_spot,
            low_futures,
            left_index=True,
            right_index=True,
            how='outer',
        ).astype('float32')
        close = pd.merge(
            close_spot,
            close_futures,
            left_index=True,
            right_index=True,
            how='outer',
        ).astype('float32')
        
        long_entries.index = long_entries.index.astype('datetime64[ns]')
        long_exits.index = long_exits.index.astype('datetime64[ns]')
        short_entries.index = short_entries.index.astype('datetime64[ns]')
        short_exits.index = short_exits.index.astype('datetime64[ns]')
        open.index = open.index.astype('datetime64[ns]')
        high.index = high.index.astype('datetime64[ns]')
        low.index = low.index.astype('datetime64[ns]')
        close.index = close.index.astype('datetime64[ns]')

        long_entries = long_entries.astype('bool')
        short_entries = short_entries.astype('bool')

        print(long_entries.shape, short_entries.shape)
        print(open.shape, high.shape, low.shape, close.shape)

        # Simulate Portfolio
        portfolio = vbt.Portfolio.from_signals(
            long_entries=False,
            long_exits=False,
            short_entries=False,
            short_exits=False,
            open=open,
            high=high,
            low=low,
            close=close,
            price = -np.inf,
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