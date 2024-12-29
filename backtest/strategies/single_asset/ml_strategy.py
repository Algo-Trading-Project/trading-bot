import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import joblib

from backtest.strategies.single_asset.base_strategy import BaseStrategy
from xgboost import XGBClassifier
from utils.db_utils import QUERY

class MLStrategy(BaseStrategy):

    indicator_factory_dict = {
        'class_name':'MLStrategy',
        'short_name':'ml_strategy',
        'input_names':['open', 'high', 'low', 'close', 'volume', 'trades_count'],
        'param_names':['prediction_threshold_entry', 'trade_size_multiplier'],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    optimize_dict = {
        # Minimum predicted probability for entering a trade
        'prediction_threshold_entry': [0.7],

        # Multiplier for position size based on predicted class probabilities
        'trade_size_multiplier': [0.1]
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.00,
        'size': 0.1,
        'size_type': 2,
        'sl_stop': 'std',
        'sl_trail': False,
        'tp_stop': 'std',
    }

    def __init__(self):
        self.symbol_id = ''
        # Map each month to their previous quarter
        self.month_to_quarter_map = {
            1: 'oct_to_dec',
            2: 'oct_to_dec',
            3: 'oct_to_dec',
            4: 'jan_to_march',
            5: 'jan_to_march',
            6: 'jan_to_march',
            7: 'april_to_june',
            8: 'april_to_june',
            9: 'april_to_june',
            10: 'july_to_sept',
            11: 'july_to_sept',
            12: 'july_to_sept'
        }

    def __get_model(self, min_date: pd.Timestamp):
        min_year = min_date.year
        min_month = min_date.month

        # Get the previous quarter
        quarter = self.month_to_quarter_map[min_month]

        # If the quarter is the first quarter of the year, then the previous quarter is the last quarter of the previous year
        if quarter == 'oct_to_dec':
            min_year -= 1

        path = f'/Users/louisspencer/Desktop/Trading-Bot/data/pretrained_models/classification/xgboost_model_{min_year}_{quarter}.pkl'

        # Load the model
        model = joblib.load(path)
        return model

    def indicator_func(self, open, high, low, close, volume, trades_count,
                       prediction_threshold_entry, trade_size_multiplier):
        
        # Get backtest parameters
        backtest_params = MLStrategy.backtest_params

        # Create OHLCV dataframe
        price_data = pd.DataFrame({
            'open': open,
            'high': high,
            'low': low,
            'close': close,
            'volume_traded': volume,
            'trades_count': trades_count
        })
        price_data = price_data.astype(float)

        # Deduplicate the index of the price data
        price_data = price_data[~price_data.index.duplicated(keep = 'first')]

        start_date_oos = price_data.index[0]
        end_date_oos = price_data.index[-1]

        print('Start Date:', start_date_oos)
        print('End Date:', end_date_oos)
        print()

        base, quote, exchange = self.symbol_id.split('_')
        query = f"""
            SELECT *
            FROM market_data.ml_features
            WHERE
                time_period_end BETWEEN '{start_date_oos}' AND '{end_date_oos}' AND
                asset_id_base = '{base}' AND
                asset_id_quote = '{quote}' AND
                exchange_id = '{exchange}'
            ORDER BY time_period_end
        """

        out_of_sample_data = QUERY(query).set_index('time_period_end')
        out_of_sample_data['symbol_id'] = out_of_sample_data['symbol_id'].astype('category')
        out_of_sample_data = out_of_sample_data.replace([np.inf, -np.inf], np.nan)
        
        windows_triple_barrier_label = [7]
        max_holding_times_triple_barrier_label = [7]

        triple_barrier_label_cols = [
            f'triple_barrier_label_h{h}' 
            for h in max_holding_times_triple_barrier_label
        ]

        trade_returns_cols = [
            f'trade_returns_h{h}' 
            for h in max_holding_times_triple_barrier_label
        ]

        non_numeric_cols = [
            'asset_id_base', 'asset_id_quote', 'exchange_id', 'Unnamed: 0',
        ]

        forward_returns_cols = [
            'open', 'high', 'low', 'close', 'start_date_triple_barrier_label_h7', 
            'end_date_triple_barrier_label_h7', 'avg_uniqueness', 'time_period_end'
        ]


        cols_to_drop = triple_barrier_label_cols + trade_returns_cols + non_numeric_cols + forward_returns_cols
        out_of_sample_data = out_of_sample_data.drop(columns = cols_to_drop + ['asset_id_base', 'asset_id_quote', 'exchange_id'], errors = 'ignore')

        # Import pretrained XGBoost model
        model = self.__get_model(min_date = start_date_oos)

        # Predict the class probabilities
        y_pred = model.predict(out_of_sample_data)
        out_of_sample_data['y_pred'] = y_pred
        # y_pred_prob_class_1 = y_pred_prob[:, 1]
        # y_pred_prob_class_0 = y_pred_prob[:, 0]

        # out_of_sample_data['y_pred_proba_class_1'] = y_pred_prob_class_1
        # out_of_sample_data['y_pred_proba_class_0'] = y_pred_prob_class_0

        # Turn the predictions into entry signals
        filter_entry = (
            (out_of_sample_data['y_pred'] == 1) &
            (out_of_sample_data['returns_1_rz_7'].abs() >= 1)
        )

        entries = filter_entry.astype(int)
        exits = pd.Series(np.zeros(len(entries)), index = entries.index)
            
        tp = self.calculate_tp(
            price_data.open, 
            price_data.high,
            price_data.low, 
            price_data.close, 
            price_data.volume_traded,
            backtest_params,
            window = 7
        )

        sl = self.calculate_sl(
            price_data.open, 
            price_data.high, 
            price_data.low, 
            price_data.close,
            price_data.volume_traded,
            backtest_params,
            window = 7
        )

        # size = y_pred_prob_class_1 * trade_size_multiplier
        size = pd.Series(np.ones(len(entries)) * trade_size_multiplier, index = entries.index)

        return entries, exits, tp, sl, size