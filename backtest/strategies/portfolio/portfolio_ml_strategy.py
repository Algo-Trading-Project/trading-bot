from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy
from utils.db_utils import QUERY

class PortfolioMLStrategy(BasePortfolioStrategy):
    
    indicator_factory_dict = {
        'class_name':'PortfolioMLStrategy',
        'short_name':'portfolio_ml_strategy',
        'input_names':['universe'],
        'param_names':['prediction_threshold_entry'],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    optimize_dict = {
        # Minimum predicted probability for entering a trade
        'prediction_threshold_entry': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0,
        'sl_stop': 'std',
        'sl_trail': True,
        'tp_stop': 'std',
        'size': 0.05,
        'size_type': 2, # e.g. 2 if 'size' is 'Fixed Percent' and 0 otherwise,
        'cash_sharing': True
    }

    def __init__(self):
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

        # Columns we need to drop before training the model
        triple_barrier_label_cols = [
            col for col in ml_features if 'triple_barrier_label_h' in col
        ]

        trade_returns_cols = [
            col for col in ml_features if 'trade_returns' in col
        ]

        non_numeric_cols = [
            'asset_id_base', 'asset_id_quote', 'exchange_id', 'Unnamed: 0',
        ]

        forward_returns_cols = [
            'open', 'high', 'low', 'close', 'start_date_triple_barrier_label_h7', 
            'end_date_triple_barrier_label_h7', 'avg_uniqueness', 'time_period_end'
        ]

        cols_to_drop = (
            triple_barrier_label_cols + 
            trade_returns_cols + 
            non_numeric_cols +
            forward_returns_cols
        )

        self.ml_features = QUERY(
            """
            SELECT *
            FROM market_data.ml_features
            """
        )
        self.ml_features = self.ml_features.drop(cols_to_drop, axis=1)
        self.ml_features = self.ml_features.set_index('time_period_end')
    
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

    def indicator_func(
        self, 
        universe,
        prediction_threshold_entry
    ):
        min_year = universe.index.dt.year.min()
        max_year = universe.index.dt.year.max()
        
        self.ml_features['y_pred_prob'] = 0.0

        for year in range(min_year, max_year + 1):
            for month in range(1, 13):
                # Split year into 4 equal parts
                month_map = {'jan_to_march': [1, 2, 3], 'april_to_june': [4, 5, 6], 'july_to_sept': [7, 8, 9], 'oct_to_dec': [10, 11, 12]}
                for quarter, months in month_map.items():
                    max_month = max(months)
                    universe_filter = (
                        (universe.index.year == year) &
                        (universe.index.month.isin(months))
                    )
                    min_date = universe.loc[universe_filter].index.min()

                    # Get model
                    model = self.__get_model(min_date)

                    # Get features for this quarter
                    feature_filter = (
                        (self.ml_features.index.year == year) &
                        (self.ml_features.index.month.isin(months))
                    )
                    X = self.ml_features.loc[feature_filter]

                    # Predictions on this quarter
                    y_pred_prob = model.predict_proba(X.drop(['y_pred_prob'], axis=1))[:, 1]
                    self.ml_features.loc[feature_filter, 'y_pred_prob'] = y_pred_prob

        # Pivot model's predictions to the universe for ranking signals
        model_probs = self.ml_features.reset_index()[['time_period_end', 'symbol_id', 'y_pred_prob']].pivot_table(index='time_period_end', columns='symbol_id', values='y_pred_prob').sort_index()
        model_probs = model_probs.loc[universe.index]

        print(model_probs)

                    