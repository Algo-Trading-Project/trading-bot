from base_strategy import BasePortfolioStrategy

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
        print('Running PortfolioMLStrategy __init__')
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

    def indicator_func(
        self, 
        universe,
        prediction_threshold_entry
    ):
        print('Running PortfolioMLStrategy indicator_func')
