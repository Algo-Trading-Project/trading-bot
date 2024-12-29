#################################
#             MISC              #
#################################
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

import pandas as pd
import numpy as np
import json
import duckdb
import vectorbt as vbt

from utils.db_utils import QUERY

class PortfolioBackTester:
    
    def __init__(
        self,
        strategies,
        resample_period
    ):
        self.strategies = strategies
        self.resample_period = resample_period
        
        price_data = QUERY(
        """
        SELECT 
            time_period_end,
            asset_id_base || '_' || asset_id_quote || '_' || exchange_id as symbol_id,
            close
        FROM market_data.ml_dataset
        """
        )
        price_data['time_period_end'] = pd.to_datetime(price_data['time_period_end'])
        price_data = price_data.set_index('time_period_end')

        # Pivot the data to provide the asset universe as columns
        self.prices = price_data.pivot_table(index = 'time_period_end', columns = 'symbol_id', values = 'close')
        self.prices = self.prices.sort_index()

        self.stragegy_dict = {}

        for strategy in self.strategies:
            strat = (vbt.IndicatorFactory(**strategy.indicator_factory_dict)
                                 .from_apply_func(strategy.indicator_func, 
                                                  keep_pd = True,
                                                  to_2d = False))

            self.stragegy_dict[strategy.indicator_factory_dict['short_name']] = (strat, strategy)

    def execute(self):
        for strat_name in self.stragegy_dict.keys():
            strat_vbt, strat = self.stragegy_dict[strat_name]

            if len(strat.optimize_dict) == 0:
                entries, exits, tp, sl, size = strat_vbt.run(
                    universe = self.prices,
                    param_product = False
                )
            else:
                entries, exits, tp, sl, size = strat_vbt.run(
                    universe = self.prices,
                    param_product = True,
                    **strat.optimize_dict
                )

            portfolio = vbt.Portfolio.from_signals(
                close = self.prices,
                entries = entries,
                exits = exits,
                size = size,
                sl_stop = sl,
                tp_stop = tp,
                **strat.backtest_params
            )

            returns = portfolio.total_return()
            print(f'{strat_name} returns: {returns}')
