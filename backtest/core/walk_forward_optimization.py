import vectorbt as vbt
import pandas as pd
import numpy as np
import itertools

class WalkForwardOptimization:

    def __init__(self, 
                 strategy,
                 backtest_data: pd.DataFrame,
                 is_start_i: int,
                 is_end_i: int,
                 oos_start_i: int,
                 oos_end_i: int,
                 optimization_metric: str,
                 backtest_params: dict
                 ):
        
        """
        Performs a walk-forward optimization on an arbitrary strategy over an arbitrary token.  
        Logs the resulting trades and equity curve to Redshift for further dashboarding/analysis.

        Parameters
        ----------
        strategy : Strategy
            Strategy class in backtest/strategies to backtest.

        backtest_data : pd.DataFrame
            Dataframe of OHLCV data indexed by timestamp.

        is_start_i : int
            Starting index of in-sample optimization period.

        is_end_i : int
            End index of in-sample optimization period.

        oos_start_i : int
            Starting index of out-of-sample period.

        oos_end_i : int
            End index of out-of-sample period.

        optimization_metric : str
            Performance metric to optimize backtests on.

        backtest_params : dict     
            Dictionary of parameters to configure the backtest.
        """
        
        self.strategy = strategy
        self.backtest_data = backtest_data
        self.is_start_i = is_start_i
        self.is_end_i = is_end_i
        self.oos_start_i = oos_start_i
        self.oos_end_i = oos_end_i
        self.optimization_metric = optimization_metric
        self.backtest_params = backtest_params

        self.metric_map = {
            'Total Return':'total_return',
            'Max Drawdown':'drawdowns.max_drawdown',
            'Max Drawdown Duration':'drawdowns.max_duration',
            'Win Rate':'trades.win_rate',
            'Profit Factor':'trades.profit_factor',
            'Expectancy':'trades.expectancy',
            'Sharpe Ratio':'returns_acc.sharpe_ratio',
            'Deflated Sharpe Ratio':'returns_acc.deflated_sharpe_ratio',
            'Calmar Ratio':'returns_acc.calmar_ratio',
            'Omega Ratio':'returns_acc.omega_ratio',
            'Sortino Ratio':'returns_acc.sortino_ratio'
        }

        self.metric_min_max_map = {
            'Total Return':'Max',
            'Max Drawdown':'Min',
            'Max Drawdown Duration':'Min',
            'Win Rate':'Max',
            'Profit Factor':'Max',
            'Expectancy':'Max',
            'Sharpe Ratio':'Max',
            'Calmar Ratio':'Max',
            'Omega Ratio':'Max',
            'Sortino Ratio':'Max'
        }

        self.custom_indicator = (vbt.IndicatorFactory(**strategy.indicator_factory_dict)
                                 .from_apply_func(strategy.indicator_func, 
                                                  to_2d = False))
        
    def __generate_signals(self, params: dict, optimize: bool, param_product: bool = False) -> (pd.DataFrame, pd.DataFrame):
        """
        Generates and returns entry/exit signals by applying the strategy to the provided OHLCV data
        with a specific combination of parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the specific parameter combination to use to generate entry/exit signals.
            If params contains lists for values instead of ints/floats, then apply the strategy on every
            combination of parameters and return them all

        optimize : bool
            Indicates whether or not we are in an in-sample optimization period or an out-of-sample period

        param_product : bool
            Indicates whether or not we are applying the strategy to all combinations of the input parameters

        Returns
        -------
        pd.DataFrame
            DataFrame containing entry signals from applying the strategy on the input data

        pd.DataFrame
            DataFrame containing exit signals from applying the strategy on the input data

        """
        if optimize:
            backtest_window = self.backtest_data.iloc[self.is_start_i:self.is_end_i]
        else:
            backtest_window = self.backtest_data.iloc[self.is_start_i:self.oos_end_i]

        res = self.custom_indicator.run(
            backtest_window.price_open,
            backtest_window.price_high,
            backtest_window.price_low,
            backtest_window.price_close,
            backtest_window.volume_traded,
            param_product = param_product,
            **params
        )

        entries = res.entries
        exits = res.exits

        return entries, exits
    
    def walk_forward(self, params: dict, hyperparams: dict) -> (pd.DataFrame, pd.Series):
        """
        Performs an out-of-sample backtest w/ the given strategy.  Returns the trades and equity curve of
        the backtest.

        Parameters
        ----------
        params : dict
            Dictionary of optimal strategy parameters to use in the out-of-sample backtest.

        hyperparams : dict
            Dictionary of optimal backtest hyperparameters to use in the out-of-sample backtest.

        Returns
        -------
        pd.DataFrame
            DataFrame of trades made during the out-of-sample backtest.

        pd.Series
            Equity curve of out-of-sample backtest.

        """
        entries, exits = self.__generate_signals(params = params, optimize = False)
        
        backtest_data = self.backtest_data.iloc[self.oos_start_i:self.oos_end_i]
        
        entries = entries.dropna()
        entries = entries[entries.index.isin(backtest_data.index)]

        exits = exits.dropna()
        exits = exits[exits.index.isin(backtest_data.index)]

        backtest_params_copy = self.backtest_params.copy()
        
        if not hyperparams == {}:
            backtest_params_copy['sl_stop'] = hyperparams['sl_stop']
            backtest_params_copy['tp_stop'] = hyperparams['tp_stop']
            backtest_params_copy['size'] = hyperparams['size']

        portfolio = vbt.Portfolio.from_signals(
            close = self.backtest_data.iloc[self.oos_start_i:self.oos_end_i].price_close,
            entries = entries,
            exits = exits,
            freq = 'h',
            **backtest_params_copy
        )

        equity_curve = (1 + portfolio.returns()).cumprod() * self.backtest_params['init_cash']
        trades = portfolio.trades.records_readable
        trades = trades[['Entry Timestamp', 'Exit Timestamp', 'PnL', 'Return', 'Direction']]
        
        rename_dict = {
            'Entry Timestamp':'entry_date', 'Exit Timestamp':'exit_date', 
            'PnL':'pnl', 'Return':'pnl_pct', 'Direction':'is_long'
        }

        trades = trades.rename(rename_dict, axis = 1)
        trades['is_long'] = trades['is_long'] == 'Long'

        equity_curve = equity_curve.to_frame().rename({0:'equity'}, axis = 1)

        return trades, equity_curve

    def optimize(self) -> (dict, vbt.Portfolio):
        """
        Optimizes strategy performance on the in-sample data over all parameter combinations.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Strategy parameter combination which optimizes the selected performance metric over
            the in-sample data.

        vbt.Portfolio
            The portfolio of the backtest that optimizes the selected performance metric over the
            in-sample data.
            
        """
        entries, exits = self.__generate_signals(
            params = self.strategy.optimize_dict,
            optimize = True,
            param_product = True
        )

        sl_stop_params = self.backtest_params.get('sl_stop')
        tp_stop_params = self.backtest_params.get('tp_stop')
        size_params = self.backtest_params.get('size')

        metric_attribute_path = self.metric_map.get(self.optimization_metric)
        split_path = metric_attribute_path.split('.')
        
        backtest_result_metrics_list = []
        best_backtest_params = {}

        if type(sl_stop_params) == list and type(tp_stop_params) == list and type(size_params) == list:
            combs = list(itertools.product(sl_stop_params, tp_stop_params, size_params))

            for comb in combs:
                sl_stop = comb[0]
                tp_stop = comb[1]
                size = comb[2]

                backtest_params_copy = self.backtest_params.copy()
                del backtest_params_copy['tp_stop']
                del backtest_params_copy['sl_stop']
                del backtest_params_copy['size']

                portfolio = vbt.Portfolio.from_signals(
                    close = self.backtest_data.iloc[self.is_start_i:self.is_end_i].price_close,
                    entries = entries,
                    exits = exits,
                    freq = 'h',
                    sl_stop = sl_stop,
                    tp_stop = tp_stop,
                    size = size,
                    **backtest_params_copy
                )

                if len(split_path) == 1:
                    backtest_result_metrics = getattr(portfolio, split_path[0])()
                else:
                    backtest_result_metrics = getattr(getattr(portfolio, split_path[0]), split_path[1])()

                backtest_result_metrics.replace([np.inf, -np.inf], np.nan, inplace = True)
                backtest_result_metrics.dropna(inplace = True)

                backtest_result_metrics = pd.concat([backtest_result_metrics], keys = [sl_stop], names = ['sl_stop'])
                backtest_result_metrics = pd.concat([backtest_result_metrics], keys = [tp_stop], names = ['tp_stop'])
                backtest_result_metrics = pd.concat([backtest_result_metrics], keys = [size], names = ['size'])
                
                backtest_result_metrics_list.append(backtest_result_metrics)

            backtest_result_metrics = pd.concat(backtest_result_metrics_list)
        else:
            portfolio = vbt.Portfolio.from_signals(
                close = self.backtest_data.iloc[self.is_start_i:self.is_end_i].price_close,
                entries = entries,
                exits = exits,
                freq = 'h',
                **self.backtest_params
            )

            if len(split_path) == 1:
                backtest_result_metrics = getattr(portfolio, split_path[0])()
            else:
                backtest_result_metrics = getattr(getattr(portfolio, split_path[0]), split_path[1])()

            backtest_result_metrics.replace([np.inf, -np.inf], np.nan, inplace = True)
            backtest_result_metrics.dropna(inplace = True)
        
        best_param_comb = {}

        if self.metric_min_max_map.get(self.optimization_metric) == 'Max':
            if type(sl_stop_params) == list and type(tp_stop_params) == list and type(size_params) == list:
                maximizing_index = backtest_result_metrics.idxmax()
                
                best_backtest_params['size'] = maximizing_index[0]
                best_backtest_params['tp_stop'] = maximizing_index[1]
                best_backtest_params['sl_stop'] = maximizing_index[2]

                maximizing_index = maximizing_index[3:]
            else:
                maximizing_index = backtest_result_metrics.idxmax()
            
            for param_name, best_value in zip(self.strategy.optimize_dict.keys(), maximizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb, portfolio, best_backtest_params
        else:
            if type(sl_stop_params) == list and type(tp_stop_params) == list and type(size_params) == list:
                minimizing_index = backtest_result_metrics.idxmin()

                best_backtest_params['size'] = maximizing_index[0]
                best_backtest_params['tp_stop'] = maximizing_index[1]
                best_backtest_params['sl_stop'] = maximizing_index[2]
                
                minimizing_index = minimizing_index[3:]
            else:
                minimizing_index = backtest_result_metrics.idxmin()

            for param_name, best_value in zip(self.strategy.optimize_dict.keys(), minimizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb, portfolio, best_backtest_params