import vectorbt as vbt

class Strategy:

    def __init__(self, 
                 price_data, 
                 indicator_factory_params, 
                 indicator_func,
                 indicator_func_defaults,
                 optimize_dict = {}, 
                 optimization_metric = 'Total Return'):
        
        """
        price_data -               Dataframe of closing price data indexed by timestamp
        
        optimize_dict -            Dictionary of all hyperparameter combinations to use in backtest

        optimization_metric -      Performance metric to optimize backtests on

        indicator_factory_params - Dictionary of parameters needed to create a custom
                                   indicator in vectorbt

        indicator_func -           User-defined function that takes in an arbitrary amount
                                   of input data including close price data as well as
                                   an arbitrary amount of hyperparameters and returns
                                   entry and exit signals for the price data
        
        indicator_func_defaults - Dictionary of default values for the hyperparameters
                                  used in indicator_func
        """
        
        self.price_data = price_data
        self.optimize_dict = optimize_dict
        self.optimization_metric = optimization_metric

        self.metric_map = {
            'Total Return':'total_return',
            'Max Drawdown':'drawdowns.max_drawdown',
            'Max Drawdown Duration':'drawdowns.max_duration',
            'Win Rate':'trades.win_rate',
            'Profit Factor':'trades.profit_factor',
            'Expectancy':'trades.expectancy',
            'Sharpe Ratio':'returns_acc.sharpe_ratio',
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

        self.custom_indicator = (vbt.IndicatorFactory(**indicator_factory_params)
                                 .from_apply_func(indicator_func, **indicator_func_defaults))

    def generate_signals(self):        
        res = self.custom_indicator.run(
            self.price_data, 
            param_product = True,
            **self.optimize_dict
        )

        entries = res.entries
        exits = res.exits

        return entries, exits
    
    def backtest(self, params):
        res = self.custom_indicator.run(
            self.price_data,
            **params
        )

        entries = res.entries
        exits = res.exits

        portfolio = vbt.Portfolio.from_signals(
            close = self.price_data,
            entries = entries,
            exits = exits
        )

        backtest_results = portfolio.stats().to_dict()

        for del_col in ['Start', 'End', 'Period', 'Total Fees Paid', 
                        'Total Closed Trades', 'Total Open Trades', 'Open Trade PnL']:
            try:
                del backtest_results[del_col]
            except:
                pass

        return backtest_results

    def optimize(self):
        entries, exits = self.generate_signals()

        portfolio = vbt.Portfolio.from_signals(
            close = self.price_data,
            entries = entries,
            exits = exits,
            freq = 'h'
        )

        metric_attribute_path = self.metric_map.get(self.optimization_metric)
        split_path = metric_attribute_path.split('.')

        if len(split_path) == 1:
            backtest_result_metrics = getattr(portfolio, split_path[0])()
        else:
            backtest_result_metrics = getattr(getattr(portfolio, split_path[0]), split_path[1])()

        best_param_comb = {}

        if self.metric_min_max_map.get(self.optimization_metric) == 'Max':
            maximizing_index = backtest_result_metrics.idxmax()
            
            for param_name, best_value in zip(self.optimize_dict.keys(), maximizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb, portfolio.loc[maximizing_index].stats().to_dict()
        else:
            minimizing_index = backtest_result_metrics.idxmin()

            for param_name, best_value in zip(self.optimize_dict.keys(), minimizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb, portfolio.loc[minimizing_index].stats().to_dict()