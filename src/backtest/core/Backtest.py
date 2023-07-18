import vectorbt as vbt
import numpy as np

class Backtest:

    def __init__(self, 
                 strategy,
                 price_data, 
                 optimization_metric,
                 backtest_params):
        
        """
        strategy            - Strategy class in src/backtest/strategies to backtested
        price_data          - Dataframe of OHLCV data indexed by timestamp
        optimization_metric - Performance metric to optimize backtest on
        backtest_params     - Miscellaneous parameters to configure the backtest
        """
        
        self.price_data = price_data
        self.optimization_metric = optimization_metric
        self.strategy = strategy
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
            'Deflated Sharpe Ratio':'Max',
            'Calmar Ratio':'Max',
            'Omega Ratio':'Max',
            'Sortino Ratio':'Max'
        }

        self.custom_indicator = (vbt.IndicatorFactory(**strategy.indicator_factory_dict)
                                 .from_apply_func(strategy.indicator_func, 
                                                  to_2d = False,
                                                  **strategy.default_dict))

    def generate_signals(self, params, param_product = False): 
        res = self.custom_indicator.run(
            self.price_data.price_open,
            self.price_data.price_high,
            self.price_data.price_low,
            self.price_data.price_close,
            self.price_data.volume_traded,
            param_product = param_product,
            **params
        )

        entries = res.entries
        exits = res.exits

        return entries, exits
    
    def backtest(self, params):
        entries, exits = self.generate_signals(params = params)

        portfolio = vbt.Portfolio.from_signals(
            close = self.price_data.price_close,
            entries = entries,
            exits = exits,
            freq = 'h',
            **self.backtest_params
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

    def optimize(self):
        entries, exits = self.generate_signals(
            params = self.strategy.optimize_dict,
            param_product = True
        )

        portfolio = vbt.Portfolio.from_signals(
            close = self.price_data.price_close,
            entries = entries,
            exits = exits,
            freq = 'h',
            **self.backtest_params
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
            
            for param_name, best_value in zip(self.strategy.optimize_dict.keys(), maximizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb
        else:
            minimizing_index = backtest_result_metrics.idxmin()

            for param_name, best_value in zip(self.strategy.optimize_dict.keys(), minimizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb