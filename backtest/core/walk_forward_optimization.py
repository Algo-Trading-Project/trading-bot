import vectorbt as vbt
import numpy as np

class WalkForwardOptimization:

    def __init__(self, 
                 strategy,
                 backtest_data,
                 is_start_i,
                 is_end_i,
                 oos_start_i,
                 oos_end_i,
                 optimization_metric,
                 backtest_params):
        
        """
        Performs a walk-forward optimization on an arbitrary strategy over an arbitrary
        token.  Logs the resulting trades and equity curve to Redshift for further dashboarding/analysis.


        strategy            - Strategy class in backtest/strategies to backtest
        backtest_data       - Dataframe of OHLCV data indexed by timestamp
        is_start_i          - Starting index of
        optimization_metric - Performance metric to optimize backtest on
        backtest_params     - Miscellaneous parameters to configure the backtest
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
            'Deflated Sharpe Ratio':'Max',
            'Calmar Ratio':'Max',
            'Omega Ratio':'Max',
            'Sortino Ratio':'Max'
        }

        self.custom_indicator = (vbt.IndicatorFactory(**strategy.indicator_factory_dict)
                                 .from_apply_func(strategy.indicator_func, 
                                                  to_2d = False,
                                                  **strategy.default_dict))
        
    def generate_signals(self, params, optimize, param_product = False):
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
    
    def walk_forward(self, params):
        entries, exits = self.generate_signals(params = params, optimize = False)
        
        backtest_data = self.backtest_data.iloc[self.oos_start_i:self.oos_end_i]
        
        entries = entries.dropna()
        entries = entries[entries.index.isin(backtest_data.index)]

        exits = exits.dropna()
        exits = exits[exits.index.isin(backtest_data.index)]

        portfolio = vbt.Portfolio.from_signals(
            close = self.backtest_data.iloc[self.oos_start_i:self.oos_end_i].price_close,
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
            optimize = True,
            param_product = True
        )

        portfolio = vbt.Portfolio.from_signals(
            close = self.backtest_data.iloc[self.is_start_i:self.is_end_i].price_close,
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

            return best_param_comb, portfolio
        else:
            minimizing_index = backtest_result_metrics.idxmin()

            for param_name, best_value in zip(self.strategy.optimize_dict.keys(), minimizing_index):
                best_param_comb[param_name] = best_value

            return best_param_comb, portfolio