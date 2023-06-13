import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from datetime import timedelta
from .numba_funcs.numba_funcs import *

class PairsTradingBacktest:
    # BACKTESTING PARAMETERS

    # Window size to calculate the rolling z score
    # of the price spread
    z_window = 96

    # Window size to calculate the rollling hedge ratios 
    hedge_ratio_window = 168

    # Value of price spread rolling z score to initiate 
    # an entry/exit when crossed above
    z_score_upper_thresh = 2

    # Value of price spread rolling z score to initiate 
    # an entry/exit when crossed below
    z_score_lower_thresh = -2.5

    def __init__(self, 
                 price_data,
                 symbol_id_1,
                 symbol_id_2,
                 backtest_params = {
                    'initial_capital':10_000, 'pct_capital_per_trade': 0.95,
                    'comission': 0.005,
                    'sl': float('-inf'),
                    'tp':float('inf'),
                    'max_slippage':0.00
                    }
                ):
        """
        price_data - DataFrame w/ two columns of price data named 
                     symbol_id_1, symbol_id_2) and indexed by timestamp

        symbol_id_1 - Token we're shorting in the backtest

        symbol_id_2 - Token we're longing in the backtest

        backtest_params - Dictionary of hyperparameters for the current backtest
        """

        self.symbol_id_1 = symbol_id_1
        self.symbol_id_2 = symbol_id_2

        self.start_date = pd.to_datetime(price_data.index[0])
        self.end_date = pd.to_datetime(price_data.index[-1])

        self.backtest_params = backtest_params
        self.curr_capital = self.backtest_params['initial_capital']

        # Fetch data required for backtest
        self.data = price_data
        
        # Backtest initialized w/ no position
        self.position = 0

        # Number of units long in our current position
        self.curr_position_long_units = 0

        # Number of units short in our current position
        self.curr_position_short_units = 0

        # Trades executed throughout the backtest
        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', 'entry_i', 'exit_i', self.symbol_id_2, self.symbol_id_1, 'pnl', 'pnl_pct', 'is_long'])

        # PnL in dollars at each timestep
        self.pnl = None

        # Equity at each timestep
        self.equity = None
        
    ################################# HELPER METHODS #################################
        
    def __rolling_hedge_ratios(self):
        return rolling_hedge_ratios_numba(
            y = self.data[self.symbol_id_2].values, 
            x = self.data[self.symbol_id_1].values,
            window = self.hedge_ratio_window
        )
            
    def __rolling_spread_z_score(self):
        return rolling_spread_z_score_numba(
            rolling_hedge_ratios = self.data['rolling_hedge_ratios'].values,
            y = self.data[self.symbol_id_2].values,
            x = self.data[self.symbol_id_1].values,
            window = self.z_window
        )
    
    def __generate_trading_signals(self):
        rolling_spread_z_score = self.data['rolling_spread_z_score'].values
        rolling_spread_z_score_prev = self.data['rolling_spread_z_score'].shift(1).values

        return generate_trading_signals_numba(
            z = rolling_spread_z_score,
            z_prev = rolling_spread_z_score_prev,
            zl = self.z_score_lower_thresh,
            zu = self.z_score_upper_thresh
        )
        
    def __generate_positions(self):
        # Long and short positions throughout backtest
        positions = pd.DataFrame(index = self.data.index, columns = [self.symbol_id_2, self.symbol_id_1]).fillna(0)
        
        trades, curr_capital = generate_positions_numba(
            positions = positions.values.astype(float),
            data = self.data.values.astype(float),
            dates = self.data.index.values.astype(float),
            trades = self.trades.values.astype(float),
            position = self.position,
            curr_capital = self.curr_capital,
            pct_capital_per_trade = self.backtest_params['pct_capital_per_trade'],
            commission = self.backtest_params['comission'],
            slippage = self.backtest_params['max_slippage'],
            sl = self.backtest_params['sl'],
            tp = self.backtest_params['tp']
        ) 

        self.trades = pd.DataFrame(data = trades, columns = self.trades.columns)
        self.trades['entry_date'] = pd.to_datetime(self.trades['entry_date'])
        self.trades['exit_date'] = pd.to_datetime(self.trades['exit_date'])
        
        self.curr_capital = curr_capital
                
    def __generate_pnl(self):
        pnl_data = pd.DataFrame(index = self.data.index, columns = ['pnl']).fillna(0)
        max_slippage = self.backtest_params['max_slippage']
        
        for trade in self.trades.to_dict(orient = 'records'):
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            long_symbol = self.symbol_id_2 if trade['is_long'] else self.symbol_id_1
            short_symbol = self.symbol_id_1 if trade['is_long'] else self.symbol_id_2
            
            units_long = trade[long_symbol]
            units_short = trade[short_symbol]
            
            trade_period = self.data.loc[entry_date:exit_date].copy()
            
            trade_period.at[entry_date, long_symbol] = trade_period.at[entry_date, long_symbol] * (1 + max_slippage)
            trade_period.at[exit_date, long_symbol] = trade_period.at[exit_date, long_symbol] * (1 - max_slippage)
            trade_period.at[entry_date, short_symbol] = trade_period.at[entry_date, short_symbol] * (1 - max_slippage)
            trade_period.at[exit_date, short_symbol] = trade_period.at[exit_date, short_symbol] * (1 + max_slippage)
            
            trade_period['long_pnl'] = trade_period[long_symbol].diff() * units_long
            trade_period['short_pnl'] = -trade_period[short_symbol].diff() * units_short
            trade_period['total_pnl'] = trade_period['long_pnl'] + trade_period['short_pnl']

            for date in trade_period.index:
                pnl_data.at[date, 'pnl'] = trade_period.at[date, 'total_pnl']
                
        return pnl_data

    def __generate_equity_curve(self):
        return (self.backtest_params['initial_capital'] + self.pnl.cumsum()).rename({'pnl':'equity'}, axis = 1)
    
    def __calculate_performance_metrics(self):
        ########################### HELPER FUNCTIONS ####################################
        def exposure_pct(duration):
            exposure = timedelta(days = 0, hours = 0)

            for i in range(len(self.trades)):
                entry_date = pd.to_datetime(self.trades.at[i, 'entry_date'])
                exit_date = pd.to_datetime(self.trades.at[i, 'exit_date'])
                trade_duration = exit_date - entry_date
                exposure += trade_duration
            
            try:
                return round(exposure / duration * 100, 2)
            except:
                return np.nan

        def buy_and_hold_return():
            token1_start_value = self.data.at[self.data.index[0], self.symbol_id_1]
            token1_end_value = self.data.at[self.data.index[-1], self.symbol_id_1]
            buy_and_hold_return_token_1 = round((token1_end_value - token1_start_value) / token1_start_value * 100, 2)
            
            token2_start_value = self.data.at[self.data.index[0], self.symbol_id_2]
            token2_end_value = self.data.at[self.data.index[-1], self.symbol_id_2]
            buy_and_hold_return_token_2 = round((token2_end_value - token2_start_value) / token2_start_value * 100, 2)

            return max([buy_and_hold_return_token_1, buy_and_hold_return_token_2])
        
        def sharpe_ratio():
            returns = self.equity.equity.pct_change()
            mean_returns = returns.mean()
            std_returns = returns.std()

            try:
                return np.sqrt(8760) * mean_returns / std_returns 
            except:
                return np.nan
            
        def sortino_ratio():
            returns = self.equity.pct_change()
            negative_returns = returns[returns['equity'] < 0]
            
            mean_returns = returns.mean()
            std_negative_returns = negative_returns.std()
            
            try:
                return np.sqrt(8760) * mean_returns / std_negative_returns 
            except:
                return np.nan
            
        def calmar_ratio():
            num_years = len(self.data) / 8760
            cum_ret_final = (1 + self.equity.equity.pct_change()).prod().squeeze()
            annual_returns = cum_ret_final ** (1 / num_years) - 1
            
            try:
                return annual_returns / abs(max_drawdown() / 100)
            except:
                return np.nan

        def cagr_over_avg_drawdown():
            num_years = len(self.data) / 8760
            cum_ret_final = (1 + self.equity.equity.pct_change()).prod().squeeze()
            annual_returns = cum_ret_final ** (1 / num_years) - 1
            
            try:
                return annual_returns / abs(avg_drawdown() / 100)
            except:
                return np.nan

        def profit_factor():
            if len(self.trades) == 0:
                return np.nan
            
            gross_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
            gross_loss = self.trades[self.trades['pnl'] < 0]['pnl'].abs().sum()
            
            try:
                return gross_profit / gross_loss
            except:
                return np.nan
        
        def max_drawdown():
            rolling_max_equity = self.equity.cummax()
            drawdown = (self.equity / rolling_max_equity) - 1
            max_dd = drawdown.min()
            return round(max_dd * 100, 2)
        
        def avg_drawdown():
            rolling_max_equity = self.equity.cummax()
            drawdown = (self.equity / rolling_max_equity) - 1
            avg_dd = drawdown.mean()
            return round(avg_dd * 100, 2)
        
        def max_drawdown_duration():
            dates = pd.Series([pd.to_datetime(self.start_date)])
            
            diff = self.equity.cummax().diff().fillna(0)
            diff = diff[diff['equity'] != 0]

            for date in diff.index:
                date = pd.to_datetime(date)
                dates = pd.concat([dates, pd.Series(date)], ignore_index = True)

            return dates.diff().max()
        
        def avg_drawdown_duration():
            dates = pd.Series([pd.to_datetime(self.start_date)])
            
            diff = self.equity.cummax().diff().fillna(0)
            diff = diff[diff['equity'] != 0]

            for date in diff.index:
                date = pd.to_datetime(date)
                dates = pd.concat([dates, pd.Series(date)], ignore_index = True)

            return dates.diff().mean()

        def win_rate():
            if len(self.trades) == 0:
                return np.nan
            
            num_winning_trades = len(self.trades[self.trades['pnl_pct'] > 0])
            num_trades_total = len(self.trades)

            try:
                return round(num_winning_trades / num_trades_total * 100, 2)
            except:
                return np.nan
        
        def best_trade():
            if len(self.trades) == 0:
                return np.nan
            
            return round(self.trades['pnl_pct'].max() * 100, 2)
            
        def worst_trade():
            if len(self.trades) == 0:
                return np.nan
            
            return round(self.trades['pnl_pct'].min() * 100, 2)
        
        def avg_trade():
            if len(self.trades) == 0:
                return np.nan
            
            return round(self.trades['pnl_pct'].mean() * 100, 2)
        
        def max_trade_duration():
            if len(self.trades) == 0:
                return np.nan
            
            return (pd.to_datetime(self.trades['exit_date']) - pd.to_datetime(self.trades['entry_date'])).max()            
            
        def avg_trade_duration():
            if len(self.trades) == 0:
                return np.nan
            
            return (pd.to_datetime(self.trades['exit_date']) - pd.to_datetime(self.trades['entry_date'])).mean()            
        #################################### HELPER FUNCTIONS END ####################################
                
        start = self.start_date
        end = self.end_date
        duration = pd.to_datetime(end) - pd.to_datetime(start)

        return_pct = ((self.equity['equity'].iloc[-1] - self.equity['equity'].iloc[0]) / self.equity['equity'].iloc[0]) * 100
        
        metrics_dict = {
            'duration':duration, 
            'exposure_pct': exposure_pct(duration),
            'equity_final':self.equity.iloc[-1]['equity'],
            'equity_peak':self.equity['equity'].max(),
            'return_pct':return_pct,
            'buy_and_hold_return_pct':buy_and_hold_return(),
            'sharpe_ratio':sharpe_ratio(), 
            'sortino_ratio':sortino_ratio(),
            'calmar_ratio':calmar_ratio(),
            'cagr_over_avg_drawdown':cagr_over_avg_drawdown(),
            'profit_factor':profit_factor(),
            'max_dd_pct':max_drawdown(),
            'avg_dd_pct':avg_drawdown(),
            'max_dd_duration':max_drawdown_duration(), 
            'avg_dd_duration':avg_drawdown_duration(),
            'num_trades':len(self.trades), 
            'win_rate_pct':win_rate(), 
            'best_trade_pct':best_trade(),
            'worst_trade_pct':worst_trade(), 
            'avg_trade_pct':avg_trade(),
            'max_trade_duration':max_trade_duration(),
            'avg_trade_duration':avg_trade_duration()
        }

        return pd.DataFrame(metrics_dict).reset_index()[metrics_dict.keys()]
    
    ################################# HELPER METHODS #################################
    
    def visualize_results(self):
        fig, (a0, a1, a2) = plt.subplots(nrows = 3, ncols = 1, figsize = (16, 10), gridspec_kw={'height_ratios': [0.4, 0.4, 0.6]})
        fig.subplots_adjust(hspace=.5)
        
        # Plot % returns curve
        plt.subplot(3,1,1)
        
        (self.equity.rename({'equity':''}, axis = 1) / self.backtest_params['initial_capital']).plot(ax = a0, grid = True, title = 'Return [%]')
        
        a0.get_legend().remove()
        plt.xlabel('')

        # Plot PnL % for all trades taken
        plt.subplot(3,1,2)
        
        trade_pnl_pct = pd.DataFrame(index = self.data.index, columns = ['pnl_pct']).fillna(0)
        trade_marker_colors = pd.DataFrame(index = self.data.index, columns = ['color']).fillna('green')
        alphas = pd.DataFrame(index = self.data.index, columns = ['alpha']).fillna(0)
        
        for i in range(len(self.trades)):
            exit_date = self.trades.at[i, 'exit_date']
            pnl_pct = self.trades.at[i, 'pnl_pct']

            trade_pnl_pct.at[exit_date, 'pnl_pct'] = pnl_pct * 100
            alphas.at[exit_date, 'alpha'] = 1

            if pnl_pct > 0:
                trade_marker_colors.at[exit_date, 'color'] = 'green'
            else:
                trade_marker_colors.at[exit_date, 'color'] = 'red'

        trade_pnl_pct.reset_index().plot(
            ax = a1,
            kind = 'scatter', 
            x = 'time_period_end', 
            y = 'pnl_pct',
            title = 'Profit / Loss [%]',
            marker = '^', 
            c = trade_marker_colors.color.values, 
            s = 100,
            alpha = alphas.alpha.values,
            grid = True
        )
        
        plt.ylabel('')
        plt.xlabel('')

        # Plot normalized price data of the two tokens used
        plt.subplot(3,1,3)
        
        prices = self.data[[self.symbol_id_2, self.symbol_id_1]]
        prices = (prices - prices.min()) / (prices.max() - prices.min())

        title = '{} vs. {} (Normalized)'.format(self.symbol_id_2, self.symbol_id_1)

        prices.plot(ax = a2, grid = True, title = title, xlabel = '')
                
    def optimize_parameters(self, optimize_dict, performance_metric = 'Equity Final [$]', minimize = False):
        """
        Optimize strategy over every combination of parameters given in
        
        optimize_dict and sets the class strategy parameters to the values that
        
        maximize/minimize the requested performance metric.        
        """
        
        lists = []
        parameter_combinations = []

        best_comb_so_far = [self.z_window, self.hedge_ratio_window, self.z_score_upper_thresh, self.z_score_lower_thresh]
        best_metric_so_far = float('inf') if minimize else float('-inf')
        
        for key in optimize_dict.keys():
            lists.append(optimize_dict[key])

        parameter_combinations = list(itertools.product(*lists))
        
        for i in range(len(parameter_combinations)):
            
            print('{}/{}'.format(i + 1, len(parameter_combinations)), end = '\r', flush = True)            

            self.z_window = parameter_combinations[i][0]
            self.hedge_ratio_window = parameter_combinations[i][1]
            self.z_score_upper_thresh = parameter_combinations[i][2]
            self.z_score_lower_thresh = parameter_combinations[i][3]
            
            self.backtest()

            backtest_result_metric = float(self.performance_metrics[performance_metric][0])

            if not minimize and backtest_result_metric > best_metric_so_far:
                best_comb_so_far = parameter_combinations[i]
                best_metric_so_far = backtest_result_metric

            elif minimize and backtest_result_metric < best_metric_so_far:
                best_comb_so_far = parameter_combinations[i]
                best_metric_so_far = backtest_result_metric

        self.z_window = best_comb_so_far[0]
        self.hedge_ratio_window = best_comb_so_far[1]
        self.z_score_upper_thresh = best_comb_so_far[2]
        self.z_score_lower_thresh = best_comb_so_far[3]
        
        self.backtest()

        optimal_params = {'z_window': self.z_window, 'hedge_ratio_window': self.hedge_ratio_window,
                          'z_score_upper_thresh': self.z_score_upper_thresh, 'z_score_lower_thresh': self.z_score_lower_thresh}
        
        return optimal_params
    
    def backtest(self):        
        # Calculate rolling hedge ratios at each timestep
        self.data['rolling_hedge_ratio'] = self.__rolling_hedge_ratios()
                
        # Calculate rolling spread z score at each timestep
        self.data['rolling_spread_z_score'] = self.__rolling_spread_z_score()

        # Calculate entry and exit signals at each timestep
        entry_signals, exit_signals = self.__generate_trading_signals()

        self.data['entry_signals'] = entry_signals
        self.data['exit_signals'] = exit_signals
        
        # Reset trades DataFrame to make sure it's empty
        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', 'entry_i', 'exit_i', self.symbol_id_2, self.symbol_id_1, 'pnl', 'pnl_pct', 'is_long'])
        
        # Calculate long and short positions at each timestep
        self.__generate_positions()

        # Calculate PnL in dollars at each timestep
        self.pnl = self.__generate_pnl()

        # Calculate equity at each timestep
        self.equity = self.__generate_equity_curve()

        # Calculate performance metrics for backtest
        self.performance_metrics = self.__calculate_performance_metrics()  

        # Reset position variable back to default state for subsequent backtests
        self.position = 0

        # Reset curr_capital variable back to initial value for subsequent backtests
        self.curr_capital = self.backtest_params['initial_capital']