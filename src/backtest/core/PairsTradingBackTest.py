import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant
from datetime import timedelta

class PairsTradingBacktest:
    # BACKTESTING PARAMETERS

    # Window size to calculate the rolling z score
    # of the price spread
    z_window = 96

    # Window size to calculate the rollling hedge ratios 
    hedge_ratio_window = 168

    # Value of price spread rolling z score to initiate 
    # an exit when crossed above
    z_score_upper_thresh = 2

    # Value of price spread rolling z score to initiate 
    # an entry when crossed below
    z_score_lower_thresh = -2.5

    def __init__(self, 
                 price_data,
                 symbol_id_1,
                 symbol_id_2,
                 backtest_params = {
                    'initial_capital':10_000, 'pct_capital_per_trade': 1,
                    'comission': 0.01,
                    'sl': 0.05,
                    'tp':0.1,
                    'max_slippage':0.005
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
        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', self.symbol_id_2, self.symbol_id_1, 'pnl', 'pnl_pct', 'is_long'])

        # PnL in dollars at each timestep
        self.pnl = None

        # Equity at each timestep
        self.equity = None

        # Performance metrics for backtest
        self.performance_metrics = None
        
    ################################# HELPER METHODS #################################

    def __rolling_hedge_ratios(self):
        return RollingOLS(
            endog = self.data[self.symbol_id_2],
            exog = add_constant(self.data[self.symbol_id_1]),
            window = self.hedge_ratio_window
        ).fit().params[self.symbol_id_1]
    
    def __rolling_spread(self):
        return self.data[self.symbol_id_2].values - self.data['rolling_hedge_ratio'].values * self.data[self.symbol_id_1].values
    
    def __rolling_spread_z_score(self):
        def rolling_window(a, window):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        
        rolling_spread = self.data['rolling_spread'].values
        rolling_spread_window = rolling_window(rolling_spread, self.z_window)

        rolling_mean = np.mean(rolling_spread_window, axis = -1)
        rolling_std = np.std(rolling_spread_window, axis = -1)
        
        len_diff = abs(len(rolling_spread) - len(rolling_mean))
        for _ in np.arange(len_diff):
            rolling_mean = np.insert(rolling_mean, 0, np.nan)
            rolling_std = np.insert(rolling_std, 0, np.nan)

        z = (rolling_spread - rolling_mean) / rolling_std

        return z
    
    def __generate_trading_signals(self):
        rolling_spread_z_score = self.data['rolling_spread_z_score'].values
        rolling_spread_z_score_prev = self.data['rolling_spread_z_score'].shift(1).values
        
        # Long Entry
        entry_signals = np.where(
            (rolling_spread_z_score_prev > self.z_score_lower_thresh) & (rolling_spread_z_score < self.z_score_lower_thresh),
            1,
            0
        )

        # Short Entry
        entry_signals = np.where(
            (rolling_spread_z_score_prev < self.z_score_upper_thresh) & (rolling_spread_z_score > self.z_score_upper_thresh),
            -1,
            entry_signals
        )

        # Long Exit
        exit_signals = np.where(
            (rolling_spread_z_score_prev < self.z_score_upper_thresh) & (rolling_spread_z_score > self.z_score_upper_thresh),
            1,
            0
        )

        # Short Exit
        exit_signals = np.where(
            (rolling_spread_z_score_prev > self.z_score_lower_thresh) & (rolling_spread_z_score < self.z_score_lower_thresh),
            -1,
            exit_signals
        )

        return entry_signals, exit_signals
    
    def __exit_trade(self, positions, i, is_long):
        long_symbol = self.symbol_id_2 if is_long else self.symbol_id_1
        short_symbol = self.symbol_id_1 if is_long else self.symbol_id_2

        # Set long and short position amount for current timestamp to 0
        # to simulate closing the trade
        positions.at[self.data.index[i], long_symbol] = 0
        positions.at[self.data.index[i], short_symbol] = 0
        
        # Set the exit date of the current trade to the current timestamp
        self.trades.at[len(self.trades) - 1,'exit_date'] = self.data.index[i]

        # Indicate we're no longer in a trade
        self.position = 0
        
        # Calculate pnl and pnl % from trade

        # Entry and exit dates of most current trade
        start = self.trades.at[len(self.trades) - 1,'entry_date']
        end = self.trades.at[len(self.trades) - 1,'exit_date']

        # Calculate the PnL from the long position
        slippage = self.backtest_params['max_slippage']

        start_value_long = self.data.at[start, long_symbol] * (1 + slippage) * positions.at[start, long_symbol]
        end_value_long = self.data.at[end, long_symbol] * (1 - slippage) * positions.at[start, long_symbol]
        long_pnl = (end_value_long - start_value_long) * (1 - self.backtest_params['comission'])

        # Calculate the PnL from the short position
        start_value_short = self.data.at[start, short_symbol] * (1 - slippage) * positions.at[start, short_symbol]
        end_value_short = self.data.at[end, short_symbol] * (1 + slippage) * positions.at[start, short_symbol]
        short_pnl = (start_value_short - end_value_short) * (1 - self.backtest_params['comission'])

        # Calculate the PnL from the entire trade 
        trade_pnl = long_pnl + short_pnl
        total_investment = start_value_long + start_value_short
        trade_pnl_pct = trade_pnl / total_investment
        
        # Set the PnL and PnL % of the current trade
        self.trades.at[len(self.trades) - 1, 'pnl_pct'] = trade_pnl_pct
        self.trades.at[len(self.trades) - 1, 'pnl'] = trade_pnl
        
        return end_value_long + end_value_short
    
    def __enter_trade(self, row, positions, i, is_long = True):
        pct_capital_per_trade = self.backtest_params['pct_capital_per_trade']
        comission = self.backtest_params['comission']
        max_slippage = self.backtest_params['max_slippage']
        
        long_symbol = self.symbol_id_2 if is_long else self.symbol_id_1
        short_symbol = self.symbol_id_1 if is_long else self.symbol_id_2

        # Amount of dollars to allocate to the long position
        long_allocation = (1 - comission) * self.curr_capital * pct_capital_per_trade / 2
        
        # Number of units of symbol_id_2 to long
        units_long = long_allocation / (row[long_symbol] * (1 + max_slippage))
        
        # Amount of dollars to allocate to the short position
        short_allocation = (1 - comission) * self.curr_capital * pct_capital_per_trade / 2

        # Number of units of symbol_id_1 to short
        units_short = short_allocation / (row[short_symbol] * (1 - max_slippage))

        self.curr_capital -= (long_allocation + short_allocation)

        # Set long and short position amounts for current timestamp
        positions.at[self.data.index[i], long_symbol] = units_long
        positions.at[self.data.index[i], short_symbol] = units_short
                
        # Initialize a new trade and append it to the trades DataFrame
        new_trade = {
            'entry_date':self.data.index[i],
            'exit_date':np.nan,
            long_symbol:units_long,
            short_symbol:units_short,
            'pnl':np.nan,
            'pnl_pct':np.nan,
            'is_long':is_long
        }

        new_trade = pd.DataFrame(new_trade, index = [0])
        self.trades = pd.concat([self.trades, new_trade], ignore_index = True)
        
        # Track number of units in current long and short position
        self.curr_position_long_units = units_long
        self.curr_position_short_units = units_short
        
        # Indicate we're in a trade now
        self.position = 1

    def __generate_positions(self):
        # Long and short positions throughout backtest
        positions = pd.DataFrame(index = self.data.index, columns = [self.symbol_id_2, self.symbol_id_1])

        # Iterate through each timestamp of dataset
        i = 0
        for row in self.data.to_dict(orient = 'records'):  
            # Retrieve entry and exit signals at current timestamp          
            entry_signal = row['entry_signals']
            exit_signal = row['exit_signals']

            # If not in a trade at current timestamp
            if not self.position:

                # If long or short entry signal is received
                if entry_signal == 1 or entry_signal == -1: 
        
                    # If we're on the last timestamp or if rolling hedge ratio is negative 
                    # then don't open a trade
                    if (i == len(self.data) - 1) or (row['rolling_hedge_ratio'] <= 0):
                        i += 1
                        continue
                        
                    is_long = entry_signal == 1
                    self.__enter_trade(
                        row = row, 
                        positions = positions,
                        i = i,
                        is_long = is_long
                    )

                # If no entry signal is received
                else:

                    # Set long and short position amounts for current timestamp to 0
                    # since we're not in a trade
                    positions.at[self.data.index[i], self.symbol_id_2] = 0
                    positions.at[self.data.index[i], self.symbol_id_1] = 0

            # If in a trade at current timestamp
            else:
                is_long = self.trades.at[len(self.trades) - 1, 'is_long']

                # If curr trade is long and we get a long exit or curr trade is short and we get a short exit
                if (is_long and exit_signal == 1) or (not is_long and exit_signal == -1):
                    
                    # Close the current trade
                    end_trade_amount = self.__exit_trade(
                        positions = positions,
                        i = i,
                        is_long = is_long
                    )

                    # Add money from closing position back into our available capital
                    self.curr_capital += end_trade_amount
                                
                # Otherwise 
                else:
                    # Set long and short position amounts for current timestamp to
                    # the amounts in the current trade since we haven't exited the
                    # current trade yet
                    is_long = self.trades.at[len(self.trades) - 1, 'is_long']
                    long_symbol = self.symbol_id_2 if is_long else self.symbol_id_1
                    short_symbol = self.symbol_id_1 if is_long else self.symbol_id_2

                    positions.at[self.data.index[i], long_symbol] = self.curr_position_long_units
                    positions.at[self.data.index[i], short_symbol] = self.curr_position_short_units
            
            # If the backtest reaches the final timestamp and the current trade hasn't
            # been exited yet 
            if i == len(self.data) - 1 and len(self.trades) > 0 and type(self.trades.at[len(self.trades) - 1, 'exit_date']) == type(np.nan):

                # Reverse the trade and remove it from the trades DataFrame
                self.trades = self.trades.drop([len(self.trades) - 1])
                self.position = 0

            i += 1
                
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
            
            return round(exposure / duration * 100, 2)

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
            
            return gross_profit / gross_loss
        
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

            return round(num_winning_trades / num_trades_total * 100, 2)
        
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
            'equity_final':self.equity.dropna().iloc[-1]['equity'],
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

        best_comb_so_far = None
        best_metric_so_far = 1000000000000 if minimize else -1000000000000
        
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
        
        return optimal_params, self.performance_metrics.to_dict(orient = 'records')[0]
        
    def backtest(self):        
        # Calculate rolling hedge ratios at each timestep
        self.data['rolling_hedge_ratio'] = self.__rolling_hedge_ratios()
        
        # Calculate rolling spread at each timestep
        self.data['rolling_spread'] = self.__rolling_spread()
        
        # Calculate rolling spread z score at each timestep
        self.data['rolling_spread_z_score'] = self.__rolling_spread_z_score()

        # Calculate entry and exit signals at each timestep
        entry_signals, exit_signals = self.__generate_trading_signals()

        self.data['entry_signals'] = entry_signals
        self.data['exit_signals'] = exit_signals
        
        # Reset trades DataFrame to make sure it's empty
        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', self.symbol_id_2, self.symbol_id_1, 'pnl', 'pnl_pct', 'is_long'])
        
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

        return self.performance_metrics.to_dict(orient = 'records')[0]
    