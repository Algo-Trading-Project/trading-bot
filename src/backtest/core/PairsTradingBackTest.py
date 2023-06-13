import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant
from datetime import timedelta
from numba import njit, prange

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
    @njit(parallel = True)
    def __rolling_hedge_ratios_numba(y, x, window):
        n = len(y)
        hedge_ratios = np.full(n, np.nan)  # Initialize an array to store hedge ratios
        
        for i in prange(window - 1, n):
            Y = y[i - window + 1 : i + 1]
            X = x[i - window + 1 : i + 1]

            # Adding a constant to X
            X = np.vstack((X, np.ones(len(X)))).T
            
            # Least square solution
            result = np.linalg.lstsq(X, Y)[0]
            
            hedge_ratios[i] = result[0]  # Coefficient of x
            
        return hedge_ratios
    
    def __rolling_hedge_ratios(self):
        return PairsTradingBacktest.__rolling_hedge_ratios_numba(
            y = self.data[self.symbol_id_2].values, 
            x = self.data[self.symbol_id_1].values,
            window = self.hedge_ratio_window
        )
    
    @njit(parallel = True)
    def rolling_z_score(data, window):
        z_scores = np.empty(len(data))
        z_scores[:window] = np.nan  # first `window` values have no preceding window

        for i in prange(window, len(data)):
            mean_i = np.mean(data[i-window:i])
            std_i = np.std(data[i-window:i])
            z_scores[i] = (data[i] - mean_i) / std_i if std_i > 0 else 0.0  # avoid division by zero

        return z_scores
        
    def __rolling_spread_z_score(self):
        return PairsTradingBacktest.rolling_z_score(
            data = self.data['rolling_hedge_ratio'].values,
            window = self.hedge_ratio_window
        )
    
    @njit(parallel = True)
    def generate_trading_signals(z, z_prev, zl, zu):
        en = np.where(
            ((z_prev > zl) & (z < zl)),
            1,
            0
        )

        en = np.where(
            ((z_prev < zu) & (z > zu)),
            -1,
            en
        )

        ex = np.where(
            ((z_prev < zu) & (z > zu)),
            1,
            0
        )

        ex = np.where(
            ((z_prev > zl) & (z < zl)),
            -1,
            ex
        )
        
        return en, ex

    def __generate_trading_signals(self):
        rolling_spread_z_score = self.data['rolling_spread_z_score'].values
        rolling_spread_z_score_prev = self.data['rolling_spread_z_score'].shift(1).values

        return PairsTradingBacktest.generate_trading_signals(
            z = rolling_spread_z_score,
            z_prev = rolling_spread_z_score_prev,
            zl = self.z_score_lower_thresh,
            zu = self.z_score_upper_thresh
        )
    
    @njit
    def __exit_trade_numba(positions, data, dates, trades, i, is_long, commission, slippage, curr_capital):
        if is_long:
            long_symbol_index_pos = 0
            long_symbol_index_data = 1

            short_symbol_index_pos = 1
            short_symbol_index_data = 0
        else:
            long_symbol_index_pos = 1
            long_symbol_index_data = 0
            
            short_symbol_index_pos = 0
            short_symbol_index_data = 1
            
        # Set long and short position amount for current timestamp to 0
        # to simulate closing the trade
        positions[i][long_symbol_index_pos] = 0
        positions[i][short_symbol_index_pos] = 0

        # Set the exit date of the current trade to the current timestamp
        trades[-1][1] = dates[i]
        trades[-1][3] = i
        
        # Calculate pnl and pnl % from trade

        # Entry and exit dates of most current trade
        start = int(trades[-1][2])
        end = int(trades[-1][3])

        # Calculate the PnL from the long position
        start_value_long = data[start][long_symbol_index_data] * (1 + slippage) * positions[start][long_symbol_index_pos]
        end_value_long = data[end][long_symbol_index_data] * (1 - slippage) * positions[start][long_symbol_index_pos]
        long_pnl = (end_value_long - start_value_long) * (1 - commission)

        # Calculate the PnL from the short position
        start_value_short = data[start][short_symbol_index_data] * (1 - slippage) * positions[start][short_symbol_index_pos]
        end_value_short = data[end][short_symbol_index_data] * (1 + slippage) * positions[start][short_symbol_index_pos]
        short_pnl = (start_value_short - end_value_short) * (1 - commission)

        # Calculate the PnL from the entire trade 
        trade_pnl = long_pnl + short_pnl
        total_investment = start_value_long + start_value_short
        trade_pnl_pct = trade_pnl / total_investment
        
        # Set the PnL and PnL % of the current trade
        trades[-1][7] = trade_pnl_pct
        trades[-1][6] = trade_pnl

        curr_capital -= (end_value_short * (1 + commission))
        curr_capital += (end_value_long * (1 - commission))

        return positions, trades, curr_capital
    
    def __enter_trade_numba(positions, data, dates, trades, curr_capital, pct_capital_per_trade, commission, slippage, is_long, i, h):
        if is_long:
            long_symbol_index_pos = 0
            long_symbol_index_data = 1

            short_symbol_index_pos = 1
            short_symbol_index_data = 0
        else:
            long_symbol_index_pos = 1
            long_symbol_index_data = 0
            
            short_symbol_index_pos = 0
            short_symbol_index_data = 1

        p1 = data[i][short_symbol_index_data]
        p2 = data[i][long_symbol_index_data]
        c = curr_capital * pct_capital_per_trade

        if is_long:
            n1 = (c / (h * p1 + p2)) * h
            n2 = c / (h * p1 + p2)

            long_allocation = n2 * p2
            short_allocation = n1 * p1
            
            # Set long and short position amounts for current timestamp
            positions[i][long_symbol_index_pos] = n2
            positions[i][short_symbol_index_pos] = n1
            
            # Initialize a new trade and append it to the trades DataFrame
            new_trade = np.array([
                dates[i],
                np.nan,
                i,
                np.nan,
                n2,
                n1,
                np.nan,
                np.nan,
                is_long
            ])

            trades = np.vstack((trades, new_trade))
            
            # Track number of units in current long and short position
            curr_position_long_units = n2
            curr_position_short_units = n1
        else:
            n1 = c / (h * p1 + p2)
            n2 = (c / (h * p1 + p2)) * h

            long_allocation = n1 * p2
            short_allocation = n2 * p1
            
            # Set long and short position amounts for current timestamp
            positions[i][long_symbol_index_pos] = n1
            positions[i][short_symbol_index_pos] = n2
            
            # Initialize a new trade and append it to the trades DataFrame
            new_trade = np.array([
                dates[i],
                np.nan,
                i,
                np.nan,
                n2,
                n1,
                np.nan,
                np.nan,
                is_long
            ])

            trades = np.vstack((trades, new_trade))
            
            # Track number of units in current long and short position
            curr_position_long_units = n1
            curr_position_short_units = n2
        
        curr_capital -= (long_allocation * (1 + commission))
        curr_capital += (short_allocation * (1 - commission))

        return positions, trades, curr_capital, curr_position_long_units, curr_position_short_units
                                
    def __generate_positions_numba(positions, data, dates, trades, position, curr_capital, pct_capital_per_trade, commission, slippage, sl, tp):
        cp_long = 0
        cp_short = 0

        for i in range(len(data)):
            if curr_capital < 0:
                return
            
            # Retrieve entry and exit signals at current timestamp          
            entry_signal = data[i][4]
            exit_signal = data[i][5]
            
            # If not in a trade at current timestamp
            if not position:
                
                # If long or short entry signal is received and we have capital remaining
                if (entry_signal == 1 or entry_signal == -1): 
                    
                    # If we're on the last timestamp or if rolling hedge ratio is negative 
                    # then don't open a trade
                    if (i == len(data) - 1) or (data[i][2] <= 0):
                        continue
                        
                    is_long = entry_signal == 1
                    positions, trades, curr_capital, curr_position_long_units, curr_position_short_units = PairsTradingBacktest.__enter_trade_numba(
                        positions = positions,
                        data = data,
                        dates = dates,
                        trades = trades,
                        curr_capital = curr_capital,
                        pct_capital_per_trade = pct_capital_per_trade,
                        commission = commission,
                        slippage = slippage,
                        is_long = is_long,
                        i = i,
                        h = data[i][2]
                    )
                    
                    cp_long = curr_position_long_units
                    cp_short = curr_position_short_units
                    
                    position = 1

                # If no entry signal is received
                else:

                    # Set long and short position amounts for current timestamp to 0
                    # since we're not in a trade
                    positions[i][0] = 0
                    positions[i][1] = 0

            # If in a trade at current timestamp
            else:
                is_long = trades[-1][-1]
                if is_long:
                    long_symbol_index_pos = 0
                    long_symbol_index_data = 1

                    short_symbol_index_pos = 1
                    short_symbol_index_data = 0
                else:
                    long_symbol_index_pos = 1
                    long_symbol_index_data = 0
                    
                    short_symbol_index_pos = 0
                    short_symbol_index_data = 1

                # Entry and exit dates of most current trade
                start = int(trades[-1][2])
                curr_date = i

                start_value_long = data[start][long_symbol_index_data] * positions[start][long_symbol_index_pos]
                curr_value_long = data[curr_date][long_symbol_index_data] * positions[start][long_symbol_index_pos]
                long_pnl = (curr_value_long - start_value_long) * (1 - commission)

                # Calculate the PnL from the short position
                start_value_short = data[start][short_symbol_index_data] * positions[start][short_symbol_index_pos]
                curr_value_short = data[curr_date][short_symbol_index_data] * positions[start][short_symbol_index_pos]
                short_pnl = (start_value_short - curr_value_short) * (1 - commission)

                # Calculate the PnL from the entire trade 
                trade_pnl = long_pnl + short_pnl
                total_investment = start_value_long + start_value_short
                trade_pnl_pct = trade_pnl / total_investment

                if (trade_pnl_pct <= -sl) or (trade_pnl_pct >= tp):
                    
                    # Close the current trade
                    positions, trades, curr_capital = PairsTradingBacktest.__exit_trade_numba(
                        positions = positions,
                        data = data,
                        dates = dates,
                        trades = trades,
                        i = i,
                        is_long = is_long,
                        commission = commission,
                        slippage = slippage,
                        curr_capital = curr_capital
                    )

                    position = 0

                # If curr trade is long and we get a long exit or curr trade is short and we get a short exit
                if (is_long and exit_signal == 1) or (not is_long and exit_signal == -1):
                    
                    # Close the current trade
                    positions, trades, curr_capital = PairsTradingBacktest.__exit_trade_numba(
                        positions = positions,
                        data = data,
                        dates = dates,
                        trades = trades,
                        i = i,
                        is_long = is_long,
                        commission = commission,
                        slippage = slippage,
                        curr_capital = curr_capital
                    )

                    position = 0

                # Otherwise 
                else:
                    # Set long and short position amounts for current timestamp to
                    # the amounts in the current trade since we haven't exited the
                    # current trade yet

                    positions[i][long_symbol_index_pos] = cp_long
                    positions[i][short_symbol_index_pos] = cp_short

            # If the backtest reaches the final timestamp and the current trade hasn't
            # been exited yet 
            if i == len(data) - 1 and len(trades) > 0 and type(trades[-1][1]) == type(np.nan):

                # Close the current trade
                positions, trades, curr_capital = PairsTradingBacktest.__exit_trade_numba(
                    positions = positions,
                    data = data,
                    dates = dates,
                    trades = trades,
                    i = i,
                    is_long = is_long,
                    commission = commission,
                    slippage = slippage,
                    curr_capital = curr_capital
                )

                position = 0

        return trades, curr_capital

    def __generate_positions(self):
        # Long and short positions throughout backtest
        positions = pd.DataFrame(index = self.data.index, columns = [self.symbol_id_2, self.symbol_id_1]).fillna(0)
        
        trades, curr_capital = PairsTradingBacktest.__generate_positions_numba(
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