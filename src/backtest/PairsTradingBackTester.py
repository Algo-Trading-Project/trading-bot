%%time

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import time

from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant
from backtesting.lib import crossover, cross

class PairsTradingBacktester:

    z_window = 24
    hedge_ratio_window = 24
    z_score_upper_thresh = 2
    z_score_lower_thresh = -2

    def __init__(self, symbol_id_1, symbol_id_2, start_date, end_date, initial_capital = 10_000, pct_capital_per_trade = 0.1):
        """
        symbol_id_1 - The token we're shorting in the backtest

        symbol_id_2 - The token we're longing in the backtest

        start_date - The start date of the backtest

        end_date - The end date of the backtest
        """

        self.symbol_id_1 = symbol_id_1
        self.symbol_id_2 = symbol_id_2

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.pct_capital_per_trade = pct_capital_per_trade

        self.data = self.get_data()
        self.data['rolling_hedge_ratio'] = self.__rolling_hedge_ratios()
        self.data['rolling_spread_z_score'] = self.__rolling_spread_z_score()
        self.data['entry_signals'] = self.__generate_entry_signals()
        self.data['exit_signals'] = self.__generate_exit_signals()
        
        self.position = False
        self.curr_position_long_units = 0
        self.curr_position_short_units = 0

        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', self.symbol_id_2, self.symbol_id_1])
        self.positions = self.__generate_positions()
        self.equity = self.__generate_equity_curve()

        # self.returns = None

    ################################# HELPER METHODS #################################

    def __convert_to_usd(self, df, eth):
        if 'ETH_USD' in df.columns[0]:
            return df
        
        m = df.merge(eth, on = 'time_period_end', how = 'inner')
        m[df.columns[0]] = m[df.columns[0]] * m['ETH_USD_COINBASE']
        return m[df.columns[0]].to_frame()
    
    def __rolling_hedge_ratios(self):
        return RollingOLS(
            endog = self.data[self.symbol_id_2].to_frame(),
            exog = add_constant(self.data[self.symbol_id_1].to_frame()),
            window = PairsTradingBacktester.hedge_ratio_window
        ).fit().params[self.symbol_id_1]
    
    def __rolling_spread_z_score(self):
        rolling_spread = self.data[self.symbol_id_2] - self.data['rolling_hedge_ratio'] * self.data[self.symbol_id_1]
        return (rolling_spread - rolling_spread.rolling(window = PairsTradingBacktester.hedge_ratio_window).mean()) / rolling_spread.rolling(window = PairsTradingBacktester.hedge_ratio_window).std()
    
    def __generate_entry_signals(self):
        entry_signals = []
        for i in range(len(self.data)):
            if i == 0:
                entry_signals.append(0)
                continue

            prev_time_period_z_score = self.data.loc[self.data.index[i - 1], 'rolling_spread_z_score']
            curr_time_period_z_score = self.data.loc[self.data.index[i], 'rolling_spread_z_score']

            if prev_time_period_z_score >= PairsTradingBacktester.z_score_lower_thresh and curr_time_period_z_score < PairsTradingBacktester.z_score_lower_thresh:
                entry_signals.append(1)
            else:
                entry_signals.append(0)

        return entry_signals

    def __generate_exit_signals(self):
        exit_signals = []
        for i in range(len(self.data)):
            if i == 0:
                exit_signals.append(0)
                continue

            prev_time_period_z_score = self.data.loc[self.data.index[i - 1], 'rolling_spread_z_score']
            curr_time_period_z_score = self.data.loc[self.data.index[i], 'rolling_spread_z_score']

            if prev_time_period_z_score <= PairsTradingBacktester.z_score_upper_thresh and curr_time_period_z_score > PairsTradingBacktester.z_score_upper_thresh:
                exit_signals.append(1)
            else:
                exit_signals.append(0)

        return exit_signals

    def __generate_positions(self):
        positions = pd.DataFrame(index = self.data.index, columns = [self.symbol_id_2, self.symbol_id_1])

        for i in range(len(self.data['entry_signals'])):
            entry_signal = self.data.loc[self.data.index[i], 'entry_signals']
            exit_signal = self.data.loc[self.data.index[i], 'exit_signals']

            if not self.position:
                if entry_signal: 
                    rolling_hedge_ratio = self.data.loc[self.data.index[i], 'rolling_hedge_ratio'] 
                    
                    long_allocation = self.initial_capital * self.pct_capital_per_trade / 2
                    units_long = long_allocation / self.data.loc[self.data.index[i], self.symbol_id_2]
 
                    short_allocation = self.initial_capital * self.pct_capital_per_trade / 2
                    units_short = short_allocation / self.data.loc[self.data.index[i], self.symbol_id_1]

                    if long_allocation + short_allocation * rolling_hedge_ratio > self.initial_capital:
                        positions.loc[self.data.index[i], self.symbol_id_2] = 0
                        positions.loc[self.data.index[i], self.symbol_id_1] = 0
                        continue


                    positions.loc[self.data.index[i], self.symbol_id_2] = units_long
                    positions.loc[self.data.index[i], self.symbol_id_1] = -units_short * rolling_hedge_ratio

                    self.initial_capital -= (long_allocation + short_allocation * rolling_hedge_ratio)
                    
                    self.trades = self.trades.append({
                        'entry_date':self.data.index[i],
                        'exit_date':np.nan,
                         self.symbol_id_2:units_long,
                         self.symbol_id_1:units_short
                    }, ignore_index = True)
                    
                    self.curr_position_long_units = units_long
                    self.curr_position_short_units = units_short
                    self.position = 1
                else:
                    positions.loc[self.data.index[i], self.symbol_id_2] = 0
                    positions.loc[self.data.index[i], self.symbol_id_1] = 0

            else:
                if exit_signal:
                    long_value = positions.loc[self.data.index[i - 1], self.symbol_id_2] * self.data.loc[self.data.index[i], self.symbol_id_2]
                    short_value = positions.loc[self.data.index[i - 1], self.symbol_id_1] * self.data.loc[self.data.index[i], self.symbol_id_1]
                    
                    positions.loc[self.data.index[i], self.symbol_id_2] = 0
                    positions.loc[self.data.index[i], self.symbol_id_1] = 0

                    self.initial_capital += (long_value + short_value)
                    self.trades.loc[len(self.trades) - 1,'exit_date'] = self.data.index[i]
                    self.position = 0
                else:
                    positions.loc[self.data.index[i], self.symbol_id_2] = self.curr_position_long_units
                    positions.loc[self.data.index[i], self.symbol_id_1] = self.curr_position_short_units

        return positions
    
    def __generate_equity_curve(self):
        pass
    
    ################################# HELPER METHODS #################################

    def buy(self):
        pass

    def sell(self):
        pass

    def get_data(self):
        cols = [
            'price_open', 'price_high', 'price_low', 
            'price_close', 'asset_id_base', 'asset_id_quote', 'exchange_id'
        ]

        df = pd.read_csv(
            '/Users/louisspencer/Desktop/Trading-Bot/src/backtest/backtest_data/price_data_1hr.csv', 
            index_col='time_period_end'
        )[cols]

        df['symbol_id'] = df.asset_id_base + '_' + df.asset_id_quote + '_' + df.exchange_id

        pivot = df.pivot_table(
            index = 'time_period_end',
            columns = 'symbol_id', 
            values = ['price_open', 'price_high', 'price_low', 'price_close']
        )

        eth = pivot['price_close']['ETH_USD_COINBASE']
        to_return = pd.DataFrame(index = pivot.index, columns = ['Open', 'High', 'Low', 'Close'])
        
        to_return['Open'] = pivot['price_open'][self.symbol_id_2] / pivot['price_open'][self.symbol_id_1]
        to_return['High'] = pivot['price_high'][self.symbol_id_2] / pivot['price_high'][self.symbol_id_1]
        to_return['Low'] = pivot['price_low'][self.symbol_id_2] / pivot['price_low'][self.symbol_id_1]
        to_return['Close'] = pivot['price_close'][self.symbol_id_2] / pivot['price_close'][self.symbol_id_1]
        
        to_return[self.symbol_id_1] = self.__convert_to_usd(pivot['price_close'][self.symbol_id_1].to_frame(), eth)
        to_return[self.symbol_id_2] = self.__convert_to_usd(pivot['price_close'][self.symbol_id_2].to_frame(), eth)

        to_return_index = np.array(to_return.index.to_list())
        to_return = to_return[(to_return_index >= self.start_date) & (to_return_index <= self.end_date)].fillna(method = 'pad')
        to_return.index = pd.to_datetime(to_return.index)

        return to_return

    def calculate_performance_metrics(self):
        # Your implementation here
        pass

    def visualize_results(self):
        # Your implementation here
        pass

    def optimize_parameters(self):
        # Your implementation here
        pass

    def backtest(self):
        # Your implementation here
        pass

# Example usage
p = PairsTradingBacktester(
    symbol_id_1 = 'ETH_USD_COINBASE',
    symbol_id_2 = 'ALEPH_ETH_GATEIO',
    start_date = '2022-08-01',
    end_date = '2022-09-01'
)