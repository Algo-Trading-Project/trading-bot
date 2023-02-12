import pandas as pd

class BaseStrategy:

    def __init__(self, strat_time_frame_minutes, sl, symbol_id):
        cols = [
            'time_period_start', 'time_period_end', 'price_open',
            'price_high', 'price_low', 'price_close',
            'volume_traded'
        ]

        self.data = pd.DataFrame([], columns = cols)
        self.strat_time_frame_minutes = strat_time_frame_minutes
        self.sl = sl
        self.symbol_id = symbol_id

        self.position = None
        self.curr_price = None
        self.tick_count = 0

    def __generate_trading_signal(self):
        if self.position and self.curr_price <= self.position.buy_price - (self.position.buy_price * self.sl):
            return 'Sell'
        else:
            return None

    def process_tick(self, tick_data):
        self.curr_price = tick_data['price_close']

        strat_time_frame_flag = False

        if self.tick_count % self.strat_time_frame_minutes == 0:
            self.data = pd.concat(self.data, pd.DataFrame(tick_data))
            strat_time_frame_flag = True

        trading_signal = self.__generate_trading_signal()

        self.tick_count += 1

        if trading_signal == 'Sell':
            return trading_signal
        
        elif trading_signal == None and strat_time_frame_flag:
            return 'Generate'

        elif trading_signal == None and not strat_time_frame_flag:
            return None

        
