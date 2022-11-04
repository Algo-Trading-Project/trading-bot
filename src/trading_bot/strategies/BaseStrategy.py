import pandas as pd

class BaseStrategy:

    def __init__(self, strat_time_frame_minutes, sl):
        cols = [
            'time_period_start', 'time_period_end', 'price_open',
            'price_high', 'price_low', 'price_close',
            'volume_traded'
        ]

        self.data = pd.DataFrame([], columns = cols)
        self.strat_time_frame_minutes = strat_time_frame_minutes
        self.sl = sl

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
        self.tick_count += 1

        if (len(self.data) == 0) or ((len(self.data) -  1) % self.strat_time_frame_minutes == 0 and len(self.data) != 1):
            self.data.append([tick_data])
            print(tick_data)

        return self.__generate_trading_signal()

        
