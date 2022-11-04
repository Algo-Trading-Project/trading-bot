from BaseStrategy import BaseStrategy



class ARIMAStrategy(BaseStrategy):

    def __init__(self, strat_time_frame_minutes, sl, symbol_id):
        super().__init__(strat_time_frame_minutes = strat_time_frame_minutes, sl = sl)

    def process_tick(self, tick_data):
        trading_signal = super().process_tick(tick_data)

        if trading_signal == 'Sell':
            return trading_signal
        else:
            # Implement Stragegy Logic and return a trading signal (Buy, Sell, or None)
            ...