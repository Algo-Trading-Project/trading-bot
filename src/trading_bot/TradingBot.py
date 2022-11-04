import websocket
import json
import time

from Position import Position
from strategies.ARIMAStrategy import ARIMAStrategy

class TradingBot:

    def __init__(self, strategy, symbol_id, test = True):
        self.strategy = strategy
        self.symbol_id = symbol_id   
        self.time = 0
        self.test = test

    def execute(self):
        def on_open(ws):
            print('connection opened')
            print()
            
            hello_message = {
                'type': 'hello',
                'apikey': '3D7A02BB-EC62-40F9-8C5C-8625AB51D5ED',
                'heartbeat': False,
                'subscribe_data_type': ['ohlcv'],
                'subscribe_filter_period_id': ['1MIN'],
                'subscribe_filter_symbol_id': [self.symbol_id]
            }
            
            ws.send(json.dumps(hello_message))

            self.time = time.time()

        def on_close(ws, close_status_code, close_message):
            print('connection closed')
            print('close status code: {}'.format(close_status_code))
            print(close_message)
            print()

        def on_message(ws, message):
            curr_time = time.time()
            time_elapsed_minutes = curr_time - self.time

            if time_elapsed_minutes >= 1:
                self.time = curr_time

                message = json.loads(message)
                tick_data = {
                    'time_period_start': message['time_period_start'],
                    'time_period_end': message['time_period_end'],
                    'price_open': message['price_open'],
                    'price_high': message['price_high'],
                    'price_low': message['price_low'],
                    'price_close': message['price_close'],
                    'volume_traded': message['volume_traded']
                }

                trading_signal = self.strategy.process_tick(tick_data)

                if trading_signal == 'Buy':
                    # log buy in Redshift and create position to store in Strategy class
                    pass
                elif trading_signal == 'Sell':
                    # log sell in Redshift and delete position in Strategy class
                    
                    pass
                else:
                    return

        def on_error(ws, error):
            print('error occurred!')
            print(error)
            print()

        endpoint = 'ws://ws-sandbox.coinapi.io/v1/'

        w = websocket.WebSocketApp(
            url = endpoint,
            on_open = on_open,
            on_close = on_close,
            on_error = on_error,
            on_message = on_message
        )

        w.run_forever()

strat = ARIMAStrategy(strat_time_frame_minutes = 60, sl = 0.1)

bot = TradingBot(strategy = strat, symbol_id = 'COINBASE_SPOT_ETH_USD')
bot.execute()