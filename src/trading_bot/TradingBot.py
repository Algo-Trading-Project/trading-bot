import websocket
import json
import time
import datetime
import redshift_connector
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from Trade import Trade
from strategies.ARIMAStrategy import ARIMAStrategy

class TradingBot:

    def __init__(self, strategy, symbol_id, backtest = True):
        self.strategy = strategy
        self.symbol_id = symbol_id   
        self.time = 0

    def __log_trade(self, trade_type, tick_data):
        print(trade_type)
        # with redshift_connector.connect(
        #     host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
        #     database = 'administrator',
        #     user = 'administrator',
        #     password = 'Free2play2'
        # ) as conn:
        #     with conn.cursor() as cursor:
        #         # Query to fetch open price data for an ETH pair & exchange of interest
        #         # within a specified date range
        #         exchange, _, base, quote = self.symbol_id.split('_')
        #         query = """
        #         INSERT INTO trading_bot.eth.trading_log VALUES ({}, {}, {}, {}, {}, {}, {})
        #         """.format(base, quote, exchange, trade_type, tick_data['open_price'], 1, tick_data['time_period_start'])

        #         # Execute query on Redshift and return result
        #         cursor.execute(query)

    def buy(self, tick_data):
        # Buy-related logic
        if self.strategy.add_position(tick_data):
            self.__log_trade('BUY', tick_data)

    def sell(self, tick_data):
        # Sell-related logic
        if self.strategy.remove_position():
            self.__log_trade('SELL', tick_data)

    def execute(self):
        def on_open(ws):
            print('connection opened')
            print()
            
            hello_message = {
                'type': 'hello',
                'apikey': '34FA476E-CCCB-4AC7-A033-B472113BBD22',
                'heartbeat': False,
                'subscribe_data_type': ['ohlcv'],
                'subscribe_filter_period_id': ['1MIN'],
                'subscribe_filter_symbol_id': [self.symbol_id + '$']
            }

            ws.send(json.dumps(hello_message))

            self.time = time.time()

        def on_close(ws, close_status_code, close_message):
            print('connection closed')
            print('close status code: {}'.format(close_status_code))
            print(close_message)

            self.strategy.preds = np.insert(self.strategy.preds, 0, self.strategy.preds[0])
            self.strategy.actual = np.append(self.strategy.actual, self.strategy.actual[-1])

            fig = plt.figure(figsize=(10,8))
            r = list(range(len(self.strategy.preds)))
            
            plt.scatter(r, self.strategy.preds, c = 'r')
            plt.scatter(r, self.strategy.actual, c = 'b')
            plt.plot(r, self.strategy.preds, c = 'r')
            plt.plot(r, self.strategy.actual, c = 'b')
            plt.show()

        def on_message(ws, message):
            curr_time = time.time()
            time_elapsed_minutes = (curr_time - self.time) / 60

            print('mins: {}, len: {}'.format(time_elapsed_minutes, len(self.strategy.data)))

            if time_elapsed_minutes >= 1 or len(self.strategy.data) == 0:
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

                if trading_signal == 'Buy' and self.strategy.position == None:
                    # log buy in Redshift and create position to store in Strategy class
                    self.buy()
                    pass

                elif trading_signal == 'Sell' and self.strategy.position != None:
                    # log sell in Redshift and delete position in Strategy class
                    self.sell()
                    pass

                elif trading_signal == 'None':
                    pass

        def on_error(ws, error):
            print('error occurred!')
            print(error)
            print()

        endpoint = 'ws://ws.coinapi.io/v1/'

        w = websocket.WebSocketApp(
            url = endpoint,
            on_open = on_open,
            on_close = on_close,
            on_error = on_error,
            on_message = on_message
        )

        w.run_forever()

strat = ARIMAStrategy(strat_time_frame_minutes=1, sl = 0.1, symbol_id = 'COINBASE_SPOT_SUSHI_USD')
bot = TradingBot(strategy = strat, symbol_id = 'COINBASE_SPOT_SUSHI_USD')
bot.execute()
 