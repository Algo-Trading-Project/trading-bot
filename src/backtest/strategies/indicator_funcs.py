import vectorbt as vbt

# This file contains all custom indicator functions or strategies
# that will be backtested in the file BackTester.py

def indicator_func(close, fast_window, slow_window):
    fast_ma = vbt.MA.run(close, window = fast_window)
    slow_ma = vbt.MA.run(close, window = slow_window)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    return entries, exits            

