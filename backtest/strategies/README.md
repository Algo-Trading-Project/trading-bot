# Project Poseidon: Trading Strategy Creation Tutorial
##### 7/18/23  |  Author: Eamon, Louis
This document serves as a tutorial for creating an arbitrary trading strategy and then backtesting it via Project Poseidon.

## Example: Moving Average Crossover Strategy
In this example I will demonstrate how to implement a moving average crossover trading strategy. The strategy involves entering a long position when the short timeframe moving average crosses above the long timeframe moving average and exiting the long position when the short timeframe moving average crosses below the long timeframe moving average.

## Step 1: Create Strategy Class Skeleton
Every trading strategy defined in the **backtest.strategies** directory has the following general structure:

```py
class Strategy:

    indicator_factory_dict = {
        'class_name':'Strategy',
        'short_name':'strat',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['p1', 'p2', ..., 'pn'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'p1': [...],
        'p2': [...],
        ...
        'pn': [...]
    }

    def indicator_func(open, high, low, close, volume,
                       p1, p2, ..., pn):

        # Execute arbitrary code
        ...
        
        return entries, exits
```

Here are the descriptions for each component of the Strategy class skeleton:

## indicator_factory_dict
- A dictionary with the following keys:

    - class_name : Name of the trading strategy.

    - short_name : Shortened name of the trading strategy.

    - input_names : Data that gets passed as input to the function indicator_funcs for trading signal generation.  Includes OHLCV data by default.

    - param_names : Parameters of the trading strategy that will be optimized during backtesting.

    - output_names : Outputs of the trading strategy.  Always has the values ['entries', 'exits'].

## optimize_dict
- A dictionary whose keys are the values in indicator_factory_dict['param_names'] and values are a list of values for that parameter to use in optimization.

## def indicator_func
- A user-defined function that takes in the input data defined in input_names as well as values for parameters defined in param_names and executes the strategy on the input data with the given parameters.  Returns the entry and exit signals produced by the strategy.

## Step 2: Implement Moving Average Crossover Strategy
Now that we have the general structure for a Strategy class, here's what it would look like for the moving average crossover strategy:

```py
import vectorbt as vbt

class MACrossOver:
    
    indicator_factory_dict = {
        'class_name':'MACrossOver',
        'short_name':'ma_crossover',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['fast_window', 'slow_window'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'fast_window': [6, 12, 24, 48],
        'slow_window': [24 * 7, 24 * 14, 24 * 21, 24 * 30]
    }

    def indicator_func(open, high, low, close, volume,
                       fast_window, slow_window):  
        
        sma_slow = vbt.MA.run(close, window = slow_window, ewm = True)       
        sma_fast = vbt.MA.run(close, window = fast_window, ewm = True)      
        
        entries = sma_fast.ma_crossed_above(sma_slow).values
        exits = sma_fast.ma_crossed_below(sma_slow).values

        return entries, exits     
```

For this strategy, its name is MACrossOver and its short name is ma_crossover.  It has parameters fast_window & slow_window.  The indicator_func takes in OHLCV data as well as values for the fast_window and slow_window parameters and returns the entry and exit signals from applying the strategy w/ the given parameters on the OHLCV data.  At this stage, the trading strategy is fully defined and is ready to be backtested.  

## Step 3: Backtest the Strategy
Once we have defined our trading strategy in the required template, we can finally backtest it on historical price data stored in Redshift.  We do so in the **backtest.BackTester.py** file.  Shown below is the bottom of that file and the place where you'll setup your backtests:

```py
if __name__ == '__main__': 

    # Initialize a BackTester instance w/ the intended strategies to optimize
    # and an optimization metric to find the best combination of strategy
    # parameters with

    b = BackTester(
        strategies = [MACrossOver, ARIMAStrat],
        optimization_metric = 'Sharpe Ratio'
    )

    # Execute a walk-forward optimization across all strategies
    # and log the results to Redshift

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))
```

As you can see, everyting is pretty much already laid out.  The only thing you need to do when you want to backtest a strategy is to import it into the **backtest.BackTester.py** file and add it to the 'strategies' list that's passed into the BackTester class.  Running the file will initiate a walk-forward optimization of all the strategies passed into the backtester and the results are logged to Redshift for further dashboarding/analysis.