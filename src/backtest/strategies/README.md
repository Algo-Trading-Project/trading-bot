# Project Poseidon: Trading Strategy Creation Tutorial
##### 7/18/23  |  Author: Eamon
This document serves as a tutorial for creating a trading strategy and then backtesting it via Project Poseidon.
The following packages are needed:

```{bash}
pip install vectorbt 
pip install ta 
pip install redshift_connector 
pip install numpy
pip install statsmodels
```

Importing dependencies here:

```{py}
import numpy as np 
import vectorbt as vbt 
```

# EX1: ta.py Indicators Only <Donchian_KAMA_Aroon.py>
In this example I will be using vectorbt to make the following trading strategy using indicators provided by ta.py:

### 1. Donchian Channel Indicator
https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html?highlight=donchian#ta.volatility.DonchianChannel
https://www.investopedia.com/terms/d/donchianchannels.asp

### 2. KAMA Indicator 
https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html?highlight=donchian#ta.momentum.KAMAIndicator
https://www.tradingview.com/ideas/kama/

### 3. Aroon Indicator 
https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html?highlight=donchian#ta.trend.AroonIndicator
https://www.investopedia.com/terms/a/aroon.asp

## Defining Metadata

#### Signals 

The logic for the automated trading will be simple for this example. Let's also take the time here to assign each a name that will identify it in the code going forward.  We will aggregate individual buy/sell signals and make a decision based on their combined input. 

##### BUY SIGNALS:
1. c_under_donc_l = price <= donc_low 
2. c_cross_under_donc_l = price crosses under donc_low 
3. c_under_KAMA = price < KAMA 
4. c_cross_under_KAMA = price crosses under KAMA 
5. ar_up_above_down = ar_up > ar_down 

##### SELL SIGNALS:
1. c_above_donc_h = price >= donc_high 
2. c_cross_above_donc_h = price crosses over donc_high
3. c_above_KAMA = price > KAMA 
4. c_cross_above_KAMA = price crosses above KAMA
5. ar_down_above_up = ar_up < ar_down 

Noteworthy is the distinction between a comparison and a cross/crossover event. The former is a constant comparison (the signal activates as long as the condition is true) whereas the cross/crossover signals will only be active during the candle for which the event is detected, thus placing more importance on a momentary decision to act. 

#### Signal Weightings 

Each signal will have an assigned weighting (integer weighting to simplify the already massive brute-force gridsearching effort this will require), and the combined signal will only activate when the weightings add up to more than a certain proportion of the maximum possible weighting value. We will call this "tolerance proportion" tol_p (ranging from 0 to 1). This may sound a bit confusing, so let's go back to our example and see how this applies:
Both buy and sell signals have a total of 5 constituents, with a total set of 5 weights defined by weights_list = [W1, W2,...,W5].
Therefore, the total possible weighting value at any moment is sum(weights_list). In this case this equals 6 (the aroon signal weighting = 2 for entries and exits).

##### Our activation threshhold for buying or selling will be set at (tol_p) * sum(weights_list). 

For example, if tol_p is 3/6 and each signal has weighting 1, then we are saying that we will either buy or sell if 3/6 of our signals tell us to do so at any point in time. This provides us a tolerance which we can adjust in gridsearching, which alongside the weightings, may prove useful in seeing which parts of a strategy are actually beneficial/secretly hurting performance. Now that we have a conceptual background on what we want to do, let's put a few things down to establish metadata. 

#### Price Feed Inputs 

Project Poseidon has price feeds for Open, Close, Low, High and Volume data--these price feeds serve as the 'inputs' for which we run our backtests on. Let's take a look at our indicators and see which of these we actually need:

- Donchian requires close, high, low
- KAMA requires close 
- Aroon requires close

All together, this means that we only need the close, high and low price feeds. However, it is best practice to specify all of the available time-series anyways as they will eventually get concatenated together on a dataframe and column mismatch would be an issue. This check is mostly to make sure that the price feeds avaialble are sufficient for the indicators we want to generate, which in this case is true, so we proceed.

```{py}
'input_names':['open',
               'high', 
               'low', 
               'close',
               'volume']
```

#### Parameter Metadata
Now, let's look at the parameters associated with each indicator and name them for future reference, alongside determining a range of possible values through which to gridsearch. 

##### I. Donchian Channel 
donc_window => n period. 

##### II. KAMA 
KAMA_window (int) => n period. 
KAMA_pow1 (int) => number of periods for the fastest EMA constant. 
KAMA_pow2 (int) => number of periods for the slowest EMA constant. 

##### III. Aroon
ar_window (int) => n period.

##### IV. Signal Weightings 

All will be defaulted to 1 (aside from the aroon-based signal, since only 1 signal represents aroon while two signals represent both the donchian and the KAMA). For now to make things simple, however I will still write things in a more expandable manner later on so adjusting weights will be simple. The variable for the weighting of a signal will be formatted as W_<signal_name_or_denotation>. E.g. W_ar_down_above_up is the weight value for the ar_down_above_up signal (boolean) 

To recap, our signal names are:

##### ENTRY SIGNALS 
c_under_donc_l 
c_cross_under_donc_l 
c_under_KAMA 
c_cross_under_KAMA 
ar_up_above_down

##### EXIT SIGNALS 
c_above_donc_h
c_cross_above_donc_h
c_above_KAMA
c_cross_above_KAMA
ar_down_above_up

Let's write these down in code so we don't forget, in a format that will be useful to copy-paste later:

```{py}
entry_weights_list = [W_c_under_donc_l,
                      W_c_cross_under_donc_l,
                      W_c_under_KAMA,
                      W_c_cross_under_KAMA,
                      W_ar_up_above_down]
exit_weights_list = [W_c_above_donc_h,
                     W_c_cross_above_donc_h,
                     W_c_above_KAMA,
                     W_c_cross_above_KAMA,
                     W_ar_down_above_up]
```
##### V. Decision-Making 
tol_p_entry & tol_p_exit (float) => proportion of maximum weighting value which defines the tolerance threshhold for the trading bot's actions in entering and exiting trades. 

Now, let's sort these in terms of which parameters have ranges we need to look through, and which don't. This will be done in an odd format, but only because it will allow for some copy-paste later. First, list the names of all our parameters. 

The order here is very specific so be careful! I personally like to do it in the order that I listed them out, just to be extra safe. This will be an entry in a dictionary later on, hence the syntax. 

```{py}
'param_names':['donc_window',
                'KAMA_window',
                'KAMA_pow1',
                'KAMA_pow2',
                'ar_window',
                'W_c_under_donc_l',
                'W_c_cross_under_donc_l',
                'W_c_under_KAMA',
                'W_c_cross_under_KAMA',
                'W_ar_up_above_down',
                'W_c_above_donc_h',
                'W_c_cross_above_donc_h',
                'W_c_above_KAMA',
                'W_c_cross_above_KAMA',
                'W_ar_down_above_up',
                'tol_p_entry',
                'tol_p_exit'],
```

Let's put all that we have so far into one dictionary. Why we call it indicator_factory_dict will make sense later, for now just know that this is the dictionary which effectively establishes the main metadata about the strategy 'entity' itself. 

```{py}
indicator_factory_dict = {
    'class_name':'Donchian_KAMA_Aroon',
    'short_name':'Donchian_KAMA_Aroon',
    'input_names':['open', 'high', 'low', 'close', 'volume'],
    'param_names':['donc_window',
                    'KAMA_window',
                    'KAMA_pow1',
                    'KAMA_pow2',
                    'ar_window',
                    'W_c_under_donc_l',
                    'W_c_cross_under_donc_l',
                    'W_c_under_KAMA',
                    'W_c_cross_under_KAMA',
                    'W_ar_up_above_down',
                    'W_c_above_donc_h',
                    'W_c_cross_above_donc_h',
                    'W_c_above_KAMA',
                    'W_c_cross_above_KAMA',
                    'W_ar_down_above_up',
                    'tol_p_entry',
                    'tol_p_exit'],
    'output_names':['entries', 'exits']
}
```

Please do not alter input_names or output_names! Also, it does not matter whether there is or is not any distinction between class_name and short_name. For the parameters we want to apply a gridsearch to, let's now define their ranges in a separate dictionary. 

Note that the order has the be the same as in 'param_names'. Note that the ranges for the weights are set to only be at 1, but these can be adjusted to any desired range if you desire to get more complicated with weightings. 
- Exercise caution when doing this, however, as having too many terms being backtested can cause a lot of noise and make it harder to draw meaningful conclusions about which parts of the strategy actually work. 
- I would only recommend adjusting the weighting if prior testing/a serious hunch/some theory that you believe holds true exists.

```{py}
optimize_dict = {
                 'donc_window':[7,10,21],
                 'KAMA_window':[7,10,21],
                 'KAMA_pow1':[2,5],
                 'KAMA_pow2':[20,30],
                 'ar_window':[17,25,33],
                 'W_c_under_donc_l':[1],
                 'W_c_cross_under_donc_l':[1],
                 'W_c_under_KAMA':[1],
                 'W_c_cross_under_KAMA':[1],
                 'W_ar_up_above_down':[2],
                 'W_c_above_donc_h':[1],
                 'W_c_cross_above_donc_h':[1],
                 'W_c_above_KAMA':[1],
                 'W_c_cross_above_KAMA':[1],
                 'W_ar_down_above_up':[2],
                 'tol_p_entry':[0.16, 0.32, 0.49, 0.65, 0.82, 0.99],
                 'tol_p_exit':[0.16, 0.32, 0.49, 0.65, 0.82, 0.99]
}
```

It's a good measure to count the number of individual terms within all the gridsearch ranges we input. Ideally, we have no more than 30-50 elements across all lists. Here, it seems like we have exactly 37, which is all right. 

Let's do this again, this time for defining the default values:

```{py}
default_dict = {
                'donc_window':10,
                'KAMA_window':10,
                'KAMA_pow1':2,
                'KAMA_pow2':30,
                'ar_window':25,
                'W_c_under_donc_l':1,
                'W_c_cross_under_donc_l':1,
                'W_c_under_KAMA':1,
                'W_c_cross_under_KAMA':1,
                'W_ar_up_above_down':2,
                'W_c_above_donc_h':1,
                'W_c_cross_above_donc_h':1,
                'W_c_above_KAMA':1,
                'W_c_cross_above_KAMA':1,
                'W_ar_down_above_up':2,
                'tol_p_entry':0.49,
                'tol_p_exit':0.49
}
```

To recap, at this point we should have fully explained how to structure the indicator_factory_dict, optimize_dict and default_dict. These definitions will be included in the strategy code, so don't lose them! 

For now though, we'll focus on actually generating indicator values using vectorbt, and then go over how to get signals and weighted decisioning from this. Then, we'll put every piece of code together at the end, and I'll show you what the final complete product should look like. 

## Defining Indicators 

vectorbt also has a way to automatically integrate ta indicators so that they are recognizable when our software (which is vectorbt based) conducts backtesting. Links for reference are below:

https://vectorbt.dev/api/indicators/factory/#vectorbt.indicators.factory.IndicatorFactory
https://vectorbt.dev/api/indicators/factory/#vectorbt.indicators.factory.IndicatorFactory.from_pandas_ta
https://vectorbt.dev/api/indicators/factory/#vectorbt.indicators.factory.IndicatorFactory.from_ta

It is accessible by calling the following:

```
vbt_indicator = vbt.IndicatorFactory.from_ta([Indicator_Name_string])
vbt_indicator.run(close, high, low, **kwargs)
```

Let's write our indicators in terms of this syntax. Note that for indicators the Donchian and Aroon, two (or 3+) separate indicators are calculated to form the 'umbrella indicator' so we must take that into account. 

One thing to note is that ta.py does offer methods which each only calculate one of these indicator feeds, and also methods which will output all at once. Be careful which one you choose, and make sure that you route your outputs correctly!

Definitions first:

```{py}
# Define Indicators
ar_up_i = vbt.IndicatorFactory.from_ta('AroonIndicator.ar_up')
ar_down_i = vbt.IndicatorFactory.from_ta('AroonIndicator.ar_down')
donc_l_i = vbt.IndicatorFactory.from_ta('DonchianChannel.donc_channel_lband')
donc_h_i = vbt.IndicatorFactory.from_ta('DonchianChannel.donc_channel_hband')
kama_i = vbt.IndicatorFactory.from_ta('KAMAIndicator')
```

Now use the .run() method. 

NOTE: MAKE SURE fillna = True for each, this is to make sure that all output lists are of the same length (some will activate at different times because their window lengths are subject to vary)

```{py}
# Run Indicators
ar_up = ar_up_i.run(close,window=ar_window, fillna = True)
ar_down = ar_down_i.run(close,window=ar_window, fillna = True)
donc_l_i = donc_l_i.run(high,low,close,window=donc_window, fillna = True)
donc_h_i = donc_h_i.run(high,low,close,window=donc_window, fillna = True)
kama = kama_i.run(close,window=KAMA_window,pow1=KAMA_pow1,pow2=KAMA_pow2, fillna = True)
```

## Defining Signals & Trading Logic 
- For constant-comparison signals, list comprehension works perfectly fine to generate the signals. 
- For signals generated only during crossover events, vbt offers helpful functions which can be used to easily compare two time-series and output a signal result.

```{py}
# Define BUY Signals 
c_under_donc_l = [0 if (d_l is None) else (W_c_under_donc_l if (cl < d_l) else 0) for cl, d_l in zip(close, donc_l)]
c_cross_under_donc_l = close.close_crossed_below(donc_l)
c_under_KAMA = [0 if (k is None) else (W_c_under_KAMA if (cl < k) else 0) for cl, k in zip(close, KAMA)]
c_cross_under_KAMA = close.close_crossed_below(kama)
ar_up_above_down = [0 if ((a_d is None) or (a_u is None)) else (W_ar_up_above_down if a_u > a_d else 0) for a_d, a_u in zip(ar_down, ar_up)]

# Define SELL Signals 
c_above_donc_h = [0 if (d_l is None) else (W_c_under_donc_l if (cl > d_h) else 0) for cl, d_l in zip(close, donc_h)]
c_cross_above_donc_h = close.close_crossed_above(donc_h)
c_above_KAMA = [0 if (k is None) else (W_c_above_KAMA if (cl > k) else 0) for cl, k in zip(close, KAMA)]
c_cross_above_KAMA = close.close_crossed_above(kama)
ar_down_above_up = [0 if ((a_d is None) or (a_u is None)) else (W_ar_down_above_up if (a_d > a_u) else 0) for a_d, a_u in zip(ar_down, ar_up)]
```

Then, we need to calculate a few more things in order to do our proportioned weighting--namely, the actual value of the entry/exit threshhold weighting is then found by multiplying the sum of all entry/exit weights, respectively, by tol_p_entry/tol_p_exit.

```{py}
# Entry Weighted Decisioning Threshhold Calculation
entry_weights_list = [W_c_under_donc_l,
                      W_c_cross_under_donc_l,
                      W_c_under_KAMA,
                      W_c_cross_under_KAMA,
                      W_ar_up_above_down]
total_entry_weights = sum(entry_weights_list)
entry_threshhold_weight = tol_p_entry * total_entry_weights 

# Exit Weighted Decisioning Threshhold Calculation
exit_weights_list = [W_c_above_donc_h,
                     W_c_cross_above_donc_h,
                     W_c_above_KAMA,
                     W_c_cross_above_KAMA,
                     W_ar_down_above_up]
total_exit_weights = sum(exit_weights_list)
exit_threshhold_weight = tol_p_exit * total_exit_weights
```

##### Our output is entries, exits where each is a list of binaries. 
0 -> Do not enter/exit at the current moment
1 -> Signal to enter/exit at the current momnent

```{py}
# BUY/SELL Signal Calculation 
entry_weight_sums = [sum(x) for x in zip(c_under_donc_l, c_cross_under_donc_l, c_under_KAMA, c_cross_under_KAMA, ar_down_above_up)]
exit_weight_sums = [sum(x) for x in zip(c_above_donc_h, c_cross_above_donc_h, c_above_KAMA, c_cross_above_KAMA, ar_down_above_up)]
entries = [1 if (ews > entry_threshhold_weight) else 0 for ews in entry_weight_sums]
exits = [1 if (ews > exit_threshhold_weight) else 0 for ews in exit_weight_sums]
return entries, exits
```

## Putting it All Together

Here's a template first before you see the completed product after it:
```{py}
import numpy as np
import vectorbt as vbt

class [class name]:

    indicator_factory_dict = {...} # **kwargs*  
    optimize_dict = {...} # **kwargs* 
    default_dict = {...} # **kwargs*

    # * The **kwargs represents the strat params. Another reminder: THEY ALL MUST BE IN THE SAME ORDER THROUGHOUT FOR THE CODE TO WORK!

    def indicator_func(open, high, low, close, volume, **kwargs*):   

        [DEFINE INDICATORS HERE]

        [DETERMINE SIGNAL BOOLS HERE]

        [DEFINE SIGNAL WEIGHTINGS HERE]

        [DEFINE WEIGHTING PARAMS HERE]

        [DEFINE TRADING BOT LOGIC HERE IN TERMS OF entries, exits TIME-SERIES]

        return entries, exits
```

Now, for our final product in this example!

```{py}
import numpy as np
import vectorbt as vbt

class Donchian_KAMA_Aroon:
    
    indicator_factory_dict = {
        'class_name':'Donchian_KAMA_Aroon',
        'short_name':'Donchian_KAMA_Aroon',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['donc_window',
                       'KAMA_window',
                       'KAMA_pow1',
                       'KAMA_pow2',
                       'ar_window',
                       'W_c_under_donc_l',
                       'W_c_cross_under_donc_l',
                       'W_c_under_KAMA',
                       'W_c_cross_under_KAMA',
                       'W_ar_up_above_down',
                       'W_c_above_donc_h',
                       'W_c_cross_above_donc_h',
                       'W_c_above_KAMA',
                       'W_c_cross_above_KAMA',
                       'W_ar_down_above_up',
                       'tol_p_entry',
                       'tol_p_exit'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
                     'donc_window':[7,10,21],
                     'KAMA_window':[7,10,21],
                     'KAMA_pow1':[2,5],
                     'KAMA_pow2':[20,30],
                     'ar_window':[17,25,33],
                     'W_c_under_donc_l':[1],
                     'W_c_cross_under_donc_l':[1],
                     'W_c_under_KAMA':[1],
                     'W_c_cross_under_KAMA':[1],
                     'W_ar_up_above_down':[2],
                     'W_c_above_donc_h':[1],
                     'W_c_cross_above_donc_h':[1],
                     'W_c_above_KAMA':[1],
                     'W_c_cross_above_KAMA':[1],
                     'W_ar_down_above_up':[2],
                     'tol_p_entry':[0.16, 0.32, 0.49, 0.65, 0.82, 0.99],
                     'tol_p_exit':[0.16, 0.32, 0.49, 0.65, 0.82, 0.99]
    }

    default_dict = {
                    'donc_window':10,
                    'KAMA_window':10,
                    'KAMA_pow1':2,
                    'KAMA_pow2':30,
                    'ar_window':25,
                    'W_c_under_donc_l':1,
                    'W_c_cross_under_donc_l':1,
                    'W_c_under_KAMA':1,
                    'W_c_cross_under_KAMA':1,
                    'W_ar_up_above_down':2,
                    'W_c_above_donc_h':1,
                    'W_c_cross_above_donc_h':1,
                    'W_c_above_KAMA':1,
                    'W_c_cross_above_KAMA':1,
                    'W_ar_down_above_up':2,
                    'tol_p_entry':0.49,
                    'tol_p_exit':0.49
    }

    def indicator_func(
                       open, high, low, close, volume, 
                       donc_window,
                       KAMA_window,
                       KAMA_pow1,
                       KAMA_pow2,
                       ar_window,
                       W_c_under_donc_l,
                       W_c_cross_under_donc_l,
                       W_c_under_KAMA,
                       W_c_cross_under_KAMA,
                       W_ar_up_above_down,
                       W_c_above_donc_h,
                       W_c_cross_above_donc_h,
                       W_c_above_KAMA,
                       W_c_cross_above_KAMA,
                       W_ar_down_above_up,
                       tol_p_entry,
                       tol_p_exit
                      ):        

        # Define Indicators
        ar_up_i = vbt.IndicatorFactory.from_ta('AroonIndicator.ar_up')
        ar_down_i = vbt.IndicatorFactory.from_ta('AroonIndicator.ar_down')
        donc_l_i = vbt.IndicatorFactory.from_ta('DonchianChannel.donc_channel_lband')
        donc_h_i = vbt.IndicatorFactory.from_ta('DonchianChannel.donc_channel_hband')
        kama_i = vbt.IndicatorFactory.from_ta('KAMAIndicator')

        # Run Indicators
        ar_up = ar_up_i.run(close,window=ar_window, fillna = True)
        ar_down = ar_down_i.run(close,window=ar_window, fillna = True)
        donc_l_i = donc_l_i.run(high,low,close,window=donc_window, fillna = True)
        donc_h_i = donc_h_i.run(high,low,close,window=donc_window, fillna = True)
        kama = kama_i.run(close,window=KAMA_window,pow1=KAMA_pow1,pow2=KAMA_pow2, fillna = True)


        len_dict = {'ar_up':len(ar_up),'ar_down':len(ar_down),'donc_l':len(donc_l),'donc_h':len(donc_h),'kama':len(kama)}
        print('Length Check for Indicator Time-Series:')
        for ld in len_dict:
            print(ld)
            print(len_dict[ld])
            print('\n')

       # Define BUY Signals 
        c_under_donc_l = [0 if (d_l is None) else (W_c_under_donc_l if (cl < d_l) else 0) for cl, d_l in zip(close, donc_l)]
        c_cross_under_donc_l = close.close_crossed_below(donc_l)
        c_under_KAMA = [0 if (k is None) else (W_c_under_KAMA if (cl < k) else 0) for cl, k in zip(close, KAMA)]
        c_cross_under_KAMA = close.close_crossed_below(kama)
        ar_up_above_down = [0 if ((a_d is None) or (a_u is None)) else (W_ar_up_above_down if a_u > a_d else 0) for a_d, a_u in zip(ar_down, ar_up)]

        # Define SELL Signals 
        c_above_donc_h = [0 if (d_l is None) else (W_c_under_donc_l if (cl > d_h) else 0) for cl, d_l in zip(close, donc_h)]
        c_cross_above_donc_h = close.close_crossed_above(donc_h)
        c_above_KAMA = [0 if (k is None) else (W_c_above_KAMA if (cl > k) else 0) for cl, k in zip(close, KAMA)]
        c_cross_above_KAMA = close.close_crossed_above(kama)
        ar_down_above_up = [0 if ((a_d is None) or (a_u is None)) else (W_ar_down_above_up if (a_d > a_u) else 0) for a_d, a_u in zip(ar_down, ar_up)]

        # Entry Weighted Decisioning Threshhold Calculation
        entry_weights_list = [W_c_under_donc_l,
                              W_c_cross_under_donc_l,
                              W_c_under_KAMA,
                              W_c_cross_under_KAMA,
                              W_ar_up_above_down]
        total_entry_weights = sum(entry_weights_list)
        entry_threshhold_weight = tol_p_entry * total_entry_weights 

        # Exit Weighted Decisioning Threshhold Calculation
        exit_weights_list = [W_c_above_donc_h,
                              W_c_cross_above_donc_h,
                              W_c_above_KAMA,
                              W_c_cross_above_KAMA,
                              W_ar_down_above_up]
        total_exit_weights = sum(exit_weights_list)
        exit_threshhold_weight = tol_p_exit * total_exit_weights
        
        # BUY/SELL Signal Calculation 
        entry_weight_sums = [sum(x) for x in zip(c_under_donc_l, c_cross_under_donc_l, c_under_KAMA, c_cross_under_KAMA, ar_down_above_up)]
        exit_weight_sums = [sum(x) for x in zip(c_above_donc_h, c_cross_above_donc_h, c_above_KAMA, c_cross_above_KAMA, ar_down_above_up)]
        entries = [1 if (ews > entry_threshhold_weight) else 0 for ews in entry_weight_sums]
        exits = [1 if (ews > exit_threshhold_weight) else 0 for ews in exit_weight_sums]
        return entries, exits


```

## Running Backtesting with the Custom BackTester.py Object
This part is actually very simple! We only need to concern ourselves with one import statement, alongside another part at the bottom of the script and replace STRATEGY_NAME with the name of our strategy-do not adjust anything else! The import statement will have the format (the strategy file must be in the src.trading_bot.strategies directory for this to work):
```{py}
from strategies.TestStratVBT import TestStratVBT
```
And the bit at the end where we have to edit the STRATEGY_NAME looks like this:
```{py}
if __name__ == '__main__': 
    backtest_params = {'init_cash': 10_000, 'fees': 0.005}

    b = BackTester(
        strategies = [STRATEGY_NAME],
        optimization_metric = 'Deflated Sharpe Ratio',
        backtest_params = backtest_params
    )

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end backtest_start) / 60.0, 2)))
```
So, our import will look like this for our current example:
```{py}
from strategies.Donchian_KAMA_Aroon import Donchian_KAMA_Aroon
```
And the bottom section will look like this:
```{py}
if __name__ == '__main__': 
    backtest_params = {'init_cash': 10_000, 'fees': 0.005}

    b = BackTester(
        strategies = [Donchian_KAMA_Aroon],
        optimization_metric = 'Deflated Sharpe Ratio',
        backtest_params = backtest_params
    )

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end backtest_start) / 60.0, 2)))
```

Then, from your terminal, run:

```
cd ...path_to_src/src/backtest/
python3 Backtest.py
```
