import vectorbt as vbt
import numpy as np
import pandas as pd

class MomentumVol:

    indicator_factory_dict = {
        'class_name':'MomentumVol',
        'short_name':'MomentumVol',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':[
            'momentum_period', 'atr_window', 'volatility_ratio_window',
            'volume_window', 'volatility_threshold', 'high_volatility_threshold', 
            'time_exit_hours', 'cooldown_period'
        ],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'momentum_period': [12, 24],
        'atr_window': [12, 24],
        'volatility_ratio_window': [48, 72],
        'volume_window': [6, 12],
        'volatility_threshold': [0.5, 1],
        'high_volatility_threshold': [1.5, 2],
        'time_exit_hours': [48],
        'cooldown_period': [0, 24]
    }

    def indicator_func(open, high, low, close, volume,
                    momentum_period = 24,
                    atr_window = 14, 
                    volatility_ratio_window = 168, 
                    volume_window = 24, 
                    volatility_threshold = 0.5, 
                    high_volatility_threshold = 1.5, 
                    time_exit_hours = 24,
                    cooldown_period = 24):

        df = pd.DataFrame({
            'open': open,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        # Calculate indicators
        df['momentum'] = df['close'].pct_change(momentum_period)
        df['atr'] = vbt.ATR.run(df['high'], df['low'], df['close'], window = atr_window).atr
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window = volatility_ratio_window).mean()
        df['volume_avg'] = df['volume'].rolling(window = volume_window).mean()

        # Generate entry signals
        df['long_entry'] = ((df['momentum'] > 0) & (df['momentum'].shift(1) < df['momentum']) &
                            (df['volatility_ratio'] <= volatility_threshold) &
                            (df['volume'] > df['volume_avg']))

        # Initialize exit signals
        df['volatility_exit'] = df['volatility_ratio'] > high_volatility_threshold
        df['momentum_exit'] = df['momentum'] < 0

        # Combine exit signals
        df['long_exit'] = df[['volatility_exit', 'momentum_exit']].any(axis=1)

        # Implement time-based exit logic
        df['time_since_entry'] = 0
        position_open = False

        for i in range(1, len(df)):
            if df['long_entry'].iloc[i] and not position_open:
                df.loc[i, 'time_since_entry'] = 0
                position_open = True
            elif position_open and not df['long_exit'].iloc[i]:
                df.loc[i, 'time_since_entry'] = df['time_since_entry'].iloc[i - 1] + 1
            elif df['long_exit'].iloc[i]:
                position_open = False

        df['time_exit'] = df['time_since_entry'] >= time_exit_hours
        df['long_exit'] |= df['time_exit']
        
        # Implement cooldown logic
        df['cooldown'] = 0

        for i in range(len(df) - cooldown_period):
            if df['long_exit'].iloc[i]:
                df.loc[df.index[i+1:i+1+cooldown_period], 'cooldown'] = 1

        entries = (df['long_entry'] & (df['cooldown'] == 0)).values
        exits = df['long_exit'].values

        return entries, exits