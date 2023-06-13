from numba import njit, prange
import numpy as np

@njit
def __exit_trade_numba(positions, data, dates, trades, i, is_long, commission, slippage, curr_capital):
    if is_long:
        long_symbol_index_pos = 0
        long_symbol_index_data = 1

        short_symbol_index_pos = 1
        short_symbol_index_data = 0
    else:
        long_symbol_index_pos = 1
        long_symbol_index_data = 0
        
        short_symbol_index_pos = 0
        short_symbol_index_data = 1
        
    # Set long and short position amount for current timestamp to 0
    # to simulate closing the trade
    positions[i][long_symbol_index_pos] = 0
    positions[i][short_symbol_index_pos] = 0

    # Set the exit date of the current trade to the current timestamp
    trades[-1][1] = dates[i]
    trades[-1][3] = i
    
    # Calculate pnl and pnl % from trade

    # Entry and exit dates of most current trade
    start = int(trades[-1][2])
    end = int(trades[-1][3])

    # Calculate the PnL from the long position
    start_value_long = data[start][long_symbol_index_data] * (1 + slippage) * positions[start][long_symbol_index_pos]
    end_value_long = data[end][long_symbol_index_data] * (1 - slippage) * positions[start][long_symbol_index_pos]
    long_pnl = (end_value_long - start_value_long) * (1 - commission)

    # Calculate the PnL from the short position
    start_value_short = data[start][short_symbol_index_data] * (1 - slippage) * positions[start][short_symbol_index_pos]
    end_value_short = data[end][short_symbol_index_data] * (1 + slippage) * positions[start][short_symbol_index_pos]
    short_pnl = (start_value_short - end_value_short) * (1 - commission)

    # Calculate the PnL from the entire trade 
    trade_pnl = long_pnl + short_pnl
    total_investment = start_value_long + start_value_short
    trade_pnl_pct = trade_pnl / total_investment
    
    # Set the PnL and PnL % of the current trade
    trades[-1][7] = trade_pnl_pct
    trades[-1][6] = trade_pnl

    curr_capital -= (end_value_short * (1 + commission))
    curr_capital += (end_value_long * (1 - commission))

    return positions, trades, curr_capital

@njit
def __enter_trade_numba(positions, data, dates, trades, curr_capital, pct_capital_per_trade, commission, slippage, is_long, i, h):
    if is_long:
        long_symbol_index_pos = 0
        long_symbol_index_data = 1

        short_symbol_index_pos = 1
        short_symbol_index_data = 0
    else:
        long_symbol_index_pos = 1
        long_symbol_index_data = 0
        
        short_symbol_index_pos = 0
        short_symbol_index_data = 1

    p1 = data[i][short_symbol_index_data]
    p2 = data[i][long_symbol_index_data]
    c = curr_capital * pct_capital_per_trade

    if is_long:
        n1 = (c / (h * p1 + p2)) * h
        n2 = c / (h * p1 + p2)

        long_allocation = n2 * p2
        short_allocation = n1 * p1
        
        # Set long and short position amounts for current timestamp
        positions[i][long_symbol_index_pos] = n2
        positions[i][short_symbol_index_pos] = n1
        
        # Initialize a new trade and append it to the trades DataFrame
        new_trade = np.array([
            dates[i],
            -1,
            i,
            np.nan,
            n2,
            n1,
            np.nan,
            np.nan,
            is_long
        ])
        
        new_trade = np.reshape(new_trade, (1, -1))
        trades = np.vstack((trades, new_trade))
        
        # Track number of units in current long and short position
        curr_position_long_units = n2
        curr_position_short_units = n1
    else:
        n1 = c / (h * p1 + p2)
        n2 = (c / (h * p1 + p2)) * h

        long_allocation = n1 * p2
        short_allocation = n2 * p1
        
        # Set long and short position amounts for current timestamp
        positions[i][long_symbol_index_pos] = n1
        positions[i][short_symbol_index_pos] = n2
        
        # Initialize a new trade and append it to the trades DataFrame
        new_trade = np.array([
            dates[i],
            -1,
            i,
            np.nan,
            n2,
            n1,
            np.nan,
            np.nan,
            is_long
        ])

        new_trade = np.reshape(new_trade, (1, -1))
        trades = np.vstack((trades, new_trade))
        
        # Track number of units in current long and short position
        curr_position_long_units = n1
        curr_position_short_units = n2
    
    curr_capital -= (long_allocation * (1 + commission))
    curr_capital += (short_allocation * (1 - commission))

    return positions, trades, curr_capital, curr_position_long_units, curr_position_short_units

@njit            
def generate_positions_numba(positions, data, dates, trades, position, curr_capital, pct_capital_per_trade, commission, slippage, sl, tp):
    cp_long = 0
    cp_short = 0

    for i in range(len(data)):
        if curr_capital < 0:
            
            if trades[-1][1] == -1:

                # Close the current trade
                positions, trades, curr_capital = __exit_trade_numba(
                    positions = positions,
                    data = data,
                    dates = dates,
                    trades = trades,
                    i = i,
                    is_long = trades[-1][-1],
                    commission = commission,
                    slippage = slippage,
                    curr_capital = curr_capital
                )

                position = 0

                return trades, curr_capital
            else:
                return trades, curr_capital
        
        # Retrieve entry and exit signals at current timestamp          
        entry_signal = data[i][4]
        exit_signal = data[i][5]

        # If not in a trade at current timestamp
        if not position:
            
            # If long or short entry signal is received and we have capital remaining
            if (entry_signal == 1 or entry_signal == -1): 
                
                # If we're on the last timestamp or if rolling hedge ratio is negative 
                # then don't open a trade
                if (i == len(data) - 1) or (data[i][2] <= 0):
                    continue
                    
                is_long = entry_signal == 1
                positions, trades, curr_capital, curr_position_long_units, curr_position_short_units = __enter_trade_numba(
                    positions = positions,
                    data = data,
                    dates = dates,
                    trades = trades,
                    curr_capital = curr_capital,
                    pct_capital_per_trade = pct_capital_per_trade,
                    commission = commission,
                    slippage = slippage,
                    is_long = is_long,
                    i = i,
                    h = data[i][2]
                )
                
                cp_long = curr_position_long_units
                cp_short = curr_position_short_units
                
                position = 1

            # If no entry signal is received
            else:

                # Set long and short position amounts for current timestamp to 0
                # since we're not in a trade
                positions[i][0] = 0
                positions[i][1] = 0

        # If in a trade at current timestamp
        else:
            is_long = trades[-1][-1]
            if is_long:
                long_symbol_index_pos = 0
                long_symbol_index_data = 1

                short_symbol_index_pos = 1
                short_symbol_index_data = 0
            else:
                long_symbol_index_pos = 1
                long_symbol_index_data = 0
                
                short_symbol_index_pos = 0
                short_symbol_index_data = 1

            # Entry and exit dates of most current trade
            start = int(trades[-1][2])
            curr_date = i

            start_value_long = data[start][long_symbol_index_data] * positions[start][long_symbol_index_pos]
            curr_value_long = data[curr_date][long_symbol_index_data] * positions[start][long_symbol_index_pos]
            long_pnl = (curr_value_long - start_value_long) * (1 - commission)

            # Calculate the PnL from the short position
            start_value_short = data[start][short_symbol_index_data] * positions[start][short_symbol_index_pos]
            curr_value_short = data[curr_date][short_symbol_index_data] * positions[start][short_symbol_index_pos]
            short_pnl = (start_value_short - curr_value_short) * (1 - commission)

            # Calculate the PnL from the entire trade 
            trade_pnl = long_pnl + short_pnl
            total_investment = start_value_long + start_value_short
            trade_pnl_pct = trade_pnl / total_investment

            if (trade_pnl_pct <= -sl) or (trade_pnl_pct >= tp):
                
                # Close the current trade
                positions, trades, curr_capital = __exit_trade_numba(
                    positions = positions,
                    data = data,
                    dates = dates,
                    trades = trades,
                    i = i,
                    is_long = is_long,
                    commission = commission,
                    slippage = slippage,
                    curr_capital = curr_capital
                )

                position = 0

            # If curr trade is long and we get a long exit or curr trade is short and we get a short exit
            if (is_long and exit_signal == 1) or (not is_long and exit_signal == -1):
                
                # Close the current trade
                positions, trades, curr_capital = __exit_trade_numba(
                    positions = positions,
                    data = data,
                    dates = dates,
                    trades = trades,
                    i = i,
                    is_long = is_long,
                    commission = commission,
                    slippage = slippage,
                    curr_capital = curr_capital
                )

                position = 0

            # Otherwise 
            else:
                # Set long and short position amounts for current timestamp to
                # the amounts in the current trade since we haven't exited the
                # current trade yet

                positions[i][long_symbol_index_pos] = cp_long
                positions[i][short_symbol_index_pos] = cp_short

        # If the backtest reaches the final timestamp and the current trade hasn't
        # been exited yet 
        if i == len(data) - 1 and len(trades) > 0 and trades[-1][1] == -1:

            # Close the current trade
            positions, trades, curr_capital = __exit_trade_numba(
                positions = positions,
                data = data,
                dates = dates,
                trades = trades,
                i = i,
                is_long = is_long,
                commission = commission,
                slippage = slippage,
                curr_capital = curr_capital
            )

            position = 0

    return trades, curr_capital

@njit(parallel = True)
def rolling_hedge_ratios_numba(y, x, window):
    n = len(y)
    hedge_ratios = np.full(n, np.nan)  # Initialize an array to store hedge ratios
    
    for i in prange(window - 1, n):
        Y = y[i - window + 1 : i + 1]
        X = x[i - window + 1 : i + 1]

        # Adding a constant to X
        X = np.vstack((X, np.ones(len(X)))).T
        
        # Least square solution
        result = np.linalg.lstsq(X, Y)[0]
        
        hedge_ratios[i] = result[0]  # Coefficient of x
        
    return hedge_ratios

@njit(parallel = True)
def rolling_z_score(data, window):
    z_scores = np.empty(len(data))
    z_scores[:window] = np.nan  # first `window` values have no preceding window

    for i in prange(window, len(data)):
        mean_i = np.mean(data[i-window:i])
        std_i = np.std(data[i-window:i])
        z_scores[i] = (data[i] - mean_i) / std_i if std_i > 0 else 0.0  # avoid division by zero

    return z_scores

@njit(parallel = True)
def generate_trading_signals(z, z_prev, zl, zu):
    en = np.where(
        ((z_prev > zl) & (z < zl)),
        1,
        0
    )

    en = np.where(
        ((z_prev < zu) & (z > zu)),
        -1,
        en
    )

    ex = np.where(
        ((z_prev < zu) & (z > zu)),
        1,
        0
    )

    ex = np.where(
        ((z_prev > zl) & (z < zl)),
        -1,
        ex
    )
    
    return en, ex
