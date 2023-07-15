from datetime import timedelta
import pandas as pd
import numpy as np

def calculate_performance_metrics(oos_equity_curve, oos_trades, oos_price_data):
    ########################### HELPER FUNCTIONS ####################################
    def exposure_time(duration, trades):
        exposure = timedelta(days = 0, hours = 0)

        for i in range(len(trades)):
            entry_date = pd.to_datetime(trades.at[i, 'entry_date'])
            exit_date = pd.to_datetime(trades.at[i, 'exit_date'])
            trade_duration = exit_date - entry_date
            exposure += trade_duration
        
        return round(exposure / duration * 100, 2)

    def buy_and_hold_return(data):
        token1_start_value = data.at[data.index[0], data.columns[0]]
        token1_end_value = data.at[data.index[-1], data.columns[0]]
        buy_and_hold_return_token_1 = round((token1_end_value - token1_start_value) / token1_start_value * 100, 2)
        
        token2_start_value = data.at[data.index[0], data.columns[1]]
        token2_end_value = data.at[data.index[-1], data.columns[1]]
        buy_and_hold_return_token_2 = round((token2_end_value - token2_start_value) / token2_start_value * 100, 2)

        return max([buy_and_hold_return_token_1, buy_and_hold_return_token_2])
    
    def sharpe_ratio(equity):
        returns = equity.pct_change()
        mean_returns = returns.mean()
        std_returns = returns.std()
        
        try:
            return np.sqrt(8760) * mean_returns / std_returns 
        except:
            return np.nan
        
    def sortino_ratio(equity):
        returns = equity.pct_change()
        negative_returns = returns[returns['equity'] < 0]
        
        mean_returns = returns.mean()
        std_negative_returns = negative_returns.std()
        
        try:
            return np.sqrt(8760) * mean_returns / std_negative_returns 
        except:
            return np.nan
        
    def calmar_ratio(equity):
        num_years = len(equity) / 8760
        cum_ret_final = (1 + equity.pct_change()).prod().squeeze()
        annual_returns = cum_ret_final ** (1 / num_years) - 1
        
        try:
            return annual_returns / abs(max_drawdown(equity) / 100)
        except:
            return np.nan

    def cagr_over_avg_drawdown(equity):
        num_years = len(equity) / 8760
        cum_ret_final = (1 + equity.pct_change()).prod().squeeze()
        annual_returns = cum_ret_final ** (1 / num_years) - 1
        
        try:
            return annual_returns / abs(avg_drawdown(equity) / 100)
        except:
            return np.nan

    def profit_factor(trades):
        if len(trades) == 0:
            return np.nan
        
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        return gross_profit / gross_loss
    
    def max_drawdown(equity):
        rolling_max_equity = equity.cummax()
        drawdown = (equity / rolling_max_equity) - 1
        max_dd = drawdown.min()
        return round(max_dd * 100, 2)
    
    def avg_drawdown(equity):
        rolling_max_equity = equity.cummax()
        drawdown = (equity / rolling_max_equity) - 1
        avg_dd = drawdown.mean()
        return round(avg_dd * 100, 2)
    
    def max_drawdown_duration(equity):
        equity = equity.dropna()
        dates = pd.Series([pd.to_datetime(equity.index[0])])
        
        diff = equity.cummax().diff().fillna(0)
        diff = diff[diff['equity'] != 0]

        for date in diff.index:
            date = pd.to_datetime(date)
            dates = pd.concat([dates, pd.Series(date)], ignore_index = True)

        return dates.diff().dt.total_seconds().max()
    
    def avg_drawdown_duration(equity):
        equity = equity.dropna()
        dates = pd.Series([pd.to_datetime(equity.index[0])])
        
        diff = equity.cummax().diff().fillna(0)
        diff = diff[diff['equity'] != 0]

        for date in diff.index:
            date = pd.to_datetime(date)
            dates = pd.concat([dates, pd.Series(date)], ignore_index = True)

        return dates.diff().dt.total_seconds().mean()

    def win_rate(trades):
        if len(trades) == 0:
            return np.nan
        
        num_winning_trades = len(trades[trades['pnl_pct'] > 0])
        num_trades_total = len(trades)

        return round(num_winning_trades / num_trades_total * 100, 2)
    
    def best_trade(trades):
        if len(trades) == 0:
            return np.nan
        
        return round(trades['pnl_pct'].max() * 100, 2)
        
    def worst_trade(trades):
        if len(trades) == 0:
            return np.nan
        
        return round(trades['pnl_pct'].min() * 100, 2)
    
    def avg_trade(trades):
        if len(trades) == 0:
            return np.nan
        
        return round(trades['pnl_pct'].mean() * 100, 2)
    
    def max_trade_duration(trades):
        if len(trades) == 0:
            return np.nan
        
        return (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).dt.total_seconds().max()            
        
    def avg_trade_duration(trades):
        if len(trades) == 0:
            return np.nan
        
        return (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).dt.total_seconds().mean()           
    #################################### HELPER FUNCTIONS END ####################################
            
    start = oos_price_data.index[0]
    end = oos_price_data.index[-1]
    duration = pd.to_datetime(end) - pd.to_datetime(start)

    return_pct = (oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0]
    
    metrics_dict = {
        'duration':duration, 
        'exposure_pct': round(exposure_time(duration, oos_trades), 2),
        'equity_final':round(oos_equity_curve.dropna().iloc[-1]['equity'], 2),
        'equity_peak':round(oos_equity_curve['equity'].max(), 2),
        'return_pct':round(return_pct * 100, 2),
        'buy_and_hold_return_pct':round(buy_and_hold_return(oos_price_data), 2),
        'sharpe_ratio':round(sharpe_ratio(oos_equity_curve), 2), 
        'sortino_ratio':round(sortino_ratio(oos_equity_curve), 2),
        'calmar_ratio':round(calmar_ratio(oos_equity_curve), 2),
        'cagr_over_avg_drawdown':round(cagr_over_avg_drawdown(oos_equity_curve), 2),
        'profit_factor':round(profit_factor(oos_trades), 2),
        'max_dd_pct':round(max_drawdown(oos_equity_curve), 2),
        'avg_dd_pct':round(avg_drawdown(oos_equity_curve), 2),
        'max_dd_duration':max_drawdown_duration(oos_equity_curve), 
        'avg_dd_duration':avg_drawdown_duration(oos_equity_curve),
        'num_trades':len(oos_trades), 
        'win_rate_pct':round(win_rate(oos_trades), 2), 
        'best_trade_pct':round(best_trade(oos_trades), 2),
        'worst_trade_pct':round(worst_trade(oos_trades), 2), 
        'avg_trade_pct':round(avg_trade(oos_trades), 2),
        'max_trade_duration':max_trade_duration(oos_trades),
        'avg_trade_duration':avg_trade_duration(oos_trades),
    }

    return pd.DataFrame(metrics_dict).reset_index(drop = True)
