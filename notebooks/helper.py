import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
import vectorbt as vbt
import redshift_connector


def calculate_triple_barrier_labels(ohlcv_df, atr_window, max_holding_time):
    # Calculate ATR using vectorbt
    high = ohlcv_df['price_high']
    low = ohlcv_df['price_low']
    close = ohlcv_df['price_close']

    atr = vbt.ATR.run(high, low, close, window=atr_window).atr

    # Initialize labels array with zeros
    labels = np.zeros(len(close))

    # Loop through the close price series
    for i in range(len(close) - max_holding_time):
        # Set the upper and lower barriers
        upper_barrier = close[i] + atr[i]
        lower_barrier = close[i] - atr[i]
        barrier_hit = False

        # Check if price hits the barriers within the max holding time
        for j in range(i + 1, i + max_holding_time + 1):
            # Ensure the loop does not go beyond the length of the series
            if j >= len(close):
                break

            if high[j] >= upper_barrier:
                labels[i] = 1  # Upper barrier hit
                barrier_hit = True
                break
            elif low[j] <= lower_barrier:
                labels[i] = -1  # Lower barrier hit
                barrier_hit = True
                break

        # If no barriers are hit, determine label based on closing price after max holding time
        if not barrier_hit:
            labels[i] = 1 if close[j] > close[i] else -1

    # Create a Pandas Series from the labels array
    label_series = pd.Series(labels, index=ohlcv_df.index)

    return label_series

def execute_query(query, cols, date_col):
    with redshift_connector.connect(
        host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
        database = 'token_price',
        user = 'administrator',
        password = 'Free2play2'
    ) as conn:
        with conn.cursor() as cursor:
            # Execute query on Redshift and return result
            cursor.execute(query)
            tuples = cursor.fetchall()
            
            # Return queried data as a DataFrame
            df = pd.DataFrame(tuples, columns = cols).set_index(date_col)

            # For each column in DataFrame, convert to float, skip if not possible
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except:
                    continue

            # Fill in any gaps in data with last seen value
            df = df.drop_duplicates().asfreq(freq = 'H', method = 'ffill')

            return df

def get_ohlcv_data(base, quote, exchange):
    query = """
    SELECT
        time_period_end,
        price_open,
        price_high,
        price_low,
        price_close,
        volume_traded
    FROM token_price.coinapi.price_data_1h
    WHERE 
        asset_id_base = '{}' AND
        asset_id_quote = '{}' AND
        exchange_id = '{}'
    ORDER BY time_period_start ASC
    """.format(base, quote, exchange)

    ohlcv_1h_cols = ['time_period_end', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']
    ohlcv_1h = execute_query(
        query = query,
        cols = ohlcv_1h_cols,
        date_col = 'time_period_end'
    )

    return ohlcv_1h

def calculate_train_performance(X_train, y_train, model):
    # Train Predictions
    y_pred = model.predict(X_train)

    # Classification report
    print('Train Classification Report')
    print()
    print(classification_report(y_train, y_pred))

    # Various metrics rounded to 4 decimal places
    accuracy = round(accuracy_score(y_train, y_pred), 4)
    precision = round(precision_score(y_train, y_pred, average = 'weighted'), 4)
    recall = round(recall_score(y_train, y_pred, average = 'weighted'), 4)
    f1 = round(f1_score(y_train, y_pred, average = 'weighted'), 4)

    # Print various metrics
    print('Train Accuracy: ', accuracy)
    print('Train Precision: ', precision)
    print('Train Recall: ', recall)
    print('Train F1 Score: ', f1)

    # Plot horizontal bar plot of top 20 most important features
    feature_importances = pd.Series(model.feature_importances_, index = X_train.columns)
    feature_importances.nlargest(20).sort_values().plot(
        kind = 'barh', 
        figsize = (10, 6), 
        title = 'Top 20 Most Important Features',
        color = 'green',
        xlabel = 'Feature Importance',
        fontsize = 14,
        width = 0.8,
        edgecolor = 'black',
        linewidth = 1.2
    )

def simulate_trading(y_test, y_pred_probs, threshold, price_data, atr_data, max_holding=24):
    # Assume price_data is a DataFrame with 'open', 'high', 'low', 'close'
    # and atr_data is a Series with the ATR values for the corresponding price_data
    emv = 0
    num_trades = 0
    in_position = False
    position_entry_index = None
    position_entry_price = None
    position_upper_barrier = None
    position_lower_barrier = None
    
    for i in range(len(y_test)):
        # Check if we can take a new position
        if not in_position and y_pred_probs[i] > threshold:
            in_position = True
            num_trades += 1
            position_entry_index = i
            position_entry_price = price_data.iloc[i]['price_close']
            position_upper_barrier = position_entry_price + atr_data[i]
            position_lower_barrier = position_entry_price - atr_data[i]
            
        # Check if the position should be closed because a barrier was crossed or max holding period reached
        if in_position:
            current_high = price_data.iloc[i]['price_high']
            current_low = price_data.iloc[i]['price_low']
            if current_high >= position_upper_barrier:
                emv += 2  # Profit as upper barrier crossed
                in_position = False
            elif current_low <= position_lower_barrier:
                emv -= 1  # Loss as lower barrier crossed
                in_position = False
            elif (i - position_entry_index) >= max_holding:
                # Check if the position is in profit or loss at the end of holding period
                current_close = price_data.iloc[i]['price_close']
                if current_close > position_entry_price:
                    emv += 2  # Profit as price is above entry after holding period
                else:
                    emv -= 1  # Loss as price is below entry after holding period
                in_position = False

    return emv, num_trades

def calculate_test_performance(X_test, y_test, model, price_data, atr_data):
    # Test predictions
    y_pred_probs = model.predict_proba(X_test)[:,1]

    # Calculate precision-recall pairs for different probability thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    pr_auc = auc(recall, precision)

    # Find the threshold that maximizes EMV
    max_emv = -np.inf
    optimal_threshold = 0.0
    for threshold in thresholds:
        emv, num_trades = simulate_trading(y_test, y_pred_probs, threshold, price_data, atr_data)
        if emv > max_emv:
            max_emv = emv
            optimal_threshold = threshold

    optimal_recall = recall[np.where(thresholds == optimal_threshold)]
    optimal_precision = precision[np.where(thresholds == optimal_threshold)]
    y_pred = np.where(y_pred_probs >= optimal_threshold, 1, -1)

    # Classification report
    print('Test Classification Report\n')
    print(classification_report(y_test, y_pred))

    # Various metrics rounded to 4 decimal places
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision_metric = round(precision_score(y_test, y_pred, average='weighted'), 4)
    recall_metric = round(recall_score(y_test, y_pred, average='weighted'), 4)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)

    # Print various metrics
    print('Test Accuracy: ', accuracy)
    print('Test Precision: ', precision_metric)
    print('Test Recall: ', recall_metric)
    print('Test F1 Score: ', f1)

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))

    plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (area = {pr_auc:.3f})')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='blue')

    plt.scatter(optimal_recall, optimal_precision, marker='o', color='red', 
                label=f'Optimal Prediction % Threshold = {optimal_threshold:.4f}', s = 50)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()