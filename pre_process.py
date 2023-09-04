import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def pre_process(ticker_data,Moving_window):
    #Calculate and insert Average price along each interval
    ticker_data_close = ticker_data['Close'].to_numpy()
    ticker_data_open = ticker_data['Open'].to_numpy()
    ticker_data_average = (ticker_data_close + ticker_data_open)/2
    ticker_data.insert(1, 'Average', ticker_data_average)
    
    #Calculate RSI
    change = ticker_data['Average'].diff()
    change.dropna(inplace = True)
    change_up = change.copy()
    change_down = change.copy()
    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0
    avg_up = change_up.rolling(14).mean()
    avg_down = change_down.rolling(14).mean().abs()
    rsi = 100 * avg_up / (avg_up + avg_down)
    
    #Drop NaN values from RSI
    rsi.dropna(inplace=True)

    # Calculate 12-day and 26-day EMAs
    ema_12 = ticker_data['Average'].ewm(span=12, adjust=False).mean()
    ema_26 = ticker_data['Average'].ewm(span=26, adjust=False).mean()

    # Calculate MACD line
    macd_line = ema_12 - ema_26

    # Calculate 9-day EMA of MACD (Signal line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Calculate whether Signal is above MACD
    signal_above_macd = (signal_line > macd_line).astype(int)

    # Insert data into dataframe
    ticker_data.insert(1,'MACD_Value', macd_line)
    ticker_data.insert(1,'Signal_Value', signal_line)
    ticker_data.insert(1,'Signal_MACD_strength',signal_above_macd)

    #Calculate Moving average slope
    ma = ticker_data['Average'].rolling(window = Moving_window).mean()
    ticker_data.insert(1, 'Moving Average', ma)
    ma_slope = ticker_data['Moving Average'].diff()/Moving_window
    tangent_inverse = np.arctan(ma_slope)
    normalization_factor = np.pi/2
    normalized_tangent_inverse = tangent_inverse / normalization_factor
    ticker_data.insert(1, 'Normalized Moving Average Slope', normalized_tangent_inverse)
    ticker_data.insert(1,'RSI', rsi)

    #Insert Change in average prices per interval to the DataFrame
    change = change/ticker_data_average[1:]
    ticker_data.insert(1, 'Change', change)
    ticker_data.fillna(0,inplace = True) #Change the first row to 0 since we don't have data before the first data point

    #Calculate and insert typical price
    typical_price = (ticker_data['High'] + ticker_data['Close'] + ticker_data['Low'])/3
    ticker_data.insert(1,'Typical_Price', typical_price)

    #Calculate upper & lower Bollinger Bands
    #calculate std over last 20 days:
    std_array = np.zeros(len(typical_price))
    for i in range(Moving_window, len(typical_price)):
        std_array[i] = np.std(typical_price[i-Moving_window:i])
    BOLu = ma + 2*std_array
    BOLl = ma - 2*std_array
    ma_BOLu = BOLu.rolling(window = Moving_window).mean()
    ma_BOLl = BOLl.rolling(window = Moving_window).mean()
    squeeze = np.zeros(len(typical_price))
    BOLl_slope = ma_BOLl.diff()/Moving_window
    BOLu_slope = ma_BOLu.diff()/Moving_window
    for i in range (Moving_window, len(typical_price)):
        if (BOLl_slope[i] > 0 and BOLu_slope[i] < 0 and math.isclose(BOLl_slope[i] + BOLu_slope[i], 0,rel_tol=8e-1)):
            squeeze[i] = 1
            print('Squeeze detected at {}'.format(i))
        else:
            squeeze[i] = 0

    #calculate difference between today's typical price and moving avg
    diff = typical_price - ma
    ticker_data.insert(1,'Diff_bw_typ_and_ma', diff)
    ticker_data.insert(1,'Squeeze', squeeze)
    ticker_data.insert(1,'BOLu', BOLu)
    ticker_data.insert(1,'BOLl', BOLl)
    ticker_data.insert(1,'BOLu_s', np.abs(BOLu_slope))
    ticker_data.insert(1,'BOLl_s', np.abs(BOLl_slope))
    ticker_data.fillna(0,inplace = True) #Change the first row to 0 since we don't have data before the first data point
    return ticker_data