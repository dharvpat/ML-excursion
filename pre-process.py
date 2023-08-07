import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def RSI(data, window=14, adjust=False):
    delta = data['Close'].diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
    loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100 / (1 + RS)

    return RSI

ticker = 'AAPL'
ticker_data = yf.download(ticker, start='2021-09-01', end='2023-07-01',interval='1d')
ticker_data_close = ticker_data['Close'].to_numpy()
ticker_data_open = ticker_data['Open'].to_numpy()
ticker_data_average = (ticker_data_close + ticker_data_open)/2
ticker_data.insert(1, 'Average', ticker_data_average)

change = ticker_data['Close'].diff()
change.dropna(inplace = True)

change_up = change.copy()
change_down = change.copy()

change_up[change_up < 0] = 0
change_down[change_down > 0] = 0

change.equals(change_up+change_down)

avg_up = change_up.rolling(14).mean()
avg_down = change_down.rolling(14).mean().abs()

rsi = 100 * avg_up / (avg_up + avg_down)

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

# Print the MACD, Signal, and Signal above MACD values
ticker_data.insert(1,'MACD_Value', macd_line)
ticker_data.insert(1,'Signal_Value', signal_line)
ticker_data.insert(1,'Signal_MACD_strength',signal_above_macd)

Moving_window = 5
ma = ticker_data['Average'].rolling(window = Moving_window).mean()
ticker_data.insert(1, 'Moving Average', ma)
ma_slope = ticker_data['Moving Average'].diff()/Moving_window
tangent_inverse = np.arctan(ma_slope)
normalization_factor = np.pi/2
normalized_tangent_inverse = tangent_inverse / normalization_factor
ticker_data.insert(1, 'Normalized Moving Average Slope', normalized_tangent_inverse)
ticker_data.insert(1,'RSI', rsi)