import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Download MSFT stock data
msft = yf.Ticker("MSFT")

# Get historical market data
hist = msft.history(period="1d", start='2021-01-01', end='2023-12-31')

# Display the dataframe as a table
st.dataframe(hist)

# Display an area chart for the 'Close' price
st.area_chart(hist['Close'])
