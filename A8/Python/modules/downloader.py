import yfinance as yf
import pandas as pd
import streamlit as st
import re
    
@st.cache_data(ttl=300, show_spinner=False)
def get_ticker_data( ticker) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, period="max", interval="1d", multi_level_index=False)
        if data is None or data.empty:
            data = None
            return None
        data.set_index(
            pd.Series(data.index).apply(lambda x: x.strftime("%Y-%m-%d")), inplace=True
        )
        data.index = pd.to_datetime(data.index)
        data["Log_Returns"] = data["Close"].pct_change()
        data = data.dropna()
        return data
    except Exception as e:
        return None
    
def is_valid_ticker(ticker: str) -> bool:
    TICKER_REGEX = r"^[A-Z0-9]+([.-][A-Z0-9]+)?$"
    ticker = ticker.strip().upper()
    return re.fullmatch(TICKER_REGEX, ticker) is not None