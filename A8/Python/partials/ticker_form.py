import streamlit as st

import modules.downloader as dl

def get_form():
    ticker = (
            st.text_input(
                "Enter one stock ticker (e.g., TSLA):",
                placeholder="TSLA, NVDA, MSFT, etc.",
                key="ticker_name",
            )
            .strip()
            .upper()
        )
    st.markdown("Don't know a ticker? Visit [Yahoo Finance](https://finance.yahoo.com/) to find and copy the exact ticker symbol.")
    submit_button = st.form_submit_button(label="Get Data")
    if submit_button:
        if dl.is_valid_ticker(ticker) and ticker:
            data = dl.get_ticker_data(ticker)
            if data is not None and ticker is not None:
                st.session_state.data = data
                st.session_state.ticker = ticker
                del data
                del ticker
            else:
                st.error(f"Error fetching data for {ticker}")
                st.session_state.data = None
                st.session_state.ticker = None
        else:
            st.error("Please enter one and only one valid ticker symbol.")
            st.session_state.data = None
            st.session_state.ticker = None