import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import warnings

import modules.downloader as dl
import modules.tests as tt
import modules.univariate as uv
import modules.multivariate as mv

import partials.ticker_form as tk_form
import partials.stationarity_partial as st_partial
import partials.acf_pcf_partial as ap_partial

warnings.filterwarnings("ignore")

def plot_data(data: pd.DataFrame, ticker: str):
    fig = go.Figure()

    # Add Close price as first y-axis
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["Close"], name="Close Price", line=dict(color="blue")
        )
    )

    # Add Log Returns as second y-axis
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Log_Returns"],
            name="Log Returns",
            line=dict(color="grey"),
            yaxis="y2",
        )
    )

    # Set up layout with two y-axes
    fig.update_layout(
        title=f"{ticker} Close Price and Log Returns Over Time",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Close Price", side="left"),
        yaxis2=dict(title="Log Returns", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

def main():

    st.set_page_config(
        page_title="A8 Python Application",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    keys = ["data", "ticker", "uni_results"]

    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

    st.title("Welcome to the Stock Price Forecaster!")

    with st.form(key="ticker_form", clear_on_submit=True):
        tk_form.get_form()

    if st.session_state.data is not None and st.session_state.ticker is not None:
        st.header(f"Data for {st.session_state.ticker}")
        st.write("Here is the data fetched from Yahoo Finance:")

        st.dataframe(st.session_state.data.tail(), use_container_width=True)
        st.download_button("Download data as CSV", st.session_state.data.to_csv(), file_name=f"{st.session_state.ticker}_data.csv")

        with st.expander("Data Summary", expanded=False):
            st.write(st.session_state.data.describe())

        with st.expander("Plot Data", expanded=True):
            plot_data(st.session_state.data, st.session_state.ticker)
            
        with st.expander("Test for Stationarity (Augmented Dicky-Fuller)", expanded=True):
            st_partial.get_partial()

        with st.expander("Autocorrelation Analysis (ACF & PACF)", expanded=True):
            p, q = ap_partial.get_partial()
            
        with st.expander("ARCH Effect Test (Lagrange Multiplier)", expanded=False):
            p_value = tt.do_arch_test()
            if p_value:
                st.warning("Significant ARCH effect detected (variance not constant). GARCH modeling may be appropriate.")
            else:
                st.success("No significant ARCH effect detected (variance is constant).")
                
        if p_value:
            with st.expander("ARCH and GARCH Model Fit", expanded=False):
                tt.fit_garch_model(p, q)
                
        with st.expander("Univariate Model Comparison", expanded=True):
            uv.univariate_comparison(st.session_state.data["Close"])
            
        # with st.expander("Multivariate Model Comparison", expanded=True):
        #     mv.multivariate_comparison(st.session_state.data, "Close")
        
        with st.expander("Future Forecast", expanded=True):
            uv.forecast_future_with_best_model(st.session_state.data["Close"])
            
    else:
        st.warning("Please enter a valid ticker symbol to fetch data.")


if __name__ == "__main__":
    main()
