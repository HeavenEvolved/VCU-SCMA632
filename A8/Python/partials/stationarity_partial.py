import streamlit as st

import modules.tests as tt

def get_partial():
    if "Log_Returns" in st.session_state.data.columns:
        stationarity_results = tt.test_stationarity(st.session_state.data["Log_Returns"])
        st.json(stationarity_results)
        if stationarity_results["Is Stationary"]:
            st.success("The log returns series is stationary.")
        else:
            st.warning("The log returns series is not stationary.")
    else:
        st.error("Log Returns data is not available for stationarity testing.")