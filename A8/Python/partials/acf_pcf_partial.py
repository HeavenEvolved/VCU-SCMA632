import streamlit as st

import modules.tests as tt

def get_partial() -> tuple:
    acf_vals, pacf_vals = tt.do_acf_pacf()
    tt.plot_acf_pacf(acf_vals, pacf_vals)
    p, q = tt.get_p_q_suggestions(acf_vals, pacf_vals)
    
    st.write(f"Suggested AR (p): {p}")
    st.write(f"Suggested MA (q): {q}")

    p = st.number_input("Select AR order (p)", min_value=1, max_value=10, value=p) # type: ignore
    q = st.number_input("Select MA order (q)", min_value=1, max_value=10, value=q) # type: ignore
    
    return (p, q)