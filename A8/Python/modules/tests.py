import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model

import plotly.graph_objects as go


def test_stationarity(series: pd.Series):
    result = adfuller(series.dropna())
    return {
        "Test Statistic": result[0],
        "p-value": result[1],
        "Used Lag": result[2],
        "Number of Observations": result[3],
        "Critical Values": result[4],  # type: ignore
        "Is Stationary": result[1] < 0.05,
    }


def do_acf_pacf():
    lags = 20
    acf_vals = acf(st.session_state.data["Log_Returns"], nlags=lags)
    pacf_vals = pacf(st.session_state.data["Log_Returns"], nlags=lags)

    return (acf_vals, pacf_vals)


def plot_acf_pacf(acf_vals, pacf_vals):
    x = list(range(len(acf_vals)))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=acf_vals, name="ACF"))
    fig.add_trace(go.Bar(x=x, y=pacf_vals, name="PACF"))
    fig.update_layout(
        title="ACF and PACF",
        barmode="group",
        xaxis_title="Lag",
        yaxis_title="Correlation",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


def get_p_q_suggestions(acf_vals, pacf_vals):
    n = len(st.session_state.data["Log_Returns"].dropna())
    conf_level = 1.96 / np.sqrt(n)

    p_suggest = np.argmax(np.abs(pacf_vals[1:]) < conf_level) + 1
    q_suggest = np.argmax(np.abs(acf_vals[1:]) < conf_level) + 1

    return (p_suggest, q_suggest)


def do_arch_test():
    test_stat, p_value, _, _ = het_arch(st.session_state.data["Log_Returns"].dropna(), nlags=10)  # type: ignore
    st.write(f"Test Statistic: {test_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    return p_value < 0.05


def parse_arch_garch_results(fit):
    params = fit.params
    pvalues = fit.pvalues
    aic = fit.aic
    bic = fit.bic
    loglikelihood = fit.loglikelihood

    # Build a clean dict or DataFrame for display
    results_dict = {"Log-Likelihood": loglikelihood, "AIC": aic, "BIC": bic}
    coef_df = pd.DataFrame({"Coefficient": params, "p-value": pvalues.apply(lambda x: f"{x:.4e}")})

    return results_dict, coef_df


def fit_garch_model(p, q):

    returns = (
        st.session_state.data["Log_Returns"] * 100
    )
    arch_mod = arch_model(returns, vol="ARCH", p=p)
    arch_fit = arch_mod.fit(disp="off")

    garch_mod = arch_model(returns, vol="GARCH", p=p, q=q)
    garch_fit = garch_mod.fit(disp="off")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=st.session_state.data.index,
            y=arch_fit.conditional_volatility,
            name=f"ARCH({p}) Conditional Volatility",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=st.session_state.data.index,
            y=garch_fit.conditional_volatility,
            name=f"GARCH({p},{q}) Conditional Volatility",
        )
    )
    fig.update_layout(
        legend=dict(bgcolor='rgba(0, 0, 0, 0)'),
        title="ARCH and GARCH Model - Conditional Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    arch_stats, arch_coefs = parse_arch_garch_results(arch_fit)
    garch_stats, garch_coefs = parse_arch_garch_results(garch_fit)

    st.write(f"### ARCH({p}) Model Summary")
    st.json(arch_stats)
    st.write("Significance of coefficients:")
    st.dataframe(arch_coefs)

    st.write(f"### GARCH({p},{q}) Model Summary")
    st.json(garch_stats)
    st.write("Significance of coefficients:")
    st.dataframe(garch_coefs)

    st.write("### Model Comparison Metrics")
    comparison_df = pd.DataFrame(
        {
            "Model": [f"ARCH({p})", f"GARCH({p},{q})"],
            "AIC": [arch_stats["AIC"], garch_stats["AIC"]],
            "BIC": [arch_stats["BIC"], garch_stats["BIC"]],
        }
    )
    st.dataframe(comparison_df.set_index("Model"))

    if (garch_stats["AIC"] < arch_stats["AIC"]) and (
        garch_stats["BIC"] < arch_stats["BIC"]
    ):
        best_model = f"GARCH({p},{q})"
    else:
        best_model = f"ARCH({p})"
        
    if best_model == f"GARCH({p},{q})":
        st.success(f"GARCH({p},{q}) is the preferred model for this dataset (lower AIC/BIC).")
    else:
        st.success(f"ARCH({p}) is the preferred model for this dataset (lower AIC/BIC).")