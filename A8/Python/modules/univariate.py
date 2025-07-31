import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error

import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def create_lagged_features(series, n_lags):
    df = pd.DataFrame({'y': series})
    for lag in range(1, n_lags + 1):
        df[f'lag{lag}'] = df['y'].shift(lag)
    df = df.dropna()
    if df.empty:
        raise ValueError("Lagged feature DataFrame is empty. Increase data length or reduce lags.")
    return df


def compute_mase(true, pred):
    true = np.array(true, dtype=float)
    pred = np.array(pred, dtype=float)
    n = len(true)
    if n < 2:
        return np.nan
    scale = np.mean(np.abs(true[1:] - true[:-1]))
    if scale == 0:
        return np.nan
    mae = np.mean(np.abs(true - pred))
    return mae / scale


def compute_metrics(true, pred):
    true = np.array(true, dtype=float)
    pred = np.array(pred, dtype=float)
    if len(true) != len(pred) or len(true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "MedAE": np.nan, "MASE": np.nan}
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    medae = median_absolute_error(true, pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_arr = np.abs((true - pred) / true)
        mape_arr = mape_arr[np.isfinite(mape_arr)]
        mape = np.mean(mape_arr) * 100 if mape_arr.size > 0 else np.nan
    mase = compute_mase(true, pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "MedAE": float(medae),
        "MASE": float(mase),
    }


def forecast_ar(train, test_len, lags=5, test_index=None):
    model = AutoReg(train, lags=lags).fit()
    pred = model.predict(start=len(train), end=len(train) + test_len - 1, dynamic=False)
    if test_index is not None:
        pred.index = test_index
    else:
        pred.index = pd.date_range(
            start=train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    return pred


def forecast_ma(train, test_len, q=2, test_index=None):
    model = ARIMA(train, order=(0, 0, q)).fit()
    pred = model.forecast(steps=test_len)
    if test_index is not None:
        pred.index = test_index
    else:
        pred.index = pd.date_range(
            start=train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    return pred


def forecast_arma(train, test_len, p=2, q=2, test_index=None):
    model = ARIMA(train, order=(p, 0, q)).fit()
    pred = model.forecast(steps=test_len)
    if test_index is not None:
        pred.index = test_index
    else:
        pred.index = pd.date_range(
            start=train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    return pred


def forecast_arima(train, test_len, p=2, d=1, q=2, test_index=None):
    model = ARIMA(train, order=(p, d, q)).fit()
    pred = model.forecast(steps=test_len)
    if test_index is not None:
        pred.index = test_index
    else:
        pred.index = pd.date_range(
            start=train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    return pred


def forecast_sarima(train, test_len, p=1, d=1, q=1, s=5, test_index=None):
    seasonal_order = (1, 1, 1, s) if s > 1 else (0, 0, 0, 0)
    model = SARIMAX(train, order=(p, d, q), seasonal_order=seasonal_order).fit(disp=False)
    pred = model.forecast(steps=test_len)
    if test_index is not None:
        pred.index = test_index
    else:
        pred.index = pd.date_range(
            start=train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    return pred


def forecast_lr(train, test_len, n_lags=5, test_index=None):
    df = create_lagged_features(train, n_lags)
    X = df.drop("y", axis=1).values
    y = df["y"].to_numpy()
    model = LinearRegression().fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for _ in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = (
        test_index
        if test_index is not None
        else pd.date_range(
            train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    )
    return pd.Series(preds, index=idx)


def forecast_rf(train, test_len, n_lags=5, test_index=None):
    df = create_lagged_features(train, n_lags)
    X = df.drop("y", axis=1).values
    y = df["y"].to_numpy()
    model = RandomForestRegressor().fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for _ in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = (
        test_index
        if test_index is not None
        else pd.date_range(
            train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    )
    return pd.Series(preds, index=idx)


def forecast_xgb(train, test_len, n_lags=5, test_index=None):
    df = create_lagged_features(train, n_lags)
    X = df.drop("y", axis=1).values
    y = df["y"].values
    model = xgb.XGBRegressor(objective="reg:squarederror").fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for _ in range(test_len):
        x_pred = np.array(last[-n_lags :]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = (
        test_index
        if test_index is not None
        else pd.date_range(
            train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    )
    return pd.Series(preds, index=idx)


def forecast_lgbm(train, test_len, n_lags=5, test_index=None):
    df = create_lagged_features(train, n_lags)
    X = df.drop("y", axis=1).values
    y = df["y"].values
    model = lgb.LGBMRegressor().fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for _ in range(test_len):
        x_pred = np.array(last[-n_lags :]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = (
        test_index
        if test_index is not None
        else pd.date_range(
            train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    )
    return pd.Series(preds, index=idx)


def forecast_lstm(train, test_len, n_lags=5, epochs=20, test_index=None):
    scaler = lambda x: (x - np.mean(x)) / np.std(x)
    train_scaled = scaler(train.values)
    X, y = [], []
    for i in range(n_lags, len(train_scaled)):
        X.append(train_scaled[i - n_lags : i])
        y.append(train_scaled[i])
    X, y = np.array(X), np.array(y)
    X = X[..., np.newaxis]
    model = Sequential(
        [
            LSTM(32, input_shape=(n_lags, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
    model.fit(
        X,
        y,
        epochs=epochs,
        verbose=0,
        callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)],
    )
    last = list(train_scaled[-n_lags:])
    preds = []
    for _ in range(test_len):
        x_pred = np.array(last[-n_lags :]).reshape(1, n_lags, 1)
        y_pred = model.predict(x_pred, verbose=0)[0, 0]
        preds.append(y_pred)
        last.append(y_pred)
    preds = np.array(preds) * np.std(train.values) + np.mean(train.values)
    idx = (
        test_index
        if test_index is not None
        else pd.date_range(
            train.index[-1],
            periods=test_len + 1,
            freq=train.index.freq or pd.infer_freq(train.index),
        )[1:]
    )
    return pd.Series(preds, index=idx)


def univariate_comparison(data_close: pd.Series):
    st.subheader("Univariate Model Comparison")
    test_len = st.number_input(
        "Test window size (days for backtesting):", 10, 60, 30
    )
    n_lags = st.number_input(
        "Number of lags for ML models (lags):", 3, 20, 5
    )
    models = st.multiselect(
        "Select models:",
        [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
            "SARIMA",
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "LGBM",
            "LSTM",
        ],
        default=["AR", "ARIMA", "Linear Regression", "Random Forest", "XGBoost"],
    )
    metric_options = ["RMSE", "MAE", "MAPE", "MedAE", "MASE"]

    run_button = st.button("Run Model Comparison")

    # Run models once or if user explicitly reruns
    if run_button or "uni_results" not in st.session_state:
        if not models:
            st.warning("Please select at least one model to run.")
            return

        if len(data_close) < test_len + max(n_lags, 10):
            st.warning(
                f"Not enough data points ({len(data_close)}) for the test window ({test_len}) and lags ({n_lags})."
            )
            return

        with st.spinner("Running selected models..."):
            train = data_close[:-test_len]
            test = data_close[-test_len:]

            results = []
            for model_name in models:
                try:
                    if model_name == "AR":
                        pred = forecast_ar(
                            train, test_len, lags=n_lags, test_index=test.index
                        )
                    elif model_name == "MA":
                        pred = forecast_ma(
                            train, test_len, q=n_lags, test_index=test.index
                        )
                    elif model_name == "ARMA":
                        pred = forecast_arma(
                            train, test_len, p=n_lags, q=n_lags, test_index=test.index
                        )
                    elif model_name == "ARIMA":
                        pred = forecast_arima(
                            train, test_len, p=n_lags, d=1, q=n_lags, test_index=test.index
                        )
                    elif model_name == "SARIMA":
                        pred = forecast_sarima(
                            train,
                            test_len,
                            p=1,
                            d=1,
                            q=1,
                            s=5,
                            test_index=test.index,
                        )  # weekly seasonality
                    elif model_name == "Linear Regression":
                        pred = forecast_lr(
                            train, test_len, n_lags=n_lags, test_index=test.index
                        )
                    elif model_name == "Random Forest":
                        pred = forecast_rf(
                            train, test_len, n_lags=n_lags, test_index=test.index
                        )
                    elif model_name == "XGBoost":
                        pred = forecast_xgb(
                            train, test_len, n_lags=n_lags, test_index=test.index
                        )
                    elif model_name == "LGBM":
                        pred = forecast_lgbm(
                            train, test_len, n_lags=n_lags, test_index=test.index
                        )
                    elif model_name == "LSTM":
                        with st.spinner("Training LSTM (this may take some time)..."):
                            pred = forecast_lstm(
                                train, test_len, n_lags=n_lags, epochs=20, test_index=test.index
                            )
                    else:
                        continue

                    if pred.isnull().all():
                        st.warning(f"All predictions are NaN for {model_name}. Skipping.")
                        continue

                    metrics = compute_metrics(test.values, pred.values)
                    if any(np.isnan(list(metrics.values()))):
                        st.warning(
                            f"{model_name} produced NaN in one or more accuracy metrics. Skipping."
                        )
                        continue

                    results.append({"Model": model_name, "Prediction": pred, "Metrics": metrics})

                except Exception as e:
                    st.warning(f"Model {model_name} failed: {e}")

            st.session_state.uni_results = results

    # If no results yet, prompt
    if "uni_results" not in st.session_state or not st.session_state.uni_results:
        st.info("Run the models to see the results here.")
        return

    # Metric selection box (does not rerun models)
    metric_selected = st.selectbox("Sort ranking by metric:", metric_options, index=0)

    # Create a DataFrame of metrics and sort by selected metric
    metrics_df = pd.DataFrame([{"Model": r["Model"], **r["Metrics"]} for r in st.session_state.uni_results])
    if metric_selected in metrics_df.columns:
        metrics_df = metrics_df.sort_values(metric_selected)

    st.subheader(f"Model Accuracy Rankings (sorted by {metric_selected})")
    st.dataframe(metrics_df.style.highlight_min(axis=0, subset=metric_options), use_container_width=True)

    # Plot actual vs all predictions
    fig = go.Figure()
    test = data_close[-len(metrics_df) - 30 :] if len(data_close) > 30 else data_close[-len(metrics_df):]
    # Always plot last test_len days actuals (same as all prediction indices)
    test = data_close[-len(st.session_state.uni_results[0]["Prediction"]):]
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual", mode="lines+markers", marker=dict(symbol="circle", size=10, color="gray"), line=dict(color="gray", width=3)))
    for r in st.session_state.uni_results:
        if r["Model"] in metrics_df["Model"].values:
            fig.add_trace(go.Scatter(x=r["Prediction"].index, y=r["Prediction"].values, name=r["Model"]))
    fig.update_layout(title="Forecast Comparison (Test Window)", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Warnings for models performing much worse than best based on selected metric
    best_metric_val = metrics_df.iloc[0][metric_selected] if metric_selected in metrics_df.columns else None
    for _, row in metrics_df.iterrows():
        if best_metric_val and row[metric_selected] > 1.5 * best_metric_val:
            st.warning(
                f"{row['Model']} did not work too well on this dataset ({metric_selected} much higher than best)."
            )

def forecast_future_with_best_model(data_close: pd.Series):
    st.subheader("Future Forecast with Best Model")

    if "uni_results" not in st.session_state or not st.session_state.uni_results:
        st.info("Please run the model comparison above first to enable future forecasting.")
        return

    metric_options = ["RMSE", "MAE", "MAPE", "MedAE", "MASE"]
    metric_selected = st.selectbox("Select accuracy metric to pick the best model:", metric_options, index=0)
    future_days = st.number_input("Number of days to forecast into the future:", min_value=1, max_value=180, value=30)

    # Find best model by selected metric
    results = st.session_state.uni_results
    # Filter out invalid metrics or models with NaNs
    filtered_results = [
        r for r in results
        if r["Metrics"] is not None and not any(np.isnan(list(r["Metrics"].values())))
    ]
    if not filtered_results:
        st.warning("No valid model results found to make future forecasts.")
        return

    # Sort filtered by metric
    filtered_results.sort(key=lambda r: r["Metrics"].get(metric_selected, np.inf))
    best_model = filtered_results[0]

    best_model_name = best_model["Model"]
    st.write(f"Best model based on {metric_selected}: **{best_model_name}**")

    # Forecast on full data (no test/train split here)
    full_series = data_close

    # Map model name to forecast function
    model_func_map = {
        "AR": forecast_ar,
        "MA": forecast_ma,
        "ARMA": forecast_arma,
        "ARIMA": forecast_arima,
        "SARIMA": forecast_sarima,
        "Linear Regression": forecast_lr,
        "Random Forest": forecast_rf,
        "XGBoost": forecast_xgb,
        "LGBM": forecast_lgbm,
        "LSTM": forecast_lstm,
    }

    forecast_func = model_func_map.get(best_model_name)
    if not forecast_func:
        st.error(f"Forecast function for model '{best_model_name}' not implemented.")
        return

    # Run forecast on full series
    with st.spinner(f"Generating {future_days}-day forecast using {best_model_name}..."):
        try:
            if best_model_name == "SARIMA":
                # Optionally keep s=5 for weekly seasonality for SARIMA
                forecast_values = forecast_func(full_series, future_days, p=1, d=1, q=1, s=5)
            elif best_model_name == "LSTM":
                forecast_values = forecast_func(full_series, future_days, n_lags=st.session_state.get("n_lags", 5), epochs=20)
            else:
                # For other models, pass default lags or get from session
                n_lags = st.session_state.get("n_lags", 5)
                # Some classical models expect more params, but using defaults here for simplicity
                forecast_values = forecast_func(full_series, future_days, lags=n_lags) if best_model_name == "AR" else \
                                  forecast_func(full_series, future_days, test_index=None, n_lags=n_lags)
        except Exception as e:
            st.error(f"Failed to compute forecast: {e}")
            return

    # Plot historical + forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full_series.index,
        y=full_series.values,
        name="Historical",
        line=dict(color="gray", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_values.index,
        y=forecast_values.values,
        name=f"Forecast ({best_model_name})",
        line=dict(color="royalblue", dash="dash")
    ))
    fig.update_layout(
        title=f"{future_days}-Day Forecast Using {best_model_name}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(x=0, y=1)
    )
    
    # User controls for zoom
    zoom_mode = st.radio(
        "Graph view",
        options=["Zoom In (Forecast)", "Zoom Out (Full Series)"],
        index=0
    )

    recent_n_obs = 90  # Last 90 days, can be parameterized

    if zoom_mode == "Zoom In (Forecast)":
        if len(full_series) > recent_n_obs:
            zoom_start = full_series.index[-recent_n_obs]
        else:
            zoom_start = full_series.index[0]
        zoom_end = forecast_values.index[-1]
        fig.update_xaxes(range=[zoom_start, zoom_end])
    else:
        # Remove any custom zoom; show all data
        fig.update_xaxes(range=[full_series.index[0], forecast_values.index[-1]])

    st.plotly_chart(fig, use_container_width=True)

