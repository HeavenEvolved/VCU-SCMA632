import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
    st.write("### Lagged Features DataFrame:", df.head())
    return df

def compute_metrics(true, pred):
    st.write("### compute_metrics() called")
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    st.write(f"metrics: len(true)={len(true)}, len(pred)={len(pred)}")
    st.write(f"metrics: true[:5]={true[:5]}, pred[:5]={pred[:5]}")
    if len(true) != len(pred) or len(true) == 0:
        st.warning("metrics: true and pred have different lengths or zero length")
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    if np.all(np.isnan(true)) or np.all(np.isnan(pred)):
        st.warning("metrics: all true or pred are NaN!")
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    mae = np.nanmean(np.abs(true - pred))
    rmse = np.sqrt(np.nanmean((true - pred) ** 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_arr = np.abs((true - pred) / true)
        mape_arr = mape_arr[np.isfinite(mape_arr)]
        mape = np.mean(mape_arr) * 100 if mape_arr.size > 0 else np.nan
    st.write(f"metrics: MAE={mae}, RMSE={rmse}, MAPE={mape}")
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}

def forecast_ar(train, test_len, lags=5):
    st.write(f"### forecast_ar: train.size={len(train)}, lags={lags}")
    model = AutoReg(train, lags=lags).fit()
    pred = model.predict(start=len(train), end=len(train)+test_len-1, dynamic=False)
    st.write("### AR pred[:5]:", pred.head())
    return pred

def forecast_ma(train, test_len, q=2):
    st.write(f"### forecast_ma: train.size={len(train)}, q={q}")
    model = ARIMA(train, order=(0,0,q)).fit()
    pred = model.forecast(steps=test_len)
    st.write("### MA pred[:5]:", pred.head())
    return pred

def forecast_arma(train, test_len, p=2, q=2):
    st.write(f"### forecast_arma: train.size={len(train)}, p={p}, q={q}")
    model = ARIMA(train, order=(p,0,q)).fit()
    pred = model.forecast(steps=test_len)
    st.write("### ARMA pred[:5]:", pred.head())
    return pred

def forecast_arima(train, test_len, p=2, d=1, q=2):
    st.write(f"### forecast_arima: train.size={len(train)}, p={p}, d={d}, q={q}")
    model = ARIMA(train, order=(p,d,q)).fit()
    pred = model.forecast(steps=test_len)
    st.write("### ARIMA pred[:5]:", pred.head())
    return pred

def forecast_sarima(train, test_len, p=1, d=1, q=1, s=12):
    st.write(f"### forecast_sarima: train.size={len(train)}, p={p}, d={d}, q={q}, s={s}")
    model = SARIMAX(train, order=(p,d,q), seasonal_order=(1,1,1,s)).fit(disp=False)
    pred = model.forecast(steps=test_len)
    st.write("### SARIMA pred[:5]:", pred.head())
    return pred

def forecast_lr(train, test_len, n_lags=5):
    st.write(f"### forecast_lr: train.size={len(train)}, n_lags={n_lags}")
    df = create_lagged_features(train, n_lags)
    X, y = df.drop("y", axis=1).values, df["y"].to_numpy()
    model = LinearRegression().fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for i in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq='B')[1:]
    series = pd.Series(preds, index=idx)
    st.write("### LR pred[:5]:", series.head())
    return series

def forecast_rf(train, test_len, n_lags=5):
    st.write(f"### forecast_rf: train.size={len(train)}, n_lags={n_lags}")
    df = create_lagged_features(train, n_lags)
    X, y = df.drop("y", axis=1).values, df["y"].values
    model = RandomForestRegressor().fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for i in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq='B')[1:]
    series = pd.Series(preds, index=idx)
    st.write("### RF pred[:5]:", series.head())
    return series

def forecast_xgb(train, test_len, n_lags=5):
    st.write(f"### forecast_xgb: train.size={len(train)}, n_lags={n_lags}")
    df = create_lagged_features(train, n_lags)
    X, y = df.drop("y", axis=1).values, df["y"].values
    model = xgb.XGBRegressor(objective='reg:squarederror').fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for i in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq='B')[1:]
    series = pd.Series(preds, index=idx)
    st.write("### XGB pred[:5]:", series.head())
    return series

def forecast_lgbm(train, test_len, n_lags=5):
    st.write(f"### forecast_lgbm: train.size={len(train)}, n_lags={n_lags}")
    df = create_lagged_features(train, n_lags)
    X, y = df.drop("y", axis=1).values, df["y"].values
    model = lgb.LGBMRegressor().fit(X, y)
    last = train.values[-n_lags:].tolist()
    preds = []
    for i in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        preds.append(y_pred)
        last.append(y_pred)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq='B')[1:]
    series = pd.Series(preds, index=idx)
    st.write("### LGBM pred[:5]:", series.head())
    return series

def forecast_lstm(train, test_len, n_lags=5, epochs=20):
    st.write(f"### forecast_lstm: train.size={len(train)}, n_lags={n_lags}, epochs={epochs}")
    scaler = lambda x: (x - np.mean(x)) / np.std(x)
    train_scaled = scaler(train.values)
    X, y = [], []
    for i in range(n_lags, len(train_scaled)):
        X.append(train_scaled[i-n_lags:i])
        y.append(train_scaled[i])
    X, y = np.array(X), np.array(y)
    X = X[..., np.newaxis]
    model = Sequential([
        LSTM(32, input_shape=(n_lags,1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0,
              callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])
    last = list(train_scaled[-n_lags:])
    preds = []
    for i in range(test_len):
        x_pred = np.array(last[-n_lags:]).reshape(1, n_lags, 1)
        y_pred = model.predict(x_pred, verbose=0)[0, 0]
        preds.append(y_pred)
        last.append(y_pred)
    preds = np.array(preds) * np.std(train.values) + np.mean(train.values)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq='B')[1:]
    series = pd.Series(preds, index=idx)
    st.write("### LSTM pred[:5]:", series.head())
    return series

def univariate_comparison(data_close: pd.Series):
    st.subheader("Univariate Model Comparison (with Diagnostics)")
    test_len = st.number_input("Test window size (days for backtesting):", 10, 60, 30)
    n_lags = st.number_input("Number of lags for ML models (lags):", 3, 20, 5)
    models = st.multiselect(
        "Select models:",
        ["AR", "MA", "ARMA", "ARIMA", "SARIMA",
         "Linear Regression", "Random Forest", "XGBoost", "LGBM", "LSTM"],
        default=["AR", "ARIMA", "Linear Regression", "Random Forest", "XGBoost"]
    )
    run_button = st.button("Run Model Comparison")
    if not run_button:
        return

    with st.spinner("Running univariate models... (with debugging)"):
        if not models:
            st.warning("Please select at least one model to run.")
            return

        if len(data_close) < test_len + max(n_lags, 10):
            st.warning(f"Not enough data points ({len(data_close)}) for the test window ({test_len}) and lags ({n_lags}).")
            return

        train = data_close[:-test_len]
        test = data_close[-test_len:]
        st.write("### Raw Train Index/Head:", train.index[:5], train.head())
        st.write("### Raw Test Index/Head:", test.index[:5], test.head())
        results = []
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual", line=dict(color="gray", width=3)))

        for model_name in models:
            try:
                st.write(f"## Running {model_name}...")
                if model_name == "AR":
                    pred = forecast_ar(train, test_len, lags=n_lags)
                elif model_name == "MA":
                    pred = forecast_ma(train, test_len, q=n_lags)
                elif model_name == "ARMA":
                    pred = forecast_arma(train, test_len, p=n_lags, q=n_lags)
                elif model_name == "ARIMA":
                    pred = forecast_arima(train, test_len, p=n_lags, d=1, q=n_lags)
                elif model_name == "SARIMA":
                    pred = forecast_sarima(train, test_len, p=1, d=1, q=1, s=12)
                elif model_name == "Linear Regression":
                    pred = forecast_lr(train, test_len, n_lags=n_lags)
                elif model_name == "Random Forest":
                    pred = forecast_rf(train, test_len, n_lags=n_lags)
                elif model_name == "XGBoost":
                    pred = forecast_xgb(train, test_len, n_lags=n_lags)
                elif model_name == "LGBM":
                    pred = forecast_lgbm(train, test_len, n_lags=n_lags)
                elif model_name == "LSTM":
                    with st.spinner("Training LSTM (this may take a minute)"):
                        pred = forecast_lstm(train, test_len, n_lags=n_lags, epochs=20)
                else:
                    continue

                st.write(f"## {model_name} post-pred index head:", pred.index[:5])
                st.write(f"## {model_name} pred head:", pred.head())
                st.write(f"## {model_name} test index head:", test.index[:5])
                st.write(f"## {model_name} test head:", test.head())
                st.write(f"## {model_name} isnull().sum(): {pred.isnull().sum()}")

                # Align predictions index with test
                pred = pred.reindex(test.index).bfill().astype(float)
                st.write(f"## {model_name} reindexed pred head:", pred.head())
                if pred.isnull().all():
                    st.warning(f"All predictions are NaN for {model_name} after reindex!")
                    continue
                metrics = compute_metrics(test.values, pred.values)
                results.append((model_name, pred, metrics))
                fig.add_trace(go.Scatter(x=test.index, y=pred.values, name=model_name))
            except Exception as e:
                st.warning(f"{model_name} failed: {e}")

        fig.update_layout(title="Forecast Comparison (Test Window, debug)", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        metrics_rows = [
            {"Model": model, **metrics}
            for (model, _, metrics) in results
            if metrics is not None and all(isinstance(metrics.get(k), (float, int, np.floating)) and not pd.isnull(metrics[k]) for k in ["MAE", "RMSE", "MAPE"])
        ]
        st.write("### Compiled metrics rows: ", metrics_rows)
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
            st.subheader("Model Accuracy (Lower is Better)")
            st.dataframe(metrics_df.style.highlight_min(axis=0))
        else:
            st.warning("No valid accuracy metrics computed for selected models.")

