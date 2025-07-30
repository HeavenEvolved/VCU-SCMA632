import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def create_multivariate_lagged_features(df, n_lags):
    """Create lagged features for all columns, drops NA."""
    lagged = df.copy()
    for lag in range(1, n_lags+1):
        lagged = pd.concat([lagged, df.shift(lag).add_suffix(f"_lag{lag}")], axis=1)
    lagged = lagged.dropna()
    return lagged

def compute_metrics(true, pred):
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def forecast_var(train, test_len, maxlags=5):
    model = VAR(train)
    order = model.select_order(maxlags=maxlags).selected_orders.get('aic', maxlags)
    model_fit = model.fit(order)
    forecast_vals = model_fit.forecast(train.values[-order:], steps=test_len)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq=train.index.freq or pd.infer_freq(train.index))[1:]
    forecast_df = pd.DataFrame(forecast_vals, index=idx, columns=train.columns)
    return forecast_df

def forecast_vecm(train, test_len, k_ar_diff=1, coint_rank=1):
    model = VECM(train, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic='ci')
    model_fit = model.fit()
    forecast_vals = model_fit.predict(steps=test_len)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq=train.index.freq or pd.infer_freq(train.index))[1:]
    forecast_df = pd.DataFrame(forecast_vals, index=idx, columns=train.columns)
    return forecast_df

def forecast_sarimax(train_y, exog, test_len, order=(1,0,1), seasonal_order=(0,0,0,0)):
    model = SARIMAX(train_y, exog=exog.loc[train_y.index], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    pred = model_fit.forecast(steps=test_len, exog=exog.iloc[-test_len:])
    return pred

def forecast_ml_model(train, target_col, test_len, n_lags, model_name):
    df = create_multivariate_lagged_features(train, n_lags)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    if model_name == "Random Forest":
        model = RandomForestRegressor()
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    elif model_name == "LGBM":
        model = lgb.LGBMRegressor(n_estimators=100)
    else:
        raise ValueError("Unknown model for ML forecasting.")

    model.fit(X, y)
    last_values = train.values[-n_lags:].tolist()
    preds = []

    for _ in range(test_len):
        feats = []
        for lag in range(1, n_lags+1):
            feats.extend(last_values[-lag])
        x_pred = np.array(feats).reshape(1, -1)
        y_pred = model.predict(x_pred)[0]
        new_row = list(last_values[-1])
        target_idx = train.columns.get_loc(target_col)
        new_row[target_idx] = y_pred
        preds.append(y_pred)
        last_values.append(new_row)

    idx = pd.date_range(train.index[-1], periods=test_len+1, freq=train.index.freq or pd.infer_freq(train.index))[1:]
    return pd.Series(preds, index=idx)

def forecast_lstm_multivariate(train, test_len, n_lags=5, epochs=20):
    scaler = lambda x: (x - np.mean(x)) / np.std(x)
    train_scaled = train.apply(scaler)
    X, y = [], []
    cols = train.columns.tolist()
    for i in range(n_lags, len(train_scaled)):
        X.append(train_scaled.iloc[i-n_lags:i].values)
        y.append(train_scaled.iloc[i].values)
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(32, input_shape=(n_lags, len(cols))),
        Dense(len(cols))
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0, callbacks=[EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)])

    last_seq = train_scaled.iloc[-n_lags:].values.tolist()
    preds = []
    for _ in range(test_len):
        x_pred = np.array(last_seq[-n_lags:]).reshape(1, n_lags, len(cols))
        y_pred = model.predict(x_pred, verbose=0)[0]
        preds.append(y_pred)
        last_seq.append(y_pred)

    preds = np.array(preds)
    unscaled_preds = []
    for i, col in enumerate(cols):
        col_mean = train[col].mean()
        col_std = train[col].std()
        unscaled_preds.append(preds[:, i] * col_std + col_mean)
    idx = pd.date_range(train.index[-1], periods=test_len+1, freq=train.index.freq or pd.infer_freq(train.index))[1:]
    preds_df = pd.DataFrame(np.stack(unscaled_preds, axis=1), index=idx, columns=cols)
    return preds_df

def multivariate_comparison(data, target_col):
    st.subheader("Multivariate Model Comparison")
    test_len = st.number_input("Test window size (days for backtesting):", 10, 60, 30, key="test_len")
    n_lags = st.number_input("Number of lags for ML models (lags):", 3, 20, 5, key="n_lags")
    models = st.multiselect(
        "Select models:",
        ["VAR", "VECM", "SARIMAX", "Random Forest", "XGBoost", "LGBM", "LSTM"],
        default=["VAR", "SARIMAX", "Random Forest", "XGBoost"]
    )
    run_button = st.button("Run Multivariate Model Comparison")

    if not run_button:
        return

    with st.spinner("Running multivariate models..."):
        if not models:
            st.warning("Please select at least one model to run.")
            return
        train = data[:-test_len]
        test = data[-test_len:]
        results = []

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test[target_col].values, name="Actual", line=dict(color="gray", width=3)))

        if "VAR" in models:
            pred_df = forecast_var(train, test_len, maxlags=n_lags)
            pred = pred_df[target_col]
            pred = pred.reindex(test.index).bfill()
            results.append(("VAR", pred, compute_metrics(test[target_col].values, pred.values)))
            fig.add_trace(go.Scatter(x=test.index, y=pred, name="VAR"))

        if "VECM" in models:
            try:
                pred_df = forecast_vecm(train, test_len, k_ar_diff=n_lags, coint_rank=1)
                pred = pred_df[target_col]
                pred = pred.reindex(test.index).bfill()
                results.append(("VECM", pred, compute_metrics(test[target_col].values, pred.values)))
                fig.add_trace(go.Scatter(x=test.index, y=pred, name="VECM"))
            except Exception as e:
                results.append(("VECM", None, {"Error": str(e)}))

        if "SARIMAX" in models:
            exog_train = train.drop(columns=[target_col])
            exog_test = test.drop(columns=[target_col])
            pred = forecast_sarimax(train[target_col], pd.concat([exog_train, exog_test]), test_len)
            pred = pred.reindex(test.index).bfill()
            results.append(("SARIMAX", pred, compute_metrics(test[target_col].values, pred.values)))
            fig.add_trace(go.Scatter(x=test.index, y=pred, name="SARIMAX"))

        for ml_model in ["Random Forest", "XGBoost", "LGBM"]:
            if ml_model in models:
                pred = forecast_ml_model(train, target_col, test_len, n_lags, ml_model)
                pred = pred.reindex(test.index).bfill()
                results.append((ml_model, pred, compute_metrics(test[target_col].values, pred.values)))
                fig.add_trace(go.Scatter(x=test.index, y=pred, name=ml_model))

        if "LSTM" in models:
            with st.spinner("Training LSTM (this may take a minute)..."):
                pred_df = forecast_lstm_multivariate(train, test_len, n_lags=n_lags, epochs=20)
                pred = pred_df[target_col]
                pred = pred.reindex(test.index).bfill()
                results.append(("LSTM", pred, compute_metrics(test[target_col].values, pred.values)))
                fig.add_trace(go.Scatter(x=test.index, y=pred, name="LSTM"))

        fig.update_layout(title="Multivariate Forecast Comparison (Test Window)", xaxis_title="Date", yaxis_title=target_col, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        metrics_df = pd.DataFrame([
            {"Model": model, **metrics} for (model, _, metrics) in results if metrics is not None and isinstance(metrics, dict)
        ])
        metrics_df = metrics_df.sort_values("RMSE")
        st.subheader("Model Accuracy (Lower is Better)")
        st.dataframe(metrics_df.style.highlight_min(axis=0))