# Time Series Analysis App

This app provides interactive tools for univariate time series analysis with models, such as, AR, MA, ARMA, ARIMA, SARIMA, Linear Regression, Random Forest, XGBoost, LSTM and LightGBM. It features stationarity tests, ACF/PACF plots, model fitting, and forecasting, all accessible via a user-friendly Streamlit interface.

## Features

- Stationarity testing (ADF)
- ACF and PACF visualization
- ARCH/GARCH volatility modeling
- Automated model selection and comparison
- Forecasting with classical and machine learning models

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/HeavenEvolved/VCU-SCMA632.git
   ```

2. Navigate to the project directory:

   ```bash
   cd VCU-SCMA632/A8/Python
   ```

3. Ensure you have Python 3.10 installed. If not, download and install it from [Python.org](https://www.python.org/downloads/).
4. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

5. Activate the virtual environment:
   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

6. Install dependencies:

   ```bash
   pip install -r ../../requirements.txt
   ```

## Usage

1. Start the Streamlit app:

   ```bash
   streamlit run A8-Python.py
   ```

2. Enter a stock ticker in the app interface to fetch data.
3. Explore stationarity using the Augmented Dickey-Fuller test.
4. Visualize autocorrelation and partial autocorrelation functions (ACF & PACF) to determine model orders.
5. Test for ARCH effects using the Lagrange Multiplier test.
6. Fit ARCH and GARCH models if ARCH effects are significant.
7. Compare univariate models and forecast future values using the best model.

### Notes

- Ensure your data is in a supported format (Yahoo Finance Ticker).
- For advanced usage, modify or extend the modules in `modules/` and `partials/`.

---

For questions or issues, please contact the repository owner or open an issue on GitHub.
