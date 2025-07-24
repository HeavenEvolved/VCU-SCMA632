# Set the base directory for the project
base_dir <- getwd()
setwd(base_dir)

# Define a helper function to install packages if not already installed
install <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE, quiet = TRUE)
  }
}

# Define a helper function to load packages
load_pkg <- function(pkg) {
  message(paste("Loading package:", pkg))
  library(pkg, character.only = TRUE, quietly = TRUE)
}

# Required packages for this analysis
pkgs <- c("readxl", "vars", "urca", "tseries", "forecast", "quantmod", "rugarch", "FinTS", "tsDyn")

# Apply install and load functions
lapply(pkgs, install)
lapply(pkgs, load_pkg)

##########
# PART A #
##########

# Downloading data for NVDA ticker from Yahoo Finance
ticker <- "NVDA"
getSymbols(ticker, src = "yahoo", from = Sys.Date() - 5*365, auto.assign = TRUE)
data <- get(ticker)

# Use Cl function from quantmod to get the adjusted close series from the data
prices <- Cl(data)

# Compute log returns
log_returns <- diff(log(prices), lag = 1)
log_returns <- na.omit(log_returns)

# Inspect head
head(log_returns)

# Plot closing prices and log returns
par(mfrow = c(2,1))
plot(prices, main = paste(ticker, "Adjusted Close Price"))
plot(log_returns, main = paste(ticker, "Daily Log Returns"), col = "blue")

# ARCH test
ArchTest(ts(log_returns), lags = 5)

# Scale returns by 100 for better convergence
scaled_returns <- log_returns * 100

# Define ARCH(1) specification
spec_arch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 0)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

# Fit ARCH(1) model
fit_arch <- ugarchfit(spec = spec_arch, data = scaled_returns)
show(fit_arch)

# Define GARCH(1,1) specification
spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

# Fit GARCH(1,1) model
fit_garch <- ugarchfit(spec = spec_garch, data = scaled_returns)
show(fit_garch)

# Compare model AIC
cat("ARCH(1) AIC:", infocriteria(fit_arch)[1], "\n")
cat("GARCH(1,1) AIC:", infocriteria(fit_garch)[1], "\n")

# Forecast next 90 days
forecast_days <- 90
garch_forecast <- ugarchforecast(fit_garch, n.ahead = forecast_days)
forecast_vol <- sigma(garch_forecast)

# In-sample volatility and its time index
in_sample_vol <- sigma(fit_garch)
in_sample_idx <- index(in_sample_vol)

# Get last 90 days of conditional volatility from the sample
n_hist <- 90
last_hist_idx <- tail(index(in_sample_vol), n_hist)      # 90 in-sample dates
last_hist_vol <- tail(in_sample_vol, n_hist)             # 90 in-sample volatilities

# Get the next 90 days after the last date of the sample
first_forecast_date <- last(last_hist_idx) + 1
forecast_idx <- seq.Date(from = first_forecast_date, by = "day", length.out = forecast_days)

# Combine all dates and all volatilities
all_dates <- c(last_hist_idx, forecast_idx)
all_vol   <- c(as.numeric(last_hist_vol), as.numeric(forecast_vol))

# Plot the Conditional Volatility of the last 90 days and the forecast for the next 90 days
plot(all_dates, all_vol, type = "l", col = "blue",
     ylab = "Volatility (%)",
     xlab = "Date",
     main = "Conditional Volatility and 90-Day Forecast")
abline(v = last(last_hist_idx), col = "red", lty = 2)
legend("topright", legend = c("In-sample + Forecast", "Forecast Start"),
       col = c("blue", "red"), lty = c(1,2))

##########
# Part B #
##########

# File Config
datasets_dir <- "datasets"
datasets_path <- file.path(base_dir, datasets_dir)
file_name <- "CMO-Historical-Data-Monthly.xlsx"
file_path <- file.path(datasets_path, file_name)

# Read Excel File and handle the Date column
data_raw <- read_excel(file_path, sheet = "Monthly Prices", skip = 6)
data_raw <- as.data.frame(data_raw)
names(data_raw) <- trimws(names(data_raw))
names(data_raw)[1] <- "Year_Month"
data_raw$Year_Month <- as.Date(paste0(
  substr(data_raw$Year_Month, 1, 4), "-",
  substr(data_raw$Year_Month, 6, 7), "-01"))

# Clean up undesirable string patterns for as.numeric conversion
clean_numeric_column <- function(vec) {
  vec <- as.character(vec)
  vec <- trimws(vec)
  vec[vec %in% c("", "-", "NA", "n.a.", "N/A", ".", "#N/A N/A")] <- NA
  suppressWarnings(as.numeric(vec))
}

# Data pre-processing per category
prepare_category_df <- function(df, cols) {
  # Subset data frame to only needed columns (assumes df includes Year_Month)
  df_subset <- df[, c("Year_Month", cols), drop=FALSE]
  
  # Clean columns for numeric conversion
  for (col in cols) {
    df_subset[[col]] <- as.character(df_subset[[col]])
    df_subset[[col]] <- trimws(df_subset[[col]])
    # Replace common non-numeric placeholders with NA
    df_subset[[col]][df_subset[[col]] %in% c("", "-", "NA", "n.a.", "N/A", ".", "#N/A N/A")] <- NA
    # Then convert to numeric
    df_subset[[col]] <- suppressWarnings(as.numeric(df_subset[[col]]))
  }
  
  # Remove any rows with NA
  n_before <- nrow(df_subset)
  df_subset <- na.omit(df_subset)
  n_after <- nrow(df_subset)
  cat(sprintf("Rows before cleaning: %d. After omitting NA: %d. Removed: %d\n", n_before, n_after, n_before - n_after))
  
  # Convert to base data.frame to allow rownames
  df_subset <- as.data.frame(df_subset)
  
  # Set row names as dates for convenience (optional)
  rownames(df_subset) <- as.character(df_subset$Year_Month)
  
  # Remove Year_Month column to keep only numeric columns
  df_subset$Year_Month <- NULL
  
  # Verify all columns numeric and no NAs remain
  for (col in cols) {
    if (!is.numeric(df_subset[[col]])) stop(paste("Column", col, "is not numeric after cleaning!"))
    if (any(is.na(df_subset[[col]]))) stop(paste("Column", col, "still contains NA after cleaning!"))
  }
  
  # Log transform prices
  log_df <- log(df_subset)
  
  # Differencing log prices
  log_diff_df <- diff(as.matrix(log_df))
  log_diff_df <- as.data.frame(log_diff_df)
  
  return(list(log = log_df, diff_log = log_diff_df))
}

# Augmented Dickey-Fuller test per column for stationarity of differenced log prices
adf_test <- function(diff_df) {
  cat("ADF Test Results (on differenced log prices):\n")
  for (col in colnames(diff_df)) {
    res <- adf.test(diff_df[[col]])
    stationary <- ifelse(res$p.value < 0.05, "Stationary", "Non-stationary")
    cat(sprintf("%-15s: ADF Stat = %.3f, p-value = %.4f --> %s\n", col, res$statistic, res$p.value, stationary))
  }
}

# VAR lag selection, fitting, impulse response plotting
run_var <- function(diff_df) {
  lag_sel <- VARselect(diff_df, lag.max = 12, type = "const")
  cat("Lag order selection (AIC, BIC, HQIC):\n")
  print(lag_sel$selection)
  
  # Extract lag as numeric scalar (important!)
  selected_lag <- as.numeric(lag_sel$selection["AIC(n)"])
  cat("Selected lag order by AIC:", selected_lag, "\n")
  
  # Fit VAR model with numeric lag
  var_model <- VAR(diff_df, p = selected_lag, type = "const")
  
  cat("VAR Model summary:\n")
  print("VAR Model object created. Checking its structure:")
  print(summary(var_model))
  
  # Impulse response function plots
  irf_res <- irf(var_model, n.ahead = 10, ortho = FALSE, boot = FALSE)
  plot(irf_res)
  
  return(list(model = var_model, lag = selected_lag))
}

# Johansen cointegration test wrapper and estimation
run_johansen <- function(log_df, lag_diff) {
  johansen_result <- ca.jo(log_df, type = "trace", ecdet = "const", K = lag_diff + 1)
  cat("Johansen cointegration test summary:\n")
  print(summary(johansen_result))
  return(johansen_result)
}

get_coint_rank <- function(johansen_result) {
  trace_stats <- johansen_result@teststat
  crit_vals <- johansen_result@cval[, "5pct"]
  estimated_rank <- sum(trace_stats > crit_vals)
  cat(sprintf("Estimated cointegration rank (5%% significance): %d\n", estimated_rank))
  return(estimated_rank)
}

fit_vecm <- function(log_df, coint_rank, lag_diff) {
  vecm_model <- VECM(log_df, lag = lag_diff, r = coint_rank, estim = "ML")
  cat("VECM fit summary:\n")
  print(summary(vecm_model))
}

# High-level function for category-wise analysis
do_analysis <- function(df, category_name, cols) {
  cat("\n======================================\n")
  cat(sprintf("Analyzing %s Commodities\n", category_name))
  cat("======================================\n")
  
  # Check all columns present
  if (!all(cols %in% colnames(df))) {
    missing_cols <- cols[!cols %in% colnames(df)]
    stop(paste("Error: Missing columns in data frame:", paste(missing_cols, collapse = ", ")))
  }
  
  prepared <- prepare_category_df(df, cols)
  log_df <- prepared$log
  diff_log_df <- prepared$diff_log
  
  cat("\nADF Test for Stationarity:\n")
  adf_test(diff_log_df)
  
  cat("\nFitting VAR Model:\n")
  var_results <- run_var(diff_log_df)
  selected_lag <- as.numeric(var_results$lag)
  
  cat("\nConducting Johansen Cointegration Test:\n")
  johansen_res <- run_johansen(log_df, lag_diff = selected_lag)
  
  rank <- get_coint_rank(johansen_res)
  if (rank > 0) {
    cat("\nFitting VECM Model:\n")
    fit_vecm(log_df, rank, selected_lag)
  } else {
    cat("\nNo significant cointegration detected; skipping VECM.\n")
  }
}

# ---- Perform Analysis on Commodity Groups and Subgroups ----

# Define your analysis groups (edit these lists for your own use case)
crude_oil_cols    <- c('CRUDE_BRENT', 'CRUDE_WTI', 'CRUDE_DUBAI')
natural_gas_cols  <- c('NGAS_US', 'NGAS_EUR', 'NGAS_JP')
coal_cols         <- c('COAL_AUS', 'COAL_SAFRICA')
metals_cols       <- c('GOLD', 'SILVER', 'COPPER', 'ALUMINUM')
wheat_cols        <- c('WHEAT_US_HRW', 'WHEAT_US_SRW')
sugar_cols        <- c('SUGAR_EU', 'SUGAR_US', 'SUGAR_WLD')

do_analysis(data_raw, "Crude Oil", crude_oil_cols)
do_analysis(data_raw, "Natural Gas", natural_gas_cols)
do_analysis(data_raw, "Coal", coal_cols)
do_analysis(data_raw, "Metals", metals_cols)
do_analysis(data_raw, "Wheat", wheat_cols)
do_analysis(data_raw, "Sugar", sugar_cols)

