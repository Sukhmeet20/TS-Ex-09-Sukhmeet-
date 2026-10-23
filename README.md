
# Ex-09: The Box–Jenkins Model (ARIMA)
## NAME : Sukhmeet Kaur G
## REG NO : 2305001032
## Aim
To understand and implement the Box–Jenkins approach to time-series forecasting using an ARIMA model in Python.

## Objectives
- To create and visualize monthly sales time-series data.
- To test stationarity using the Augmented Dickey-Fuller (ADF) test.
- To apply differencing and analyze ACF and PACF plots.
- To fit an ARIMA model and compare AIC and BIC values.
- To perform residual diagnostic checking.
- To forecast future sales and visualize the predicted values.

## Program

```python
# Step 1: Import the required libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Step 2: Create sample monthly sales data
sales = [
    100, 105, 103, 110, 115, 112,
    120, 125, 123, 130, 135, 132,
    140, 145, 143, 150, 155, 152,
    160, 165, 163, 170, 175, 172
]

data = pd.Series(sales, name="Sales")

print("STEP 1: Original Sales Data")
print(data.to_string(index=False))

# Step 3: Plot the original time series
plt.figure(figsize=(10, 4))
plt.plot(data, marker="o")
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Test stationarity using the ADF test
adf_result = adfuller(data)

print("\nSTEP 2: ADF Test on Original Data")
print("ADF Statistic:", round(adf_result[0], 4))
print("p-value:", round(adf_result[1], 4))

if adf_result[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Insufficient evidence of stationarity.")

# Step 5: Apply first differencing
diff_data = data.diff().dropna()

print("\nSTEP 3: First-Differenced Data")
print(diff_data.to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(diff_data, marker="o")
plt.title("First-Differenced Sales")
plt.xlabel("Month")
plt.ylabel("Change in Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Test stationarity after differencing
adf_diff = adfuller(diff_data)

print("\nSTEP 4: ADF Test After Differencing")
print("ADF Statistic:", round(adf_diff[0], 4))
print("p-value:", round(adf_diff[1], 4))

if adf_diff[1] < 0.05:
    print("Conclusion: Evidence of stationarity.")
else:
    print("Conclusion: Stationarity is not confirmed.")

# Step 7: Plot ACF and PACF
print("\nSTEP 5: ACF and PACF Plots")

plot_acf(diff_data, lags=10)
plt.tight_layout()
plt.show()

plot_pacf(diff_data, lags=10, method="ywm")
plt.tight_layout()
plt.show()

# Step 8: Fit the ARIMA model
# Example candidate: ARIMA(1,1,1)
model = ARIMA(data, order=(1, 1, 1))
fitted_model = model.fit()

print("\nSTEP 6: ARIMA Model Summary")
print(fitted_model.summary())

# Step 9: Display AIC and BIC
print("\nSTEP 7: Model Selection Criteria")
print("AIC:", round(fitted_model.aic, 4))
print("BIC:", round(fitted_model.bic, 4))

# Step 10: Diagnostic checking of residuals
residuals = fitted_model.resid

plt.figure(figsize=(10, 4))
plt.plot(residuals, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.title("ARIMA Residuals")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nSTEP 8: Residual ACF")
plot_acf(residuals, lags=10)
plt.tight_layout()
plt.show()

# Step 11: Perform the Ljung-Box test
lb_result = acorr_ljungbox(
    residuals,
    lags=[5],
    return_df=True
)

print("\nLjung-Box Test Result")
print(lb_result)

if lb_result["lb_pvalue"].iloc[0] > 0.05:
    print("No significant residual autocorrelation detected at lag 5.")
else:
    print("Residual autocorrelation may remain; review the model.")

# Step 12: Forecast the next six months
forecast = fitted_model.get_forecast(steps=6)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("\nSTEP 9: Forecast for the Next Six Months")

for month, value in enumerate(forecast_mean, start=25):
    print(f"Month {month}: {value:.2f}")

# Display forecast confidence intervals
print("\nForecast Confidence Intervals")
print(forecast_ci.round(2))

# Step 13: Plot actual data and forecast
future_months = range(len(data), len(data) + 6)

plt.figure(figsize=(10, 5))
plt.plot(
    range(len(data)), data,
    marker="o", label="Actual Sales"
)
plt.plot(
    future_months, forecast_mean.to_numpy(),
    marker="o", linestyle="--", label="Forecast"
)

plt.fill_between(
    future_months,
    forecast_ci.iloc[:, 0].to_numpy(),
    forecast_ci.iloc[:, 1].to_numpy(),
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.title("Box-Jenkins ARIMA: Six-Month Sales Forecast")
plt.xlabel("Month (0-based index)")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 14: Display the final result
print("\nRESULT")
print("Box-Jenkins ARIMA modelling completed.")
print("Identification, estimation, diagnostics, and forecasting performed.")
```

## Output
<img width="336" height="553" alt="Screenshot 2026-10-16 155909" src="https://github.com/user-attachments/assets/ee3959f5-f20a-47db-86d5-c2aa05921e14" />
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/1dffc0f4-205d-4217-9fb7-609bacf5cada" />
<img width="1076" height="357" alt="Screenshot 2026-10-16 160017" src="https://github.com/user-attachments/assets/79cbd3d2-cbc8-4596-bbc7-33192457a8ee" />
<img width="432" height="538" alt="Screenshot 2026-10-16 160053" src="https://github.com/user-attachments/assets/634d8e6e-d28a-4c0a-9633-8694239306f6" />
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/1a885878-44b9-45d2-9973-33d730f6a60e" />
<img width="1087" height="377" alt="Screenshot 2026-10-16 160131" src="https://github.com/user-attachments/assets/e3c043e2-45ae-4290-a7e3-9a26d4220dcd" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/2ebccbdb-b768-442b-9412-eaae7b375ce8" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/f5e48ba0-d0da-4c2d-8ea6-43ba510f2fb4" />
<img width="1582" height="641" alt="Screenshot 2026-10-16 160242" src="https://github.com/user-attachments/assets/2af1bb8f-780b-4817-aad8-1103c1399c56" />
<img width="380" height="96" alt="Screenshot 2026-10-16 160511" src="https://github.com/user-attachments/assets/1dc59051-c729-4f7e-a07e-1d4632732463" />
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/3f1a51c9-27af-4116-9860-01565abf9895" />
<img width="643" height="547" alt="Screenshot 2026-10-16 160622" src="https://github.com/user-attachments/assets/b2e49203-5c3e-4047-a7e6-61c448e62fc9" />
<img width="456" height="367" alt="Screenshot 2026-10-16 160801" src="https://github.com/user-attachments/assets/f1960c00-e297-495a-971f-a569e9c76929" />
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/d3086c4b-625f-4f4c-bfbe-d92e09fe2d7f" />

## Result
The Box–Jenkins approach was successfully implemented in Python using the ARIMA model. Stationarity testing, differencing, ACF/PACF analysis, model estimation, residual diagnostic checking, and six-month sales forecasting were performed.
