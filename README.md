# Hindsight

Two dashboards for analysing temporal data: Hindsight_ts (for time series) and Hindsight_ld (for longitudinal data).

![Hindsight_ts](https://github.com/Seawyn/Hindsight/blob/main/Screenshots/Hindsight_ts_screenshot.png)

## Hindsight_ts

### Features

- Analyse variables with respect to stationarity, missing values, ACF and PACF and seasonal decomposition;
- Perform first difference and missing value imputation;
- Estimate causal relations;
- Train linear time series models and neural networks;
- Analyse results (errors, residual distribution and correlations, gradients)

## Hindsight_ld

### Features

- Analyse variables with respect to missing values;
- Perform discretization and missing value imputation;
- Train neural networks;
- Analyse results (weights and gradients)

## Requirements

- Plotly
- Dash
- Dash Bootstrap Components
- numpy
- pandas
- statsmodels
- tigramite
- tensorflow
- scikit-learn
- scipy
- pydybm
