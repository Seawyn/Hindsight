import matplotlib.pyplot as plt
import numpy as np
import pandas
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR, VARMAX
from Utils.tsu import *

# Creates and saves a plot of each variable along with
# Autocorrelation Function and Partial Autocorrelation Function
def plot_mts(data):
    for i, column in enumerate(data.columns):
        fig, axs = plt.subplots(3)
        axs[0].plot(np.arange(len(data[column])), data[column])
        plot_acf(data[column], ax=axs[1])
        plot_pacf(data[column], ax=axs[2])
        plt.savefig("Class" + str(i) + ".png")
        plt.close()

# Runs the Augmented Dickey-Fuller test for each variable
def multivariate_adf_test(data):
    for col in data.columns:
        print(col)
        adf_test(data[col])

# Inverses difference operations on the provided data
# The number of inversions and the values both depend on the
# provided values on vals (one value per inversion)
# Method only accepts one column as data input, and values should be
# in an iterable object (list or numpy array)
def reverse_difference(data_diff, vals):
    if len(data_diff.columns) != 1:
        raise ValueError('Data should contain only one Column!')
    data_diff = data_diff
    if len(vals) != 1:
        data_diff = reverse_difference(data_diff, vals[1:])
        data_diff = data_diff[data_diff.columns[0]]
    orig = [vals[0]]
    last_val = vals[0]
    for i in data_diff.index:
        last_val = data_diff[i] + last_val
        orig.append(last_val)
    orig = pd.DataFrame(orig)
    orig.columns = [data_diff.name]
    return orig

# Performs Ljung-Box test on the residuals of a given model
# Values of lb_value lower than 0.05 may indicate the model does not
# have enough parameters
def ljung_box_test(data, limit=10, order=5):
    res = VAR(data).fit(order)
    for col in data.columns:
        print(col)
        print(acorr_ljungbox(res.resid[col], lags=limit, return_df=True))
        print("-----------------------------")

# Uses the select_order function that outputs, for lags ranging
# from 0 to maxlags, the AIC, BIC, FPE and HQIC values for each lag
# Lags with higher amount of minimums are preferred
def var_order(data):
    model = VAR(data)
    res = model.select_order(maxlags=12)
    print(res.summary())

# Plots the AIC values
def plot_order_aics(data, limit=10):
    model = VAR(data)
    orders = [i for i in range(1, limit + 1)]
    aics = []
    for i in orders:
        res = model.fit(i)
        aics.append(res.aic)
    plt.plot(orders, aics)
    plt.show()

# Outputs the Durbin-Watson statistic
# Orders with values closer to 2 are preferred
def durbin_watson_statistic(data, p):
    model_fit = VAR(data).fit(p)
    return durbin_watson(model_fit.resid)

# Calculates mean average error between two vectors
def calculate_mae(ts, predicted):
    return np.mean(np.abs(predicted - ts))

# Creates and trains a VAR model with given time series and order
def fit_var(data, p, summary=True):
    model = VAR(data)
    model_fit = model.fit(p)
    if summary:
        print(model_fit.summary())
    return model_fit

# Creates a VAR model used for predicting a given time series
def predict_var(data, p, split):
	data_tr, data_ts = split_ts(data, split)
	predictions = [[] for col in data.columns]
	for i in data_ts.index:
		model_fit = fit_var(data_tr, p, summary=False)
		output = model_fit.forecast(y=data_tr.values[-p:], steps=1)
		data_tr = data_tr.append(data_ts.loc[i,:])
		for i in range(len(data.columns)):
			predictions[i].append(output[0][i])
	errors = {}
	for i in range(len(data.columns)):
		errors[data.columns[i]] = calculate_mae(data_ts[data.columns[i]], predictions[i])
	residuals = calculate_residuals(data_ts.values, np.array(predictions).transpose())
	return model_fit, predictions, errors, residuals

# Receives a trained VAR model and calculates its forecast and confidence intervals with a given size
def forecast_var(model_fit, ts, limit):
    p = model_fit.k_ar
    forecast_input = ts[-p:].values
    fc = model_fit.forecast_interval(y=forecast_input, steps=limit)
    return fc

# Creates a VARMAX model used for predicting a given time series
def fit_varmax(data, p, q, exog=None, summary=True):
    model = VARMAX(data, order=(p, q), exog=exog, initialization='approximate_diffuse')
    model_fit = model.fit(maxiter=50, disp=False)
    if summary:
        print(model_fit.summary())
    return model_fit

# Receives a trained VARMAX model and plots its forecast up to a given time value
def predict_varmax(data, p, q, split, exog=None):
	data_tr, data_ts = split_ts(data, split)
	predictions = pd.DataFrame()
	for i in data_ts.index:
		if not exog is None:
			exog_input = exog[:split + i - data_ts.index[0]]
		else:
			exog_input = None
		model_fit = fit_varmax(data_tr, p, q, exog_input, summary=False)
		output = model_fit.forecast(y=data_tr.values[-p:], steps=1)
		data_tr = data_tr.append(data_ts.loc[i, :])
		predictions = predictions.append(output)
	errors = {}
	for i in range(len(data.columns)):
		errors[data.columns[i]] = calculate_mae(data[data.columns[i]][split:], predictions[predictions.columns[i]])
    #    print('MAE(' + data.columns[i] + '):', calculate_mae(data[data.columns[i]][split:], predictions[predictions.columns[i]]))
    #    plot_predicted(data[data.columns[i]], predictions[predictions.columns[i]], split)
	return model_fit, preditions, errors

# Returns an array of AIC and BIC values up to a given parameter
def var_param_search(data, limit_p):
    aics = []
    bics = []
    for i in range(limit_p + 1):
        model = VAR(data)
        model_fit = model.fit(i)
        aics.append(model_fit.aic)
        bics.append(model_fit.bic)
    return {'aic': aics, 'bic': bics}

# Returns the evolution of minimums along the parameters for each criterion
def var_param_min_search(data, limit_p):
    # Parameters of each minimum
    param_aic = []
    param_bic = []
    # Minimums of each criterion
    mins_aic = []
    mins_bic = []
    # Current minimums of each criterion
    current_min_aic = None
    current_min_bic = None
    for i in range(limit_p):
        model = VAR(data)
        model_fit = model.fit(i)
        current_aic = model_fit.aic
        current_bic = model_fit.bic

        # Check for new AIC minimum
        if current_min_aic is None or current_min_aic > current_aic:
            current_min_aic = current_aic
            param_aic.append(str(i))
            mins_aic.append(current_aic)

        # Check for new BIC minimum
        if current_min_bic is None or current_min_bic > current_bic:
            current_min_bic = current_bic
            param_bic.append(str(i))
            mins_bic.append(current_bic)

    res = {'aic': {'parameters': param_aic, 'mins': mins_aic},
            'bic': {'parameters': param_bic, 'mins': mins_bic}}
    return res
