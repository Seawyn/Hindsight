import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# TODO: Change to Plotly
# TODO: Add time series simulation

# Opens file with the name given as input
# Use header = 0 if variables have names
# Use transpose if in a classification context, leave as is if using regression
# Outputs a DataFrame Object
def read_dataset(filename, header=None, classification=False, sep=","):
	df = pd.read_csv(filename, sep=sep, header=header)
	if classification:
		return df.transpose()
	else:
		return df

# Opens a given dataset and exports the to_extract column/columns to a csv file
# csv file is comma separated
def export_ts(filename, to_extract, output_name):
	df = pd.read_csv(filename)
	df[to_extract].to_csv(output_name)

# Creates a plot with Chronogram and Autocorrelation Function of input
def visualize_ts(df, savename=None):
	fig, axs = plt.subplots(3, 1)
	fig.set_size_inches(8, 8)
	axs[0].plot(np.arange(len(df)), df)
	smt.graphics.plot_acf(df, lags=50, alpha=0.05, ax=axs[1])
	smt.graphics.plot_pacf(df, lags=50, alpha=0.05, ax=axs[2])
	if savename is None:
		plt.show()
	else:
		plt.savefig(savename)

# Returns Autocorrelation and Partial Autocorrelation Functions of of the input
def get_acf_and_pacf(df, alpha=0.05):
	n_lags = min(50, (len(df) // 2) - 1)
	acf_data = acf(df, nlags=n_lags, fft=True, alpha=alpha)
	pacf_data = pacf(df, nlags=n_lags, method='ywm')
	return acf_data, pacf_data

# Returns the Seasonal Decomposition results of a given variable
def get_seasonal_decomposition(df, model='additive', period=3):
	return seasonal_decompose(df, model=model, period=period)

# Applies the Augmented Dickey-Fuller Test and returns only the p-value
# Used for the application
def adf_pvalue(df):
	results = adfuller(df)
	return results[1]

# Applies the Augmented Dickey-Fuller Test
# Values equal or below of 0.05 refuse the possibility of the Time Series not being Stationary
def adf_test(df):
    results = adfuller(df)
    ap = "(Series is likely to " + "not " * (results[0] >= results[4]['5%']) + "be Stationary)"
    print("-------------------------")
    print("ADF Value:", results[0], ap)
    print("P-Value:", results[1])
    print("Critical Values:")
    for key in results[4].keys():
        print(key, results[4][key])
    print("-------------------------")


# Applies the Fisher's Test to a Time Series
# Values equal or below of 0.05 refuse the possibility of the existence of a cyclic component
def Fishers_Test(df):
	n = len(df)

	if n % 2 == 0:
		s, l = 0, 1
		e = n // 2
	else:
		s, l = 1, 2
		e = math.floor((n - 1) / 2)

		# Calculate b
		b = np.zeros(e)
		for k in range(e):
			val = 0
			for i in range(n):
				val += df[i] * math.sin(2 * math.pi * (k + s) * (i + 1) / n)
			val = l * val / n
			b[k] = val

	a = np.zeros(e)

	# Calculate a
	for k in range(e):
		val = 0
		for i in range(n):
			val += df[i] * math.cos(2 * math.pi * (k + s) * (i + 1) / n)
		val = l * val / n
		a[k] = val

	# Calculate I
	I = np.zeros(e)
	if n % 2 == 0:
		for k in range(e):
			I[k] = n * math.pow(a[k], 2)
	else:
		for k in range(e):
			I[k] = (n / 2) * (math.pow(a[k], 2) + math.pow(b[k], 2))


	I_m = I.max()
	return I_m / I.sum()

# Stabilizes Variance
# Recomended use before Stabilizing the Mean
def stabilize_variance(df):
	if min(df) > 0:
		print("Using Box-Cox Power Transformation...")
		return boxcox(df)
	else:
		print("Using Yeo-Johnson Power Transformation...")
		return yeojohnson(df)

# Applies the Difference Operator to Stabilize the Mean
def difference(df, order=1):
    for i in range(order):
        df = df.diff().dropna()
    return df

# Splits a given Time Series into Training and Testing Time Series
def split_ts(ts, split):
	return ts[0:split], ts[split:]

# Calculates Mean Absolute Error between two Vectors
def calculate_mae(ts, predicted):
    return np.mean(np.abs(predicted - ts))

# Calculates the residuals of a prediction (observed - predicted)
# Note: both inputs must be numpy arrays with equal shapes
def calculate_residuals(obs, pred):
	residuals = np.zeros(obs.shape)
	for i in range(len(obs)):
		residuals[i] = obs[i] - pred[i]
	return residuals

# Creates and trains an ARIMA model with given time series and parameters
def fit_arima(ts, p, d, q, summary=True):
	model = ARIMA(ts, order=(p, d, q))
	model_fit = model.fit()
	if summary:
		print(model_fit.summary())
	return model_fit

# Plots Time Series along with Predicted Values
def plot_predicted(ts, pr, split, fill=None):
	plt.plot(np.arange(len(ts)), ts)
	plt.plot(np.arange(split, (split + len(pr))), pr, color='red')
	if not fill is None:
		plt.fill_between(np.arange(split, (split + len(pr))), fill[0], fill[1], facecolor=(0.8, 0.8, 0.8))
	plt.show()

# Creates an ARIMA model used for predicting a given time series
def predict_arima(ts, p, d, q, split, display_res=False):
	tr_ts, te_ts = split_ts(ts, split)
	predictions = []
	for i in range(len(te_ts)):
		model_fit = fit_arima(tr_ts, p, d, q, summary=False)
		output = model_fit.forecast()
		predictions.append(output[0])
		tr_ts.append(te_ts[i])
		print("Predicted:", output[0], ", Expected:", te_ts[i])
	print('MAE:', calculate_mae(np.array(te_ts), np.array(predictions)))
	if display_res:
		plot_predicted(ts, predictions, split)
	return model_fit

# Receives a trained ARIMA model and plots its forecast up to a given time value
def forecast_arima(model_fit, ts, limit):
	output = model_fit.forecast(steps=limit)
	fill = np.array(output[2])
	fill = fill.transpose()
	plot_predicted(ts, output[0], len(ts), fill)

# Creates and trains a SARIMAX model with given time series and parameters
# Used by the App
def fit_sarimax(ts, params, summary=True):
	if params[-1] == 0:
		model = SARIMAX(ts, order=params[:3], initialization="approximate_diffuse")
	else:
		model = SARIMAX(ts, seasonal_order=params, initialization="approximate_diffuse")
	model_fit = model.fit(disp=0)
	if summary:
		print(model_fit.summary())
	return model_fit

# Creates a SARIMAX model used for predicting a given time series
# Used by the App
def predict_sarimax(ts, params, split, var, status=True):
	tr_ts, te_ts = split_ts(ts, split)
	predictions = []
	# For all indexes in the testing set...
	#for i in np.arange(te_ts.index[0], te_ts.index[-1] + 1):
	for i in te_ts.index:
		model_fit = fit_sarimax(tr_ts, params, summary=False)
		output = model_fit.forecast()
		predictions.append(output.values[0])
		# Add test set observation to training set
		tr_ts.loc[i] = te_ts.loc[i]
		if status:
			print(i, "-", "Predicted:", output.values[0], ", Expected:", te_ts.loc[i])
	predictions = np.array(predictions)
	error = {var: calculate_mae(te_ts.values, predictions)}
	residuals = calculate_residuals(te_ts.values, predictions)
	return model_fit, predictions, error, residuals

# Receives a trained SARIMAX model and outputs a forecast of size "limit"
# and confidence intervals
def forecast_sarimax(model_fit, ts, limit):
	output = model_fit.get_forecast(steps=limit)
	predicted = output.predicted_mean.transpose().values
	fill = output.conf_int().transpose().values
	return predicted, fill

# Creates a heatmap with criterion results for each parameter set
def param_heatmap(ts, limit_p, limit_q, itr, s=0):
	aics = np.zeros((limit_p, limit_q))
	aiccs = np.zeros((limit_p, limit_q))
	bics = np.zeros((limit_p, limit_q))
	for i in range(limit_p):
		for j in range(limit_q):
			if s == 0:
				model = SARIMAX(ts, order=(i, itr, j), initialization="approximate_diffuse")
			else:
				model = SARIMAX(ts, seasonal_order=(i, itr, j, s), initialization="approximate_diffuse")
			model_fit = model.fit(disp=0)
			aics[i, j] = model_fit.aic
			aiccs[i, j] = model_fit.aicc
			bics[i, j] = model_fit.bic
	heatmaps = {'aic': aics, 'aicc': aiccs, 'bic': bics}
	return heatmaps

# Generates all possible combinations of two numbers whose sum is n
# Note: Combinations are restricted by matrix size (for example, 5x5 cannot have (7, 1))
def combinations_sum(n, limit_p, limit_q):
	combinations = []
	limit = round((n // 2) + (n % 2))
	for i in range(n, limit - 1, -1):
		if i < limit_p and (n - i) < limit_q:
			combinations.append((i, n - i))
		if i < limit_q and (n - i) < limit_p and (n - i) != i:
			combinations.append((n - i, i))
	return combinations

# Receives criterion matrix and returns the update of minimums
# Note: Search pattern is in ascending order of parameters
def param_min_search(heatmaps, limit_p, limit_q):
	# Parameters of each minimum
	params_aic = []
	params_aicc = []
	params_bic = []
	# Minimums of each criterion
	mins_aic = []
	mins_aicc = []
	mins_bic = []
	# Current minimums of each criterion
	current_min_aic = None
	current_min_aicc = None
	current_min_bic = None
	for i in range(0, sum(heatmaps['aic'].shape) - 1):
		# Generate coordinates of current "diagonal line"
		for comb in combinations_sum(i, limit_p, limit_q):
			p, q = comb
			current_aic = heatmaps['aic'][p, q]
			current_aicc = heatmaps['aicc'][p, q]
			current_bic = heatmaps['bic'][p, q]

			# Check for new AIC minimum
			if current_min_aic is None or current_min_aic > current_aic:
				current_min_aic = current_aic
				params_aic.append(str(comb))
				mins_aic.append(current_aic)

			# Check for new AICc minimum
			if current_min_aicc is None or current_min_aicc > current_aicc:
				current_min_aicc = current_aicc
				params_aicc.append(str(comb))
				mins_aicc.append(current_aicc)

			# Check for new BIC minimum
			if current_min_bic is None or current_min_bic > current_bic:
				current_min_bic = current_bic
				params_bic.append(str(comb))
				mins_bic.append(current_bic)
	min_res = {'aic': {'parameters': params_aic, 'mins': mins_aic},
				'aicc': {'parameters': params_aicc, 'mins': mins_aicc},
				'bic': {'parameters': params_bic, 'mins': mins_bic}}
	return min_res
