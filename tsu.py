import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

# TODO: Change ARIMA to SARIMAX
# TODO: Add parameter selection methods
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

# Creates and Trains an ARIMA Model with given Time Series and Parameters
def fit_arima(ts, p, d, q, summary=True):
	model = ARIMA(ts, order=(p, d, q))
	model_fit = model.fit(disp=0)
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

# Creates an ARIMA Model used for predicting a given Time Series
def predict_arima(ts, p, d, q, split):
	tr_ts, te_ts = split_ts(ts, split)
	predictions = []
	for i in range(len(te_ts)):
		model_fit = fit_arima(tr_ts, p, d, q, summary=False)
		output = model_fit.forecast()
		predictions.append(output[0])
		tr_ts.append(te_ts[i])
		print("Predicted:", output[0][0], ", Expected:", te_ts[i])
	print('MAE:', calculate_mae(np.array(te_ts), np.array(predictions)))
	plot_predicted(ts, predictions, split)
	return model_fit

# Receives a Trained ARIMA Model and plots its Forecast up to a given Time Value
def forecast_arima(model_fit, ts, limit):
	output = model_fit.forecast(steps=limit)
	fill = np.array(output[2])
	fill = fill.transpose()
	plot_predicted(ts, output[0], len(ts), fill)
