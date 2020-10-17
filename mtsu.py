import matplotlib.pyplot as plt
import numpy as np
import pandas
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from tsu import *

def read_dataset(filename):
    return pandas.read_csv(filename)

# Creates and Saves a plot of each Variable along with 
# Autocorrelation Function and Partial Autocorrelation Function
def plot_mts(data):
    for i, column in enumerate(data.columns):
        fig, axs = plt.subplots(3)
        axs[0].plot(np.arange(len(data[column])), data[column])
        plot_acf(data[column], ax=axs[1])
        plot_pacf(data[column], ax=axs[2])
        plt.savefig("Class" + str(i) + ".png")
        plt.close()

# Runs the Augmented Dickey-Fuller Test for each Variable
def multivariate_adf_test(data):
    for col in data.columns:
        print(col)
        adf_test(data[col])

# Calculates and Outputs a Matrix consisting of a Granger Causality Test
# for each pair of Variables, where values below 0.05 indicate that the
# Variable of the Column can be used to predict the Variable of the Row
def granger_causality_matrix(data, maxlag=10, plot=False):
    cols = len(data.columns)
    matrix = np.zeros([cols, cols])
    for i, row in enumerate(data.columns):
        for j, col in enumerate(data.columns):
            test_results = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=False)
            p_values = []
            for lag in range(1, maxlag + 1):
                p_values.append(test_results[lag][0]['ssr_chi2test'][1])
            matrix[i, j] = min(p_values)
    if plot:
        plt.imshow(matrix)
        plt.colorbar()
        plt.show()
    return matrix

# Performs Ljung-Box Test on the Residuals of a given Model
# Values of lb_value lower than 0.05 may indicate the Model does not
# have enough parameters
def ljung_box_test(data, limit=10, order=5):
    res = VAR(data).fit(order)
    for col in data.columns:
        print(col)
        print(acorr_ljungbox(res.resid[col], lags=limit, return_df=True))
        print("-----------------------------")

# Uses the select_order function that outputs, for lags ranging
# from 0 to maxlags, the AIC, BIC, FPE and HQIC values for each
# lag
# Lags with higher amount of minimums are preferred
def var_order(data):
    model = VAR(data)
    res = model.select_order(maxlags=12)
    print(res.summary())

# Plots the AIC Values
def plot_order_aics(data, limit=10):
    model = VAR(data)
    orders = [i for i in range(1, limit + 1)]
    aics = []
    for i in orders:
        res = model.fit(i)
        aics.append(res.aic)
    plt.plot(orders, aics)
    plt.show()

# Outputs the Durbin-Watson Statistic
# Orders with Values closer to 2 are preferred
def durbin_watson_statistic(data, p):
    model_fit = VAR(data).fit(p)
    return durbin_watson(model_fit.resid)

# Creates and Trains a VAR Model with given Time Series and Order
def fit_var(data, p, summary=True):
    model = VAR(data)
    model_fit = model.fit(p)
    if summary:
        print(model_fit.summary())
    return model_fit

def calculate_mae(ts, predicted):
    return np.mean(np.abs(predicted - ts))

# Creates a VAR Model used for predicting a given Time Series
def predict_var(data, p, split):
    data_tr, data_ts = split_ts(data, split)
    predictions = [[] for col in data.columns]
    for i in range(split + 1, len(data_ts) + split + 1):
        model_fit = fit_var(data_tr, p, summary=False)
        output = model_fit.forecast(y=data_tr.values[-p:], steps=1)
        data_tr = data_tr.append(data_ts.loc[i,:])
        for i in range(len(data.columns)):
            predictions[i].append(output[0][i])
    for i in range(len(data.columns)):
        print('MAE(' + data.columns[i] + '):', calculate_mae(data[data.columns[i]][split:], predictions[i]))
        plot_predicted(data[data.columns[i]], predictions[i], split + 1)

# Receives a Trained VAR Model and plots its Forecast up to a given Time Value
def forecast_var(model_fit, ts, limit):
    p = model_fit.k_ar
    forecast_input = ts[-p:].values
    fc = model_fit.forecast(y=forecast_input, steps=limit)
    for i in range(len(ts.columns)):
        plot_predicted(ts[ts.columns[i]], fc[:, i], len(ts))
    return fc
