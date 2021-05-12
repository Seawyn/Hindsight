import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from tsu import plot_predicted

def setup(data, split, x, y):
    x_train, x_test = data[x][:split], data[x][split:]
    y_train, y_test = data[y][:split], data[y][split:]

    x_train = np.reshape(list(x_train), (len(x_train), 1))
    y_train = np.reshape(list(y_train), (len(y_train), 1))
    x_test = np.reshape(list(x_test), (len(x_test), 1))
    y_test = np.reshape(list(y_test), (len(y_test), 1))

    return x_train, y_train, x_test, y_test

def fit_rf(data, split, x, y, estimators):
    x_train, y_train, x_test, y_test = setup(data, split, x, y)
    model = RandomForestRegressor(n_estimators=estimators, max_features=1, oob_score=True)
    model_fit = model.fit(x_train, y_train)
    #plot_predicted(y_train, model_fit.predict(x_train), split=0)
    plot_predicted(y_test, model_fit.predict(x_test), split=0)
