import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, activations

# TODO: Plot time series with missing data areas highlighted
# TODO: Update imputation methods
# TODO: Document methods

# Removes random values from received data
def remove_random_values(data, values=10):
  to_return = copy.deepcopy(data)
  removed = []
  for i in range(values):
    ri = random.randint(data.index.start, data.index.stop - 1)
    while ri in removed:
      ri = random.randint(data.index.start, data.index.stop - 1)
    removed.append(ri)
    to_return.iloc[ri] = None
  return to_return

# Returns an array of indices with missing values
# Only works with a single column
def get_missing_indices(data):
    return data.index[data.apply(np.isnan)]

# Returns an array of indices with missing values
# Works with multiple columns
def get_missing_indices_multiple_cols(data):
  ind = None
  for col in data.columns:
    if ind is None:
      ind = get_missing_indices(data[col])
    else:
      ind = ind.append(get_missing_indices(data[col]))
  to_return = list(set(ind))
  to_return.sort()
  return to_return

# Returns columns with missing data
def get_missing_columns(data):
    cols = []
    for col in data.columns:
        if len(get_missing_indices(data[col])) != 0:
            cols.append(col)
    return cols

# Performs spline interpolation
def spline_interpolation(data, order=2):
    return data.interpolate(method='spline', order=order)

# Returns the nearest 2 * n values from a given data point
# Does not include out-of-bound indexes or indexes with missing data
def get_nearest_n(data, ind, n):
    nearest = []
    # Search left of data point
    for i in range(ind - 1, ind - n - 1, -1):
        if i in data.index and not math.isnan(data[i]):
            nearest.append(i)
    # Search right of data point
    for i in range(ind + 1, ind + n + 1, 1):
        if i in data.index and not math.isnan(data[i]):
            nearest.append(i)
    return nearest

# Performs local average imputation
# A missing data point equals the average of the closest values
def nearest_mean_imputation(data, neighbourhood_size=2):
    to_return = copy.deepcopy(data)
    missing_indices = get_missing_indices(data)
    for el in missing_indices:
        nearest_indexes = get_nearest_n(data, el, neighbourhood_size)
        to_return[el] = np.mean(data.iloc[nearest_indexes])
    return to_return

# Creates train and test imputation sets for Multilayer Perceptron model
def train_test_imputation_sets(data, ar_lag=1, order="forwards"):
    missing_data = copy.deepcopy(data)
    # Input and Output with no Missing Values
    train_x = pd.DataFrame()
    train_y = pd.DataFrame()
    # Output with Missing Values
    test_x = pd.DataFrame()
    test_y = pd.DataFrame()
    # Input and Output with Missing Values
    both_x = pd.DataFrame()
    both_y = pd.DataFrame()
    y = missing_data.shift(-1)
    y.drop(y.tail(1).index, inplace=True)
    if ar_lag > 1:
        for i in range(1, ar_lag):
            to_append = missing_data.shift(i).fillna(0)
            for col in to_append.columns:
                to_append = to_append.rename(columns = {col: col + "-" + str(i)})
            missing_data = missing_data.join(to_append)
    missing_indices = get_missing_indices_multiple_cols(data)
    missing_indices_y = get_missing_indices_multiple_cols(y)
    for i in range(len(missing_data) - 1):
        # Input and Output have no Missing Values
        if i not in missing_indices and i not in missing_indices_y:
            train_x = train_x.append(missing_data.iloc[i])
            train_y = train_y.append(y.iloc[i])
        # Input has no Missing Values
        elif i not in missing_indices:
            test_x = test_x.append(missing_data.iloc[i])
            test_y = test_y.append(y.iloc[i])
        # Input and Output have Missing Values
        elif i in missing_indices and i in missing_indices_y:
            both_x = both_x.append(missing_data.iloc[i])
            both_y = both_y.append(y.iloc[i])
    return train_x, train_y, test_x, test_y, both_x, both_y

# Uses model to impute missing values in the data
def mlp_imputation(data, model, test_x, both_x, ar_lag=1):
# TODO: Change to use training/test sets
    to_return = copy.deepcopy(data)
    for i in test_x.index:
        val = model.predict(to_return.loc[i].values.reshape((1, 1, to_return.loc[i].values.shape[0])))
        to_return.loc[i + 1] = val
  # Generates missing sets until there are none
  # A little bit bruteforce...
    while len(both_x) != 0 and len(test_x) != 0:
        _, _, test_x, _, both_x, _ = train_test_imputation_sets(to_return, ar_lag)
        for i in test_x.index:
            val = model.predict(to_return.loc[i].values.reshape((1, 1, to_return.loc[i].values.shape[0])))
            to_return.loc[i + 1] = val
    return to_return

# Creates training/test sets, creates and trains model and performs imputation on the data
def mlp_setup(data, ar_lag=1):
    # Get training/test sets
    train_x, train_y, test_x, test_y, both_x, both_y = train_test_imputation_sets(pd.DataFrame(data), ar_lag=ar_lag)
    train_x_re = np.reshape(train_x.values, (train_x.shape[0], 1, train_x.shape[1]))
    test_x_re = np.reshape(test_x.values, (test_x.shape[0], 1, test_x.shape[1]))
    train_y_re = train_y.values
    test_y_re = test_y.values

    model = keras.models.Sequential([
        layers.Dense(10, activation="relu", input_shape=(train_x_re.shape[1], train_x_re.shape[2])),
        layers.Dense(10, activation="relu", input_shape=(train_x_re.shape[1], train_x_re.shape[2])),
        layers.Dense(train_y_re.shape[-1])
    ])

    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_x_re, train_y_re, epochs=100, shuffle=False)

    imputed_mlp = mlp_imputation(pd.DataFrame(data), model, test_x, both_x, ar_lag=ar_lag)
    return imputed_mlp
