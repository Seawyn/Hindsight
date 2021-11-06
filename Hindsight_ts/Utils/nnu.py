import math
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from Utils.tsu import calculate_mae, calculate_residuals
import warnings

# Receives a time series dataset and creates training, test and, optionally, validation sets
# data: dataset, as numpy array (or possibly DataFrame)
# order: number of previous timesteps to be included in the input, besides current timestep (0 or higher)
# split: index that separates training and test sets
# val_split: size of validation set, takes last n entrances (0 or higher)
# s: number of observations that are calculated by each model, affects how many models are trained,
#   used for DIRMO/MISMO strategies (0 or max will result in only one model)
# last_step: number of last observation entries (rows) that are present in "last" which is input for forecasting
def series_to_supervised_multi_output(data, order, split, output_window, val_split=0, s=0, last_step=1, variables=None):
  if s == 0:
    s = output_window

  if output_window % s != 0 or s > output_window:
    raise Exception("output_window must be a multiple of s")

  else:
    num_cols = data.shape[1]

  df = pandas.DataFrame(data)
  df_y = pandas.DataFrame()
  for col in df.columns:
    if variables is None:
      df = df.rename(columns = {col: 'var' + str(col)})
    else:
      df.columns = variables

  df_shifts = {}

  for i in range(1, output_window + 1):
    df_shifts['+' + str(i)] = df.shift(-i)
  for key in df_shifts.keys():
    for col in df_shifts[key].columns:
      df_y[str(col) + key] = df_shifts[key][col]

  if val_split == 0:
    val_x_split = -len(df)
    val_y_split = -len(df_y)
    warnings.warn("x_val and y_val are empty DataFrames.")
  else:
    val_x_split = val_split
    val_y_split = val_split

  shifts = {}

  if order > 0:
    for i in range(1, order + 1):
      shifts['-' + str(i)] = df.shift(i)
  for key in shifts.keys():
    for col in shifts[key].columns:
      df[str(col) + key] = shifts[key][col]

  indexes_with_nan = df_y[df_y.columns[-1]].index[df_y[df_y.columns[-1]].apply(np.isnan)]
  last = df[-last_step:]
  df = df.drop(indexes_with_nan).fillna(0)
  df_y = df_y.drop(indexes_with_nan)

  y_train = df_y[df_y.columns][:split]
  y_test = df_y[df_y.columns][split:-val_x_split]
  y_val = df_y[df_y.columns][-val_x_split:]

  x_train = df[df.columns][:split]
  x_test = df[df.columns][split:-val_x_split]
  x_val = df[df.columns][-val_x_split:]

  if s != output_window:
    y_train = split_y_data(y_train, num_cols, s)
    y_test = split_y_data(y_test, num_cols, s)
    y_val = split_y_data(y_val, num_cols, s)

  sets = {}
  sets['x_train'], sets['x_test'] = x_train, x_test
  sets['y_train'], sets['y_test'] = y_train, y_test
  sets['x_val'], sets['y_val'] = x_val, y_val
  sets['last'] = last

  return sets

# Preprocesses the dataset: normalization, train-test split, reshaping
# Returns scaler (for later denormalization) and processed data
def preprocess(data, order, split, output_window=1, val_split=0, output_vars=None):
    # Get index of output variables, if specified
    if not output_vars is None:
        filter = []
        for i in range(len(data.columns)):
            if data.columns[i] in output_vars:
                filter.append(i)

        # Get indexes of output variables factoring output size
        all_out_cols = []
        for i in range(output_window):
            for el in filter:
                all_out_cols.append(el + int(len(data.columns) * i))

    # TODO: reshape x_val and y_val

    values = data.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)

    sets = series_to_supervised_multi_output(scaled, order, split, output_window, val_split=val_split)
    original_sets = series_to_supervised_multi_output(data, order, split, 1, val_split=val_split, variables=data.columns)

    x_train = np.reshape(sets['x_train'].values, (sets['x_train'].shape[0], 1, sets['x_train'].shape[1]))
    x_test = np.reshape(sets['x_test'].values, (sets['x_test'].shape[0], 1, sets['x_test'].shape[1]))
    y_train = sets['y_train'].values
    y_test = sets['y_test'].values

    if not output_vars is None:
        y_train = y_train[:, all_out_cols]
        y_test = y_test[:, all_out_cols]
    else:
        all_out_cols = None

    # TODO: Refactor dictionary creation
    processed = {}
    processed['x_train'], processed['x_test'] = x_train, x_test
    processed['y_train'], processed['y_test'] = y_train, y_test
    processed['x_val'], processed['y_val'] = sets['x_val'], sets['y_val']
    processed['last'] = sets['last']

    return scaler, processed, original_sets, all_out_cols

# Creates and compiles a Bidirectional LSTM neural network
def create_bilstm_model(x_train, num_nodes=(20, 20), activations=('relu', 'relu'), dropout=0.2, output_size=1, loss='mae', conf_int=False, seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2]))

    # Return sequences will be True if there is no second layer
    step = layers.Bidirectional(layers.LSTM(num_nodes[0], return_sequences=(num_nodes[1] is None), activation=activations[0]))(inputs)

    # Only create second layer if there are hidden nodes
    if not num_nodes[1] is None:
        step = layers.RepeatVector(1)(step)
        step = layers.Bidirectional(layers.LSTM(num_nodes[1], return_sequences=True, activation=activations[1]))(step)

    # Dropout and output
    step = layers.Dropout(dropout)(step, training=conf_int)
    outputs = layers.TimeDistributed(layers.Dense(output_size))(step)

    lstm_model = keras.models.Model(inputs, outputs)
    lstm_model.compile(loss=loss, optimizer='adam')

    return lstm_model

# Creates and compiles a CNN neural network
def create_cnn_model(x_train, num_filters=(32, 16), kernel_sizes=(4, 2), activations=('relu', 'relu'), dropout=0.2, output_size=1, loss='mae', conf_int=False, seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2]))

    step = layers.Conv1D(filters=num_filters[0], kernel_size=kernel_sizes[0], activation=activations[1], padding='same')(inputs)

    # Only create second layer if there are hidden nodes
    if not num_filters[1] is None:
        step = layers.Conv1D(filters=num_filters[1], kernel_size=kernel_sizes[1], activation=activations[1], padding='same')(step)

    # Flatten and middle Dense layer
    step = layers.Flatten()(step)
    step = layers.Dense(units=32, activation='relu')(step)

    # Dropout and output
    step = layers.Dropout(dropout)(step, training=conf_int)
    outputs = layers.Dense(units=output_size)(step)

    cnn_model = keras.models.Model(inputs, outputs)
    cnn_model.compile(loss=loss, optimizer='adam')

    return cnn_model

# Trains received model and outputs its test set prediction and error
def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=10):
    # TODO: Change parameters to dictionary of sets
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=False)
    pr_test = model.predict(x_test)
    pr_train = model.predict(x_train)
    return model, pr_test, pr_train, history

# Calculates predictions for each input, and returns a dataframe with all averages
# Used for multi-output predictions, for single output, the results should not be changed
def get_predictions(pr_test, number_predictions, available_vars):
    results = {}
    # Creates a dictionary with arrays for each variable, starting with n (number_predictions) empty lists
    for variable in available_vars:
        results[variable] = []
        for i in range(len(pr_test) + number_predictions - 1):
            results[variable].append([])

    # Test result shapes differ according to model
    # All shapes must be (number of timesteps, 1, number of predicted values)
    pr_test = pr_test.reshape((pr_test.shape[0], 1, pr_test.shape[-1]))
    # Order of columns: var_1+1, ... var_n+1, var_1+2, ..., var_n+2, ..., var_1+t, ..., var_n+t
    for i in range(len(pr_test)):
        res = pr_test[i][0]

        # For each observation window (one timestep with all variables)
        for timestep in range(0, len(res), len(available_vars)):
            # For each variable within each timestep
            for variable_step in range(timestep, timestep + len(available_vars)):
                # Add new empty list for each prediction for a new index
                # Index = timestep // number of available variables
                # All calculations are stored, alternatively, one can only store one calculation
                results[available_vars[variable_step - timestep]][i + (timestep // len(available_vars))].append(res[variable_step])

    # Loop through all predicted observations, calculate average and store in a DataFrame
    results_pd = pandas.DataFrame()
    for variable in available_vars:
        for i in range(len(results[variable])):
            results[variable][i] = sum(results[variable][i]) / len(results[variable][i])
        results_pd[variable] = results[variable]

    return results_pd

# Creates an array with original amount of columns, undoes normalization and extracts columns
def mask_results(res, forecast_vars, output_vars, scaler):
    masked = []
    temp = np.zeros((len(res), len(forecast_vars)))
    for i in range(len(forecast_vars)):
        for j in range(len(output_vars)):
            if forecast_vars[i] == output_vars[j]:
                masked.append(i)
                temp[:, i] = res[:, j]
    return scaler.inverse_transform(temp)[:, masked]

def process_results_2(scaler, results_train, results_test, number_predictions, forecast_vars, output_vars=None):
    if (not output_vars is None) and output_vars != []:
        results_train = get_predictions(results_train, number_predictions, output_vars)
        results_train_denormalized = pandas.DataFrame(mask_results(results_train.values, forecast_vars, output_vars, scaler))
        results_train_denormalized.columns = output_vars
        results_test = get_predictions(results_test, number_predictions, output_vars)
        results_test_denormalized = pandas.DataFrame(mask_results(results_test.values, forecast_vars, output_vars, scaler))
        results_test_denormalized.columns = output_vars
    else:
        results_train = get_predictions(results_train, number_predictions, forecast_vars)
        results_train_denormalized = pandas.DataFrame(scaler.inverse_transform(results_train))
        results_train_denormalized.columns = forecast_vars
        results_test = get_predictions(results_test, number_predictions, forecast_vars)
        results_test_denormalized = pandas.DataFrame(scaler.inverse_transform(results_test))
        results_test_denormalized.columns = forecast_vars
    return results_train_denormalized, results_test_denormalized

# Forecasts interval of size foecast_window using given model and last observation
# Uses Monte Carlo Dropout for confidence interval estimation
def forecast_nn(model, forecast_window, inp, t, available_vars, number_predictions=1):
    predictions = []
    inp = np.reshape(inp.values, (1, inp.shape[0], inp.shape[1]))
    # Number of iterations until forecast size is equal or greater than forecast_window
    for i in range(forecast_window - number_predictions + 1):
        current_pred = []
        # Predict the same value t times (Monte Carlo Dropout)
        for j in range(t):
            current_pred.append(model.predict(inp).flatten())
        predictions.append(current_pred)
        # Add new predicted value to input
        mean = np.array(current_pred).mean(axis=0)
        mean = mean.reshape((1, 1, mean.shape[0]))
        # Only add earliest predicted observation (var+1)
        inp = np.concatenate((mean[:, :, :len(available_vars)], inp), axis=2)
        # Remove early observation for next input
        inp = inp[:, :, :-len(available_vars)]
    return predictions

def process_nn_predictions(scaler, predicted, number_predictions, available_vars, output_vars=None):
    pred_array = np.array(predicted)
    forecast_data = pandas.DataFrame()
    conf_int_data = pandas.DataFrame()
    means = pred_array.mean(axis=1)
    std_dev = pred_array.std(axis=1)
    means = means.reshape((means.shape[0], 1, means.shape[1]))
    std_dev = std_dev.reshape((std_dev.shape[0], 1, std_dev.shape[1]))
    if (not output_vars is None) and output_vars != []:
        means = get_predictions(means, number_predictions, output_vars)
        std_dev = get_predictions(std_dev, number_predictions, output_vars)
        lower_conf = mask_results((means - 2 * std_dev).values, available_vars, output_vars, scaler)
        upper_conf = mask_results((means + 2 * std_dev).values, available_vars, output_vars, scaler)
        means = mask_results(means.values, available_vars, output_vars, scaler)
        for i in range(len(output_vars)):
            forecast_data[output_vars[i]] = means[:, i]
            conf_int_data[output_vars[i] + '_lower_conf'] = lower_conf[:, i]
            conf_int_data[output_vars[i] + '_upper_conf'] = upper_conf[:, i]
    else:
        means = get_predictions(means, number_predictions, available_vars)
        std_dev = get_predictions(std_dev, number_predictions, available_vars)
        lower_conf = scaler.inverse_transform(means - 2 * std_dev)
        upper_conf = scaler.inverse_transform(means + 2 * std_dev)
        means = scaler.inverse_transform(means)
        for i in range(len(available_vars)):
            forecast_data[available_vars[i]] = means[:, i]
            conf_int_data[available_vars[i] + '_lower_conf'] = lower_conf[:, i]
            conf_int_data[available_vars[i] + '_upper_conf'] = upper_conf[:, i]

    return forecast_data, conf_int_data

def get_nn_gradients(model, data):
    inp = tf.Variable(data, dtype=tf.float32)

    with tf.GradientTape() as tape:
        predictions = model(inp)

    grads = tape.gradient(predictions, inp)
    grads = tf.reduce_mean(grads, axis=1).numpy()[0]
    return grads
