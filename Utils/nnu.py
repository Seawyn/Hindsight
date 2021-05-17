import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from Utils.tsu import calculate_mae, calculate_residuals
import warnings

# Receives a time series dataset and creates training, test and, optionally, validation sets
# data: dataset
# order: number of previous timesteps to be included in the input (0 or higher)
# split: index that separates training and test sets
# val_split: size of validation set (0 or higher)
# step: offset between x and y values (1 or higher)
def series_to_supervised(data, order, split, val_split=0, step=1):
  df = pandas.DataFrame(data)
  for col in df.columns:
    df = df.rename(columns = {col: 'var' + str(col)})
  df_shift = df.shift(-step).dropna()

  if val_split == 0:
    val_x_split = -len(df)
    val_y_split = -len(df_shift)
    warnings.warn("x_val and y_val are empty DataFrames.")
  else:
    val_x_split = val_split
    val_y_split = val_split

  # (split - step) is volatile (used to ensure test size is equal to other models)

  y_train = df_shift[df_shift.columns][:split - step]
  y_test = df_shift[df_shift.columns][split - step:-val_y_split]
  y_val = df_shift[df_shift.columns][-val_y_split:]

  shifts = {}

  if order > 0:
    for i in range(1, order + 1):
      shifts['-' + str(i)] = df.shift(i)
  for key in shifts.keys():
    for col in shifts[key].columns:
      df[str(col) + key] = shifts[key][col]

  df = df.drop(df.index[-step:])
  x_train = df[df.columns][:split - step]
  x_test = df[df.columns][split - step:-val_x_split]
  x_val = df[df.columns][-val_x_split:]

  x_train = x_train.fillna(0)
  x_test = x_test.fillna(0)

  sets = {}
  sets['x_train'], sets['x_test'] = x_train, x_test
  sets['y_train'], sets['y_test'] = y_train, y_test
  sets['x_val'], sets['y_val'] = x_val, y_val

  return sets

# Preprocesses the dataset: normalization, train-test split, reshaping
# Returns scaler (for later denormalization) and processed data
def preprocess(data, order, split, val_split=0, step=1):
    # TODO: reshape x_val and y_val

    values = data.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)

    sets = series_to_supervised(scaled, order, split, val_split=val_split, step=step)

    x_train = np.reshape(sets['x_train'].values, (sets['x_train'].shape[0], 1, sets['x_train'].shape[1]))
    x_test = np.reshape(sets['x_test'].values, (sets['x_test'].shape[0], 1, sets['x_test'].shape[1]))
    y_train = sets['y_train'].values
    y_test = sets['y_test'].values

    processed = {}
    processed['x_train'], processed['x_test'] = x_train, x_test
    processed['y_train'], processed['y_test'] = y_train, y_test
    processed['x_val'], processed['y_val'] = sets['x_val'], sets['y_val']

    return scaler, processed

# Creates and compiles a Bidirectional LSTM neural network
def create_bilstm_model(x_train, num_nodes=(20, 20), activations=('relu', 'relu'), dropout=0.2, output_size=1, loss='mae'):
    lstm_model = keras.models.Sequential()

    # Return sequences will be True if there is no second layer
    lstm_model.add(layers.Bidirectional(layers.LSTM(num_nodes[0], return_sequences=(num_nodes[1] is None), activation=activations[0]), input_shape=(x_train.shape[1], x_train.shape[2])))

    # Only create second layer if there are hidden nodes
    if not num_nodes[1] is None:
        lstm_model.add(layers.RepeatVector(1))
        lstm_model.add(layers.Bidirectional(layers.LSTM(num_nodes[1], return_sequences=True, activation=activations[1]), input_shape=(x_train.shape[1], x_train.shape[2])))

    # Dropout and output
    lstm_model.add(layers.Dropout(dropout))
    lstm_model.add(layers.TimeDistributed(layers.Dense(output_size)))

    lstm_model.compile(loss=loss, optimizer='adam')
    return lstm_model

# Creates and compiles a CNN neural network
def create_cnn_model(x_train, num_filters=(32, 16), kernel_sizes=(4, 2), activations=('relu', 'relu'), dropout=0, output_size=1, loss='mae'):
    cnn_model = keras.models.Sequential()

    cnn_model.add(layers.Conv1D(filters=num_filters[0], kernel_size=kernel_sizes[0], activation=activations[0], padding='same', input_shape=(x_train.shape[1], x_train.shape[2])))

    # Only create second layer if there are hidden nodes
    if not num_filters[1] is None:
        cnn_model.add(layers.Conv1D(filters=num_filters[1], kernel_size=kernel_sizes[1], activation=activations[1], padding='same'))

    # Flatten and middle Dense layer
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(units=32, activation='relu'))

    # Dropout and output
    cnn_model.add(layers.Dropout(dropout))
    cnn_model.add(layers.Dense(units=output_size))

    cnn_model.compile(optimizer='adam', loss='mae')
    return cnn_model

# Trains received model and outputs its test set prediction and error
def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=10, seed=0):
    # TODO: Change parameters to dictionary of sets
    tf.random.set_seed(seed)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=False)
    pr_test = model.predict(x_test)
    pr_train = model.predict(x_train)
    return pr_test, pr_train, history

# Reshapes and denormalizes prediction results
def process_results(scaler, pr_test, forecast_vars, y_test):
    pr_test_reshaped = pr_test.reshape(len(pr_test), len(forecast_vars))
    predicted = pandas.DataFrame(scaler.inverse_transform(pr_test_reshaped))
    predicted.columns = forecast_vars
    y_test_reshaped = y_test.reshape(len(y_test), len(forecast_vars))
    test = pandas.DataFrame(scaler.inverse_transform(y_test_reshaped))
    test.columns = forecast_vars
    return predicted, test

# Performs the entire neural network regression workflow: preprocessing, training, testing, result processing
def neural_network_regression(model, data, order, split, hyperparam_data, val_split=0, step=1, output_size=1, seed=None):
    # TODO: Update parameters: validation sets, etc
    # TODO: Size of output is larger than 1

    # Preprocess data
    forecast_vars = data.columns
    scaler, sets = preprocess(data, order, split, val_split=val_split, step=step)

    # Fix seed for replicability
    if seed is None:
        seed = np.random.randint(2 ** 16)

    # Create model
    if model == 'Bi-LSTM':
        num_nodes = (hyperparam_data['lstm-first-layer-nodes'], hyperparam_data['lstm-second-layer-nodes'])
        activations = (hyperparam_data['lstm-first-layer-activation'], hyperparam_data['lstm-second-layer-activation'])
        model = create_bilstm_model(sets['x_train'], num_nodes=num_nodes, activations=activations, output_size=output_size)
    elif model == 'CNN':
        num_filters = (hyperparam_data['cnn-first-layer-filters'], hyperparam_data['cnn-second-layer-filters'])
        kernel_sizes = (hyperparam_data['cnn-first-layer-kernel'], hyperparam_data['cnn-second-layer-kernel'])
        activations = (hyperparam_data['cnn-first-layer-activation'], hyperparam_data['cnn-second-layer-activation'])
        model = create_cnn_model(sets['x_train'], num_filters=num_filters, kernel_sizes=kernel_sizes, activations=activations, output_size=output_size)
    else:
        # This shouldn't happen
        raise ValueError('Unknown Model')

    # Train model
    pr_test, pr_train, history = train_model(model, sets['x_train'], sets['y_train'], sets['x_test'], sets['y_test'], seed=seed)

    # Obtain prediction, error and residuals
    predicted, test = process_results(scaler, pr_test, forecast_vars, sets['y_test'])
    errors = {}
    for variable in forecast_vars:
        errors[variable] = calculate_mae(test[variable].values, predicted[variable].values)
    residuals = calculate_residuals(test.values, predicted.values)

    keras.backend.clear_session()

    return predicted, seed, errors, residuals, history, pr_train, sets['y_train']
