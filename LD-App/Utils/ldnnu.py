import numpy as np
import pandas
import scipy
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras import regularizers
from Utils.ld_parser import *

# Shifts data with a given offset and renames columns
def get_shifted_data(data, offset):
  shifted = data.shift(offset)
  # Rename each column to column-offset (for example: col-1)
  columns = list(shifted.columns)
  for i in range(len(columns)):
    columns[i] = columns[i] + '-' + str(offset)
  shifted.columns = columns
  return shifted

# Preprocess data as input for neural network
# Uses sliding window method
# Mind the break points between subjects
# Receives data, window size, test size and columns (which are all, by default)
# Testing set is composed of ts entries from each subject data
# Conditions:
#   - ws must be an integer equal or above 1
#   - cols must be 'all' or an array containing data column names
def preprocess(dt, ws, inp_vars, out_var, id_col):
  data = copy.deepcopy(dt)

  training_set = pandas.DataFrame()
  y_training_set = pandas.DataFrame()

  sbjs_order = []

  # Normalize all input columns
  values = data[inp_vars].values
  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaled = scaler.fit_transform(values)
  data[inp_vars] = scaled

  for s in subjects(data, id_col):
    # Fetch and format subject data
    s_data = get_subject(data, s, id_col)
    y = pandas.DataFrame(s_data[out_var])
    s_data = s_data[inp_vars]
    # Shift dataset one index forward and append columns according to the window size
    # ws = 1: no shift; ws = 2: one shift; and so on...
    shifted = []
    for i in range(ws - 1):
      shifted.append(get_shifted_data(s_data, i + 1))
    # Add all stored shifted data to current data
    for el in shifted:
      s_data[el.columns] = el

    for row_cursor in range(len(s_data)):
        sbjs_order.append(s)

    training_set = training_set.append(s_data, ignore_index=True)
    y_training_set = y_training_set.append(y, ignore_index=True)

    sets = {'x_train': training_set, 'y_train': y_training_set, 'sbjs': sbjs_order}

  sets['x_train'] = sets['x_train'].fillna(0)

  return sets


# Creates a CNN model for classifying given data
def create_cnn(x_train, y_train, filters=8, k_size=2, activation='elu', padding='same', pool_size=2, dropout=0.5, seed=42):
  np.random.seed(seed)
  tf.random.set_seed(seed)

  model = keras.models.Sequential([
    layers.Conv1D(filters=filters, kernel_size=k_size, activation='elu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2])),
    layers.MaxPool1D(pool_size=pool_size),
    layers.Flatten(),
    layers.Dropout(dropout),
    layers.Dense(units=len(set(y_train.flatten())))
  ])

  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

  return model


# Trains model and returns accuracy and val_accuracy, as well as gradients
def train_cnn(model, x_train, y_train, x_test=None, y_test=None, epochs=50):
  if x_test is None or y_test is None:
    validation_data = None
  else:
    validation_data = (x_test, y_test)
  history = model.fit(x_train, y_train, epochs=epochs, validation_data=validation_data)

  # Get accuracies
  performances = {}
  performances['accuracy'] = history.history['accuracy']
  if not validation_data is None:
    performances['val_accuracy'] = history.history['val_accuracy']

  # Get gradients
  inp = tf.Variable(x_train, dtype=tf.float32)

  with tf.GradientTape() as tape:
    predictions = model(inp)

  grads = tape.gradient(predictions, inp)
  grads = tf.reduce_mean(grads, axis=1).numpy()[0]

  return performances, grads

# Receives data and parameters, and performs entire neural network worfklow
def nn_procedure(data, inp_vars, out_var, window_size, id_col, test_set=[], filters=8, k_size=2, activation='elu', padding='same', dropout=0.5, epochs=50, seed=42):
  if test_set != []:
    train_data = pandas.DataFrame()
    test_data = pandas.DataFrame()
    for s in subjects(data, id_col):
      current = get_subject(data, s, id_col)
      if s in test_set:
        test_data = test_data.append(current, ignore_index=True)
      else:
        train_data = train_data.append(current, ignore_index=True)
    train_sets = preprocess(train_data, window_size, inp_vars, out_var, id_col)
    test_sets = preprocess(test_data, window_size, inp_vars, out_var, id_col)
    x_test, y_test = test_sets['x_train'], test_sets['y_train']
    x_test = np.reshape(x_test.values, (x_test.shape[0], window_size, x_test.shape[1] // window_size))
    y_test = y_test.values.flatten()
  else:
    train_sets = preprocess(data, window_size, inp_vars, out_var, id_col)
    x_test, y_test = None, None

  x_train, y_train = train_sets['x_train'], train_sets['y_train']
  x_train = np.reshape(x_train.values, (x_train.shape[0], window_size, x_train.shape[1] // window_size))
  y_train = y_train.values.flatten()

  model = create_cnn(x_train, y_train, filters=filters, k_size=k_size, activation=activation, padding=padding, pool_size=window_size, dropout=dropout, seed=seed)

  performances, gradients = train_cnn(model, x_train, y_train, x_test=x_test, y_test=y_test, epochs=epochs)

  return performances, gradients
