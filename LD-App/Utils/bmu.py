import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Trains a Bernoulli Restricted Boltzmann Machine using Scikit Learn's implementation
def train_rbm(data, n_hidden, n_iter=10, learning_rate=0.1):
    model = BernoulliRBM(n_components=n_hidden)

    # Set hyperparameters
    model.n_iter = n_iter
    model.learning_rate = learning_rate

    # Train model
    model.fit(data)

    return model

# Receives a dataset and one-hot encodes columns with multiple values
# Returns a dictionary with all columns encoded
def encode(data):
    processed = {}
    col_missingness = data.isnull().any()
    onehotencoder = OneHotEncoder(sparse=False)
    label = LabelEncoder()
    for col in data.columns:
        # If column has missing values
        if col_missingness[col]:
            raise ValueError('Input has missing values')
        # Column is not binary
        if len(list(set(data[col]))) > 2:
            processed[col] = onehotencoder.fit_transform(data[col].values.reshape(-1, 1))
        # Column is binary (encode using label encoder)
        # Label encoder converts any set of binary values to [0, 1]
        else:
            processed[col] = label.fit_transform(data[col]).reshape(-1, 1)
    return processed

# Receives a dataset and reshapes one-hot encoded output into a dataframe for a model
def process(data):
    try:
        encoded = encode(data)
    except ValueError:
        raise ValueError('Input has missing values')
    else:
        # Obtain size of input for each encoded column
        input_sizes = []
        for key in encoded.keys():
            input_sizes.append(encoded[key].shape[-1])
        processed = None
        # Create all rows
        for i in range(data.shape[0]):
            row = []
            # Append each encoded value to the current row
            for col in encoded.keys():
                row = np.append(row, encoded[col][i])
            row = row.reshape((1, len(row)))
            if processed is None:
                processed = row
            else:
                # Append new row to current output
                processed = np.append(processed, row, axis=0)
        return processed, input_sizes
