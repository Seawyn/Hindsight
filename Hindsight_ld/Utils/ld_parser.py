import copy
import numpy as np
import pandas
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

# Handle version difference
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

# Returns all subjects in the data
def subjects(data, subject_col):
    return list(set(data[subject_col]))

# Returns a dataframe with data of a given subject
def get_subject(data, subject_id, subject_col):
    return data.where(data[subject_col] == subject_id).dropna(how='all')

# Replaces values in a given column
def replace_in_col(data, col, repl):
    return data.replace({col: repl})

# Quantile discretization using Pandas's qcut method
def quantile_discretize(data, qd_var, qd_size, qd_enc):
    new_data = copy.deepcopy(data)
    encode = None
    if qd_enc == True:
        encode = False
    new_data[qd_var] = pandas.qcut(new_data[qd_var], qd_size, labels=encode)
    return new_data

# Presents a list of variables of the data, followed by general statistics
def summary(data, subject=None):
    if not subject is None:
        data = get_subject(subject)

    col_data = {}
    for col in data.columns:
        number_of_nan = data[col].isna().sum()
        number_of_distinct = len(pandas.DataFrame(set(data[col])).dropna())
        col_str = '|' + col + '| | \n'
        col_str += '| :------- | -------: | \n'
        col_str += '| Observations |' + str(len(data[col])) + '| \n'
        col_str += '| Missing values |' + str(number_of_nan) + '| \n'
        col_str += '| Missing values (%) |' + str(round(number_of_nan / len(data[col]), 3)) + '| \n'
        col_str += '| Distinct values |' + str(number_of_distinct) + '| \n'
        col_str += '| Distinct values (%) |' + str(round(number_of_distinct / len(data[col]), 3)) + '| \n'

        col_data[col] = col_str

    return col_data

# Returns a dictionary of missigness percentage for each column of the input
def get_col_missingness(data):
    mi = {}
    for col in data.columns:
        number_of_nan = data[col].isna().sum()
        mi[col] = number_of_nan / len(data[col])
    return mi

# Detects and drops columns of data with percentage of missing values above 30%
def drop_cols_with_high_missingness(data, subject_col, threshold=0.3):
    for col in data.columns:
        if col == subject_col:
            continue
        percentage_of_nan = data[col].isna().sum() / len(data[col])
        if percentage_of_nan > threshold:
            data = data.drop(columns=[col])
            print("Dropped", col, " column")
    return data

# Label encodes columns received as input and returns results along with a mapping dictionary and encoder
# Only accepts a single column as input
def label_encode(inp):
    data = copy.deepcopy(inp)
    # Ensure input is a single dataframe column (Pandas Series)
    assert isinstance(data, pandas.Series)

    # if input has missing values
    has_missing = False
    if data.isnull().any():
        # Change missing values to string ('NaN')
        data[pandas.isnull(data)] = 'NaN'
        has_missing = True

    # Create and fit Label Encoder
    label = LabelEncoder()
    label.fit(data)
    encoded = pandas.Series(label.transform(data))

    # Map with encoded values as input and original values as output
    mapping = dict(zip(label.transform(label.classes_), label.classes_))

    # Convert encoded 'NaN' values back to np.nan
    if has_missing:
        encoded = encoded.replace(label.transform(['NaN'])[0], np.nan)

    return encoded, mapping, label


# Imputation based on Last Observation Carried Forward (LOCF)
def locf_impute(data, subject=None, cols=None, nocb=False):
    # If subject is null, perform for all subjects
    if subject is None:
        subject = subjects(data, data.columns[0])
    # if cols is none, perform for all columns (except first column)
    if cols is None:
        cols = data.columns[1:]
    for s in subject:
        # Get subject related data and filter columns
        current_data = get_subject(data, s, data.columns[0]).loc[:, cols]
        # Perform Forward Fill
        current_data = current_data.ffill()
        if nocb:
            current_data = current_data.bfill()
        data.loc[current_data.index, cols] = current_data
    return data

# Imputation of discrete variables based on MissForest
# Automatically encodes string columns (must be included in categorical_variables)
def mf_impute(inp, subject=None, cols=None, categorical_variables=None):
    data = copy.deepcopy(inp)
    # Prepare input
    # if cols is none, perform for all columns (except first column)
    if cols is None:
        cols = data.columns[1:]
    # If subject is null, perform for all subjects
    if subject is None:
        inp = data[cols]
    else:
        # Create a dataframe with all selected subjects
        inp = pandas.DataFrame()
        for s in subject:
            inp = inp.append(get_subject(data, s, data.columns[0]).loc[:, cols])
    if len(inp.columns) < 2:
        raise Exception("Multiple variables must be given as input")

    # Encode string columns
    # Note: only categorical variables are encoded
    if not categorical_variables is None:
        labels = {}
        for col in categorical_variables:
            if inp[col].dtype == np.dtype(object):
                encoded, mapping, label = label_encode(inp[col])
                # Convert string column to encoded result
                inp[col] = encoded
                labels[col] = label

    else:
        labels = {}

    # Prepare MissForest Imputer
    imputer = MissForest()
    cat_vars = None
    if not categorical_variables is None:
        cat_vars = []
        for categorical_variable in categorical_variables:
            cat_vars.append(list(inp.columns).index(categorical_variable))

    # Fit and Transform the input
    res = imputer.fit_transform(inp.values, cat_vars=cat_vars)
    res = pandas.DataFrame(res, index=inp.index, columns=inp.columns)

    # Convert encoded columns back to strings
    for col in labels.keys():
        res[col] = labels[col].inverse_transform(res[col].astype(int))

    data.loc[res.index, res.columns] = res
    return data

# Imputation of discrete variables using Scikit-Learn's Iterative Imputer (based in MICE)
def iter_impute(data, subject=None, cols=None, rounding=3, max_iter=10):
    # Prepare input
    # if cols is none, perform for all columns (except first column)
    if cols is None:
        cols = data.columns[1:]
    # If subject is null, perform for all subjects
    if subject is None:
        inp = data[cols]
    else:
        # Create a dataframe with all selected subjects
        inp = pandas.DataFrame()
        for s in subject:
            inp = inp.append(get_subject(data, s, data.columns[0]).loc[:, cols])
    if len(inp.columns) < 2:
        raise Exception("Multiple variables must be given as input")

    # Create imputer
    imputer = IterativeImputer(max_iter=max_iter)
    imputer.fit(inp)

    # Impute missing values and round to the third decimal point
    res = pandas.DataFrame(np.round(imputer.transform(inp), decimals=rounding), index=inp.index, columns=inp.columns)

    data.loc[res.index, res.columns] = res
    return data
