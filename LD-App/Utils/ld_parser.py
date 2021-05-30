import pandas
import sys

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

        '''
        print(col)
        print('Number of observations:', len(data[col]))
        print('Number of missing values:', number_of_nan)
        print('Percentage of missing values:', number_of_nan / len(data[col]))
        print('Number of distinct observations:', number_of_distinct)
        print('Percentage of distinct values:', number_of_distinct / len(data[col]))
        print('-----------------')
        '''

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

# Imputation based on Last Observation Carried Forward (LOCF)
def locf_impute(data, subject=None, cols=None):
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
        data.loc[current_data.index, cols] = current_data
    return data

# Imputation of discrete variables based on MissForest
def mf_impute(data, imp_vars=None, categorical_variables=None):
    # Prepare input
    inp = data
    if not imp_vars is None:
        inp = data[imp_vars]
    if len(inp.columns) < 2:
        raise Exception("Multiple variables must be given as input")

    # Prepare MissForest Imputer
    imputer = MissForest()
    if not categorical_variables is None:
        cat_vars = []
        for categorical_variable in categorical_variables:
            cat_vars.append(list(inp.columns).index(categorical_variable))

    # Fit and Transform the input
    res = imputer.fit_transform(inp.values, cat_vars=cat_vars)
    return res
