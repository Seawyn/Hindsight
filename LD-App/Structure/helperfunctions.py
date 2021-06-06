import base64
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import io
from Utils.ld_parser import *
import pandas

# Checks if uploaded file is a valid .csv file
def valid_upload(filename):
    if len(filename.split('.')) != 2:
        return False
    (name, type) = filename.split('.')
    return (type == 'csv') and (len(name) > 0)

# Receives encoded upload and returns a dataframe
def read_upload(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pandas.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df

# Sets up dataset table and dropdown menus after initial load
def setup_dataset(data, add_all=True):
    df = pandas.read_json(data).sort_index()

    # Update columns dropout
    num_default_cols = 5
    col_options = []
    col_values = df.columns[:num_default_cols]
    for col in df.columns:
        col_options.append({'label': col, 'value': col})

    # Update subjects dropout
    ind_options = []
    if add_all:
        ind_options.append({'label': 'All', 'value': 'all'})
    for subject in subjects(df, df.columns[0]):
        ind_options.append({'label': subject, 'value': subject})

    return col_options, col_values, ind_options

# Returns a dictionary containing columns related to each degree of missingness
def column_missingness(data):
    mi = get_col_missingness(data)
    missing_status = {'none': [], 'moderate': [], 'high': []}
    for col in data.columns:
        if mi[col] == 0:
            missing_status['none'].append(col)
        elif mi[col] <= 0.2:
            missing_status['moderate'].append(col)
        else:
            missing_status['high'].append(col)
    return missing_status

# Returns info related to each column
def get_column_info(data):
    tooltip_col_info = {}
    col_info = summary(data)
    for col in col_info.keys():
        tooltip_col_info[col] = {'value': col_info[col], 'type': 'markdown'}
    return tooltip_col_info

# Returns a list of subjects and columns with missing values
def check_missingness(data):
    subjects_mv = []
    cols_mv = []

    # Check subjects with missing values
    for subject in subjects(data, data.columns[0]):
        current_data = get_subject(data, subject, data.columns[0])
        number_of_nan = sum(current_data.isna().sum())
        if number_of_nan > 0:
            subjects_mv.append(subject)

    # Check columns with missing values
    nan_info = data.isna().sum()
    for i in range(len(nan_info)):
        # nan_info[i]: number of missing values
        # nan_info.index[i]: related column
        if nan_info[i] > 0:
            cols_mv.append(nan_info.index[i])

    return subjects_mv, cols_mv

# Replace values in column of given data by a given replacement (number)
def replace_in_dataset(data, col, val, replacement):
    df = pandas.read_json(data).sort_index()
    repl = {val: replacement}
    return replace_in_col(df, col, repl)

# Handles imputation behaviour
def impute_ld_dataset(method, data, chosen_subjects, variables, all_subjects=False, discrete=None):
    # Checking 'Apply to all subjects' imputes data in all subjects
    if all_subjects:
        chosen_subjects = subjects(data, data.columns[0])
    # Last Observation Carried Forward
    if method == 'locf':
        return locf_impute(data, chosen_subjects, variables)
    elif method == 'missforest':
        return mf_impute(data, chosen_subjects, cols=variables, categorical_variables=discrete)
    # This shouldn't happen
    else:
        raise ValueError('Unknown imputation method')
