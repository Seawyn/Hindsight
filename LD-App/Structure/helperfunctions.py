import base64
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import io
import plotly.express as px
import plotly.graph_objs as go
import pandas
from Utils.bmu import *
from Utils.ld_parser import *
from Utils.ldnnu import *
import warnings
try:
    from Utils.dybmu import *
except ImportError as e:
    warnings.warn('Pydybm is not installed!')

# Layout settings for all plot legends
fig_legend_layout = {
    'orientation': 'h',
    'yanchor': 'bottom',
    'y': 1.02,
    'xanchor': 'right',
    'x': 1
}

# Layout settings for all plot margins
fig_margin_layout = {
    'l': 10,
    'r': 10,
    'b': 10,
    't': 0
}

# Check if Binary Dynamic Boltzmann Machine has been imported
def check_pydybm():
    try:
        BinaryDyBM
        return False
    except NameError:
        return True

# Checks if uploaded file is a valid .csv file
def valid_upload(filename):
    if len(filename.split('.')) != 2:
        return False
    (name, type) = filename.split('.')
    return (type == 'csv') and (len(name) > 0)

# Automatically find delimiter of the input
def find_delimiter(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # Credit
    # https://stackoverflow.com/questions/45732459/retrieve-delimiter-infered-by-read-csv-in-pandas/45732580
    reader = pandas.read_csv(io.StringIO(decoded.decode('utf-8')), sep=None, iterator=True)
    sep = reader._engine.data.dialect.delimiter
    return sep

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

# Quantile discretization
def quantile_discretize_dataset_by_var(data, qd_var, qd_size, qd_enc):
    encode = False
    if not qd_enc is None and qd_enc == ['encode']:
        encode = True
    new_data = quantile_discretize(data, qd_var, qd_size, encode)
    if not encode:
        # Change intervals to strings
        new_data[qd_var] = new_data[qd_var].astype('string')
    return new_data

# Returns an empty plot with a given message
def get_empty_plot(message, width=None, height=None):
    fig = go.Figure()
    fig.update_layout({
        'xaxis': {
            'visible': False
        },
        'yaxis': {
            'visible': False
        },
        'annotations': [
            {
                'text': message,
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {
                    'size': 28
                }
            }
        ]
    })
    if not height is None:
        fig.update_layout({
            'height': height
        })
    if not width is None:
        fig.update_layout({
            'width': width
        })

    fig.update_layout(template='ggplot2', margin={'l': 10, 'r': 10, 't': 10, 'b': 10})

    return fig

# Receives data as well as other parameters and fits a Restricted Boltzmann Machine
# Returns trained model and number of input nodes for each variable
def bm_workflow(data, chosen_subjects, use_all, cols, n_hidden, iter, l_r):
    df = pandas.read_json(data).sort_index()
    # If apply to all subjects was selected
    if use_all == ['apply']:
        chosen_subjects = subjects(df, df.columns[0])
    # Create a dataframe with all selected subjects
    inp = pandas.DataFrame()
    for s in chosen_subjects:
        inp = inp.append(get_subject(df, s, df.columns[0]).loc[:, cols])

    inp, input_size = process(inp)
    model = train_rbm(inp, n_hidden, iter, l_r)
    return model, input_size

def dybm_workflow(data, chosen_subjects, use_all, cols, delay, decay, iter, out_vars, epochs, test_set=None):
    df = pandas.read_json(data).sort_index()
    id_col = df.columns[0]

    if test_set is None:
        test_set = []

    # Filter subjects
    if use_all != ['apply']:
        temp_data = pandas.DataFrame()
        for s in list(set(subjects + test_set)):
            current = get_subject(df, s, id_col)
            temp_data = temp_data.append(current, ignore_index=True)
            df = temp_data

    all_cols = [id_col] + cols + out_vars

    b_df, new_cols = parse_data(df[all_cols], cols=cols, ignore_first=True)
    _, new_out = parse_data(df[out_vars], cols=out_vars)

    # Create and Train DyBM
    train_data = pandas.DataFrame()
    test_data = pandas.DataFrame()
    for s in subjects(b_df, b_df.columns[0]):
        current = get_subject(b_df, s, b_df.columns[0])
        if s in test_set:
            test_data = test_data.append(current, ignore_index=True)
        else:
            train_data = train_data.append(current, ignore_index=True)
    if test_data.empty:
        test_data = None

    model, performances = dybm_procedure(train_data, new_cols, new_out, delay, decay=[decay], iterations=epochs, id_col=id_col, test_dataset=test_data)

    performances_df = pandas.DataFrame()
    performances_df['Training'] = performances['training']
    if test_set != []:
        performances_df['Validation'] = performances['validation']

    return model, new_cols, new_out, performances_df

def nn_workflow(data, subjects, use_all, inp_vars, out_vars, ws, dropout, epochs, test_set):
    df = pandas.read_json(data).sort_index()
    id_col = df.columns[0]

    if test_set is None:
        test_set = []

    # Filter subjects
    if use_all != ['apply']:
        temp_data = pandas.DataFrame()
        for s in list(set(subjects + test_set)):
            current = get_subject(df, s, id_col)
            temp_data = temp_data.append(current, ignore_index=True)
            df = temp_data

    # Create and train CNN and obtain performances and gradients
    performances, gradients = nn_procedure(df, inp_vars, out_vars, ws, id_col, test_set=test_set, dropout=dropout, epochs=epochs)

    performances_df = pandas.DataFrame()
    performances_df['Training'] = performances['accuracy']
    if test_set != []:
        performances_df['Validation'] = performances['val_accuracy']

    return performances_df, gradients
