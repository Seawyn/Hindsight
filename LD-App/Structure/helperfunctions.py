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
def setup_dataset(data):
    df = pandas.read_json(data).sort_index()

    # Update columns dropout
    num_default_cols = 5
    col_options = []
    col_values = df.columns[:num_default_cols]
    for col in df.columns:
        col_options.append({'label': col, 'value': col})

    # Update subjects dropout
    ind_options = [{'label': 'All', 'value': 'all'}]
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

def get_column_info(data):
    tooltip_col_info = {}
    col_info = summary(data)
    for col in col_info.keys():
        tooltip_col_info[col] = {'value': col_info[col], 'type': 'markdown'}
    return tooltip_col_info
