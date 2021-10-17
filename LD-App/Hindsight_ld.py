import dash
from dash.dash import no_update
import dash_table
from Structure.modals import *

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    dbc.Card([
        # Import dataset (only upon startup)
        dbc.CardBody([
            # Adds space that does not change
            dbc.Row(style={'height': '15vh'}),
            dbc.Row([
                # Columns automatically resize the Import Dataset Card
                dbc.Col(),
                dbc.Col(
                    dbc.Card(children=[
                        dbc.CardHeader(
                            html.H5('Import Dataset', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            'Please select a valid .csv file:',
                            dcc.Upload(
                                id='upload-ld',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select File', style={'text-decoration': 'underline', 'cursor': 'pointer'})
                                ]),
                                style={
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'marginTop': '10px',
                                    'marginBottom': '10px'
                                }
                            ),
                            html.Div('No dataset has been selected', id='current-filename'),
                            html.Br(),
                            html.Div([
                                'Suggested separator:',
                                dbc.Badge('None', id='suggested-separator', className='mr-1', style={'marginLeft': '10px'})
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col('Separator:', width=3),
                                dbc.Col(
                                    dcc.RadioItems(
                                        options=[
                                            {'label': 'Comma (,)', 'value': ','},
                                            {'label': 'Semicolon (;)', 'value':';'},
                                            {'label': 'Tab (\\t)', 'value': '\t'},
                                        ],
                                        value=',',
                                        id='data-sep',
                                        inputStyle={'margin-right': '5px'},
                                        labelStyle={'margin-right': '20px'}
                                    ), width=8
                                )
                            ], justify='between'),
                        ]),
                        dbc.CardFooter(dbc.Button('Confirm', id='upload-confirm', disabled=True, style={'backgroundColor': '#58B088', 'border': 'none'})),
                    ])
                ),
                dbc.Col()
            ])
        ], id='upload-screen', style={'height': '100vh'}),
        # Dashboard main page (after dataset has been imported)
        dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    dbc.Card(children=[
                        dbc.CardHeader(
                            html.H5('Raw dataset', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            dbc.Table.from_dataframe(pandas.DataFrame(), style={'height': '100%'})
                        ], id='raw-data', style={'overflow': 'scroll', 'maxHeight': '100%'})
                    ], style={'height': '100%'})
                , width=10, style={'height': '100%'}),
                dbc.Col(children=[
                    dbc.Card(children=[
                        dbc.CardHeader(
                            html.H5('Info and Operations', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            dataframe_modal,
                            html.Br(),
                            ld_imputation_modal,
                        ])
                    ], style={'height': '100%'})
                ], width=2, style={'height': '100%'})
            ], style={'height': '50vh'}),
            html.Br(),

            dbc.Row([
                dbc.Col(
                    dbc.Card(children=[
                        dbc.CardHeader(
                            html.H5('Structure Learning', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            dbc.Row([
                                dbc.Col([
                                    'Subjects:',
                                    dcc.Dropdown(id='bm-subjects', multi=True)
                                ], width=8),
                                dbc.Col([
                                    dbc.Checklist(
                                        options=[{'label': 'Apply to all subjects', 'value': 'apply'}],
                                        style={'marginTop': '10px'},
                                        id='bm-all-subjects',
                                        switch=True
                                    )
                                ], width=4)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    'Variables:',
                                    dcc.Dropdown(id='bm-variables', multi=True)
                                ])
                            ]),
                            html.Br(),
                            dbc.Tabs([
                                dbc.Tab(label='Restricted Boltzmann Machine', id='rbm-tab', children=[
                                    html.Br(),
                                    dbc.Row([
                                        dbc.Col([
                                            'Number of Hidden Nodes:',
                                            dcc.Input(id='bm-hidden', type='number', min=1, style={'width': '100%'})
                                        ]),
                                        dbc.Col([
                                            'Number of Iterations:',
                                            dcc.Input(id='bm-iter', type='number', min=1, value=10, style={'width': '100%'})
                                        ]),
                                        dbc.Col([
                                            'Learning Rate:',
                                            dcc.Input(id='bm-learning-rate', type='number', min=0.0001, max=1, value=0.1, style={'width': '100%'})
                                        ])
                                    ]),
                                    html.Br(),
                                    dbc.Row([
                                        dbc.Col(),
                                        dbc.Col(),
                                        dbc.Col([
                                            dbc.Button('Train BM', id='train-bm', style={'width': '100%'})
                                        ])
                                    ])
                                ]),
                                dbc.Tab(label='Dynamic Boltzmann Machine', id='dybm-tab', disabled=check_pydybm(), children=[
                                    html.Br(),
                                    dbc.Row([
                                        dbc.Col([
                                            'Delay:',
                                            dcc.Input(id='dybm-delay', type='number', min=1, max=100, value=2, style={'width': '100%'})
                                        ]),
                                        dbc.Col([
                                            'Decay:',
                                            dcc.Input(id='dybm-decay', type='number', min=0, max=1, value=0.5, style={'width': '100%'})
                                        ]),
                                        dbc.Col([
                                            'Epochs:',
                                            dcc.Input(id='dybm-epochs', type='number', min=1, style={'width': '100%'})
                                        ])
                                    ]),
                                    html.Br(),
                                    dbc.Row([
                                        dbc.Col(),
                                        dbc.Col(),
                                        dbc.Col([
                                            dbc.Button('Train DyBM', id='train-dybm', style={'width': '100%'})
                                        ])
                                    ]),
                                ]),
                                dbc.Tab(label='Neural Networks', id='nn-tab', children=[
                                    html.Br(),
                                    dbc.Row([
                                        dbc.Col([
                                            'Window size:',
                                            dcc.Input(id='nn-window-size', type='number', min=1, max=20, value=1, style={'width': '100%'})
                                        ]),
                                        dbc.Col([
                                            'Dropout:',
                                            dcc.Input(id='nn-dropout', type='number', min=0, max=1, value=0.5, style={'width': '100%'})
                                        ]),
                                        dbc.Col([
                                            'Epochs:',
                                            dcc.Input(id='nn-epochs', type='number', min=1, style={'width': '100%'})
                                        ])
                                    ]),
                                    html.Br(),
                                    dbc.Row([
                                        dbc.Col(),
                                        dbc.Col(),
                                        dbc.Col([
                                            dbc.Button('Train NN', id='train-nn', style={'width': '100%'})
                                        ])
                                    ])
                                ])
                            ]),
                            dybm_nn_train_modal,
                        ])
                    ])
                ),
                dbc.Col(
                    dbc.Card(children=[
                        dbc.CardHeader(
                            html.H5('Results', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            dbc.Tabs([
                                dbc.Tab(label='RBM', id='rbm-res-tab', children=[dcc.Graph(figure=get_empty_plot('RBM Results will be displayed here'))]),
                                dbc.Tab(label='DyBM (Relations)', id='dybm-res-tab', children=[dcc.Graph(figure=get_empty_plot('DyBM Results will be displayed here'), id='dybm-weights')]),
                                dbc.Tab(label='DyBM (Performances)', id='dybm-perf-tab', children=[dcc.Graph(figure=get_empty_plot('DyBM Performances will be displayed here'))]),
                                dbc.Tab(label='NN', id='nn-res-tab', children=[dcc.Graph(figure=get_empty_plot('NN Results will be displayed here'))]),
                                dbc.Tab(label='NN (Performances)', id='nn-perf-tab', children=[dcc.Graph(figure=get_empty_plot('NN Performances will be displayed here'), id='nn-performances')]),
                            ])
                        ], id='results-body')
                    ], style={'height': '100%'})
                )
            ])
        ], id='main-screen', style={'display': 'none'}),
    ], color='#ACF2D3', style={'border': 'none'}),

    # Hidden div holds the original dataset
    html.Div(id='dataset', style={'display': 'none'}),

    # Hidden div holds the current instance of the dataset
    html.Div(id='current-data', style={'display': 'none'}),

    # Hidden div holds the (possibly imputated) dataset
    html.Div(id='imputed-data', style={'display': 'none'}),

    # Hidden div holds the last selected model between dybm and nn
    html.Div(id='last-model', style={'display': 'none'}, children=['none']),

    # TODO: More imputation options

], style={'backgroundColor': '#ACF2D3', 'min-height': '100vh'})

# Verifies whether or not the upload is a valid csv file and updates Confirm button status, selection text and current dataset
@app.callback(
    dash.dependencies.Output('upload-confirm', 'disabled'),
    dash.dependencies.Output('current-filename', 'children'),
    dash.dependencies.Output('dataset', 'children'),
    [dash.dependencies.Input('upload-ld', 'contents')],
    [dash.dependencies.Input('upload-ld', 'filename')]
)

def update_upload(contents, file_input):
    ctx = dash.callback_context
    if ctx.triggered:
        # Upload is a valid .csv file
        if valid_upload(file_input):
            df = read_upload(contents)
            return False, file_input + ' has been selected', df.to_json()
        # Upload is not a valid .csv file
        else:
            return True, 'Upload is not a valid .csv file', None
    return True, no_update, no_update

# Automatically detect and suggest upload separator
@app.callback(
    dash.dependencies.Output('data-sep', 'value'),
    dash.dependencies.Output('suggested-separator', 'children'),
    [dash.dependencies.Input('upload-ld', 'contents')],
    [dash.dependencies.Input('upload-ld', 'filename')]
)

def check_delimiter_and_columns(contents, file_input):
    ctx = dash.callback_context
    if ctx.triggered:
        # Input must be a csv file
        valid_csv = valid_upload(file_input)
        if valid_csv:
            delimiter = find_delimiter(contents)
            # If the detected delimiter is one of the separator options
            if delimiter in ['\t', ',', ';']:
                badge_content = delimiter * (delimiter != '\t') + '\\t' * (delimiter == '\t')
                return delimiter, badge_content
    return no_update, no_update

# Upload Confirm button closes upload card and displays main page
@app.callback(
    dash.dependencies.Output('upload-screen', 'style'),
    dash.dependencies.Output('main-screen', 'style'),
    [dash.dependencies.Input('upload-confirm', 'n_clicks')]
)

def change_screen(n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        upload_screen_style = {'display': 'none'}
        main_screen_style = {'display': 'inline-block'}
        return upload_screen_style, main_screen_style
    return no_update, no_update

# Upload Confirm button updates raw data table (only runs once during entire workflow)
# Alternatively, new dropdown selections alter the raw dataset table display
@app.callback(
    dash.dependencies.Output('current-columns', 'options'),
    dash.dependencies.Output('current-columns', 'value'),
    dash.dependencies.Output('current-subject', 'options'),
    [dash.dependencies.Input('upload-confirm', 'n_clicks')],
    [dash.dependencies.Input('dataset', 'children')],
)

def update_raw_data(n_clicks, data):
    ctx = dash.callback_context
    if ctx.triggered:
        # Search all triggers for upload confirm
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            # Upload has been accepted
            if current_trigger == 'upload-confirm':
                return setup_dataset(data)

                #return content, current_df.to_json(), col_options, values, ind_options
    return no_update, no_update, no_update

# Uploading the dataset alters the current dataset instance
@app.callback(
    dash.dependencies.Output('current-data', 'children'),
    [dash.dependencies.Input('dataset', 'children')],
    [dash.dependencies.Input('imputed-data', 'children')]
)

def update_current_data(data, imp_data):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'dataset':
            return data
        elif trigger == 'imputed-data':
            return imp_data
    return no_update

# Uploading the dataset or performing imputation methods updates imputed data
# Alternatively, find and replace operations update imputed data
# Alternatively, perform quantile discretization on the given variable with the given number of quantiles

# (Any changes made to imputed data alter current data)
@app.callback(
    dash.dependencies.Output('imputed-data', 'children'),
    [dash.dependencies.Input('dataset', 'children')],
    [dash.dependencies.Input('imputed-data', 'children')],
    [dash.dependencies.Input('ld-imputation-subjects', 'value')],
    [dash.dependencies.Input('ld-imputation-variables', 'value')],
    [dash.dependencies.Input('imputation-method', 'value')],
    [dash.dependencies.Input('missforest-discrete-variables', 'value')],
    [dash.dependencies.Input('all-subjects', 'value')],
    [dash.dependencies.Input('ld-imputation-confirm', 'n_clicks')],
    [dash.dependencies.Input('find-replace-variable', 'value')],
    [dash.dependencies.Input('variable-value-to-replace', 'value')],
    [dash.dependencies.Input('variable-value-replacement', 'value')],
    [dash.dependencies.Input('replace-confirm', 'n_clicks')],
    [dash.dependencies.Input('quantile-discretization-variable', 'value')],
    [dash.dependencies.Input('quantile-discretization-size', 'value')],
    [dash.dependencies.Input('quantile-discretization-encode', 'value')],
    [dash.dependencies.Input('quantile-discretization', 'n_clicks')]
)

def ld_imputation(orig_data, imp_data, subjects, variables, method, d_variables, all_subjects, n_clicks, replace_var, replace_val, replacement, confirm_replacement, qd_var, qd_size, qd_enc, qd_n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            # Initial setup
            if current_trigger == 'dataset':
                return orig_data
            # Impute data
            elif current_trigger == 'ld-imputation-confirm':
                df = pandas.read_json(imp_data).sort_index()
                # all_subjects may be null if never interacted with (should be an empty array)
                if all_subjects is None:
                    all_subjects = []
                apply_all = 'apply' in all_subjects
                new_data = impute_ld_dataset(method, df, subjects, variables, all_subjects=apply_all, discrete=d_variables)
                new_data = new_data.to_json()
                return new_data
            # Find and replace values
            elif current_trigger == 'replace-confirm':
                if not replace_var is None and not replace_val is None and not replacement is None:
                    new_data = replace_in_dataset(imp_data, replace_var, replace_val, replacement)
                    new_data = new_data.to_json()
                    return new_data
            # Perform quantile discretization
            elif current_trigger == 'quantile-discretization':
                df = pandas.read_json(imp_data).sort_index()
                new_data = quantile_discretize_dataset_by_var(df, qd_var, qd_size, qd_enc)
                new_data = new_data.to_json()
                return new_data
    return no_update

# New column and/or subject dropdown selections alter the raw dataset table display
@app.callback(
    dash.dependencies.Output('raw-data', 'children'),
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('current-columns', 'value')],
    [dash.dependencies.Input('current-subject', 'value')],
    [dash.dependencies.Input('data-table-options-close', 'n_clicks')],
)

def update_dataset_table(data, cur_cols, cur_ind, n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        # Does not update if any dropdown option is null or empty
        if cur_cols is None or cur_cols == [] or cur_ind is None:
            return no_update
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            # Only changes if dataset was changed or if options were changed
            if current_trigger == 'data-table-options-close' or current_trigger == 'current-data':
                df = pandas.read_json(data).sort_index()
                # Display a subset of the columns to prevent heavy load
                current_df = df[cur_cols]
                if cur_ind != 'all':
                    current_df = get_subject(current_df, cur_ind, df.columns[0])

                degrees_missingness = column_missingness(current_df)
                tooltip_header = get_column_info(current_df)

                # Create data table with color coded columns and column info
                content = [dash_table.DataTable(
                            columns=[{'name': i, 'id': i} for i in current_df.columns],
                            data=current_df.to_dict('records'),
                            tooltip_header=tooltip_header,
                            tooltip_delay=0,
                            tooltip_duration=None,
                            style_header={
                                'textDecoration': 'underline',
                                'textDecorationStyle': 'dotted'
                            },
                            style_header_conditional=[
                                {'if': {'column_id': c1},
                                'backgroundColor': 'rgb(255, 170, 170)',
                                'color': 'rgb(220, 17, 44)'} for c1 in degrees_missingness['high']
                            ] + [
                                # Columns with missingness above 0% and equal or below 20% missingness
                                {'if': {'column_id': c2},
                                'backgroundColor': 'rgb(255, 255, 200)',
                                'color': 'rgb(170, 115, 8)'} for c2 in degrees_missingness['moderate']
                            ],
                            style_data_conditional=[
                                # Columns with above 20% missingness
                                {'if': {'column_id': c1},
                                'backgroundColor': 'rgb(255, 170, 170)',
                                'color': 'rgb(220, 17, 44)'} for c1 in degrees_missingness['high']
                            ] + [
                                # Columns with missingness above 0% and equal or below 20% missingness
                                {'if': {'column_id': c2},
                                'backgroundColor': 'rgb(255, 255, 200)',
                                'color': 'rgb(170, 115, 8)'} for c2 in degrees_missingness['moderate']
                            ]
                            )]
                return content
    return no_update

# Data Table Options button opens dataframe modal
@app.callback(
    dash.dependencies.Output('data-table-modal', 'is_open'),
    [dash.dependencies.Input('data-table-options', 'n_clicks')],
    [dash.dependencies.Input('data-table-options-close', 'n_clicks')],
    [dash.dependencies.State('data-table-modal', 'is_open')]
)

def toggle_data_table_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# Uploading the dataset updates quantile variable dropdown options
@app.callback(
    dash.dependencies.Output('quantile-discretization-variable', 'options'),
    [dash.dependencies.Input('dataset', 'children')]
)

def update_quantile_discretization_variables(data):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pandas.read_json(data).sort_index()
        options = []
        for variable in df.columns:
            options.append({'label': variable, 'value': variable})
        return options
    return no_update

# Discretize button is enabled once a variable and quantile size have been provided
@app.callback(
    dash.dependencies.Output('quantile-discretization', 'disabled'),
    [dash.dependencies.Input('quantile-discretization-variable', 'value')],
    [dash.dependencies.Input('quantile-discretization-size', 'value')]
)

def update_quantile_discretization_button_status(variable, size):
    ctx = dash.callback_context
    if ctx.triggered:
        # Both variable and quantile size must be provided
        if variable is None or size is None:
            return True
        # There must be at least 2 quantiles
        elif size < 2:
            return True
        return False
    return no_update

# Uploading the dataset populates find and replace variable dropdown options
@app.callback(
    dash.dependencies.Output('find-replace-variable', 'options'),
    [dash.dependencies.Input('dataset', 'children')]
)

def populate_find_replace_variable_options(data):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pandas.read_json(data).sort_index()
        options = []
        for col in df.columns:
            options.append({'label': col, 'value': col})
        return options
    return no_update

# Selecting a find and replace variable changes possible value selections
@app.callback(
    dash.dependencies.Output('variable-value-to-replace', 'options'),
    dash.dependencies.Output('variable-value-to-replace', 'value'),
    dash.dependencies.Output('variable-value-to-replace', 'disabled'),
    [dash.dependencies.Input('dataset', 'children')],
    [dash.dependencies.Input('find-replace-variable', 'value')]
)

def populate_replace_variable_value_options(data, value):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pandas.read_json(data).sort_index()
        if not value is None:
            vals = set(df[value].values)
            vals = list({x for x in vals if x==x})
            options = []
            for val in vals:
                options.append({'label': val, 'value': val})
            return options, vals[0], False
    return no_update, no_update, True

# Having a selected variable, value and replacement enables replace button
@app.callback(
    dash.dependencies.Output('replace-confirm', 'disabled'),
    [dash.dependencies.Input('find-replace-variable', 'value')],
    [dash.dependencies.Input('variable-value-to-replace', 'value')],
    [dash.dependencies.Input('variable-value-replacement', 'value')]
)

def update_replace_button_status(variable, value, replacement):
    ctx = dash.callback_context
    if ctx.triggered:
        if not variable is None and not value is None and not replacement is None:
            return False
    return True

# Imputation button opens imputation modal
@app.callback(
    dash.dependencies.Output('ld-imputation-modal', 'is_open'),
    [dash.dependencies.Input('ld-imputation', 'n_clicks')],
    [dash.dependencies.Input('ld-imputation-cancel', 'n_clicks')],
    [dash.dependencies.Input('ld-imputation-confirm', 'n_clicks')],
    [dash.dependencies.State('ld-imputation-modal', 'is_open')]
)

def toggle_imputation_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    else:
        return is_open

# Populate subjects and variables dropdown imputation options
@app.callback(
    dash.dependencies.Output('ld-imputation-subjects', 'options'),
    dash.dependencies.Output('ld-imputation-variables', 'options'),
    [dash.dependencies.Input('current-data', 'children')]
)

def populate_imputation_options(data):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pandas.read_json(data).sort_index()
        subjects_mv, columns_mv = check_missingness(df)
        subjects_options = []
        for subject in subjects_mv:
            subjects_options.append({'label': subject, 'value': subject})
        column_options = []
        for column in columns_mv:
            column_options.append({'label': column, 'value': column})
        return subjects_options, column_options
    return no_update, no_update

# Apply to all subjects option disables Subjects dropout
@app.callback(
    dash.dependencies.Output('ld-imputation-subjects', 'disabled'),
    [dash.dependencies.Input('all-subjects', 'value')]
)

def toggle_subjects_dropdown(values):
    ctx = dash.callback_context
    if ctx.triggered:
        if 'apply' in values:
            return True
        else:
            return False
    return no_update

# Selecting "MissForest" imputation method alters div with discrete variables options display
@app.callback(
    dash.dependencies.Output('discrete-variables-div', 'style'),
    [dash.dependencies.Input('imputation-method', 'value')]
)

def toggle_discrete_variables_div(method):
    ctx = dash.callback_context
    if ctx.triggered:
        style = {}
        if method == 'missforest':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style
    return no_update

# Selecting new imputation variables changes discrete variables options

@app.callback(
    dash.dependencies.Output('missforest-discrete-variables', 'options'),
    [dash.dependencies.Input('ld-imputation-variables', 'value')]
)

def change_discrete_variables_options(variables):
    ctx = dash.callback_context
    if ctx.triggered:
        options = []
        for variable in variables:
            options.append({'label': variable, 'value': variable})
        return options
    return no_update

@app.callback(
    dash.dependencies.Output('bm-subjects', 'options'),
    dash.dependencies.Output('bm-variables', 'options'),
    [dash.dependencies.Input('dataset', 'children')]
)

def populate_learning_dataset_parameters(data):
    ctx = dash.callback_context
    if ctx.triggered:
        variable_options, _, subject_options = setup_dataset(data, add_all=False)
        return subject_options, variable_options
    return no_update, no_update

# Train BM button is enabled once all related inputs are valid
@app.callback(
    dash.dependencies.Output('train-bm', 'disabled'),
    [dash.dependencies.Input('bm-subjects', 'value')],
    [dash.dependencies.Input('bm-all-subjects', 'value')],
    [dash.dependencies.Input('bm-variables', 'value')],
    [dash.dependencies.Input('bm-hidden', 'value')],
    [dash.dependencies.Input('bm-iter', 'value')],
    [dash.dependencies.Input('bm-learning-rate', 'value')]
)

def check_train_bm_status(subjects, use_all, variables, n_hidden, iter, l_r):
    # Either individual subjects must be specified or all subjects flag must be enabled
    if (subjects is None or subjects == []) and (use_all is None or use_all == []):
        return True
    # At least two variables must be specified
    if variables is None or len(variables) < 2:
        return True
    # Model must have at least one hidden variable
    if n_hidden is None or n_hidden < 1:
        return True
    # Model must have at least 1 iteration
    if iter is None or iter < 1:
        return True
    # Learning rate must be a value between 0 and 1
    if l_r is None or l_r <= 0 or l_r > 1:
        return True
    return False

# Train BM button trains a Boltzmann Machine with the given input parameters
@app.callback(
    dash.dependencies.Output('rbm-res-tab', 'children'),
    [dash.dependencies.Input('train-bm', 'n_clicks')],
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('bm-subjects', 'value')],
    [dash.dependencies.Input('bm-all-subjects', 'value')],
    [dash.dependencies.Input('bm-variables', 'value')],
    [dash.dependencies.Input('bm-hidden', 'value')],
    [dash.dependencies.Input('bm-iter', 'value')],
    [dash.dependencies.Input('bm-learning-rate', 'value')]
)

def train_bm(n_clicks, data, chosen_subjects, use_all, variables, n_hidden, iter, l_r):
    ctx = dash.callback_context
    if ctx.triggered:
        # Parse all triggers
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            if current_trigger == 'train-bm':
                model, input_size = bm_workflow(data, chosen_subjects, use_all, variables, n_hidden, iter, l_r)
                fig = px.imshow(model.components_, color_continuous_scale='RdBu_r')
                # Add borders around each column (separates encoded inputs)
                border_cursor = 0
                for el in input_size[:-1]:
                    border_cursor += el
                    fig = fig.add_vline(x=border_cursor - 0.5, line_width=3, line_color='black')
                return [dcc.Graph(figure=fig)]
    return no_update

# Train DyBM or Neural Network opens Train modal
@app.callback(
    dash.dependencies.Output('dybm-nn-train-modal', 'is_open'),
    dash.dependencies.Output('last-model', 'children'),
    [dash.dependencies.Input('train-dybm', 'n_clicks')],
    [dash.dependencies.Input('train-nn', 'n_clicks')],
    [dash.dependencies.Input('dybm-nn-train-confirm', 'n_clicks')],
    [dash.dependencies.State('dybm-nn-train-modal', 'is_open')]
)

def toggle_dybm_nn_train_modal(n1, n2, n3, is_open):
    # DyBM
    if n1:
        return not is_open, 'dybm'
    # NN
    elif n2:
        return not is_open, 'nn'
    # Model is trained
    elif n3:
        return not is_open, 'none'
    else:
        return is_open, no_update

@app.callback(
    dash.dependencies.Output('dybm-nn-output-variables', 'options'),
    dash.dependencies.Output('dybm-nn-test-set', 'options'),
    [dash.dependencies.Input('current-data', 'children')]
)

def populate_train_opts(data):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pandas.read_json(data).sort_index()

        output_options = []
        for col in df.columns:
            if df[col].isna().sum() == 0:
                output_options.append({'label': col, 'value': col})

        test_options = []
        for s in subjects(df, df.columns[0]):
            test_options.append({'label': s, 'value': s})

        return output_options, test_options

    return no_update, no_update

# Train DyBM button is enabled once all related inputs are valid
@app.callback(
    dash.dependencies.Output('train-dybm', 'disabled'),
    [dash.dependencies.Input('bm-subjects', 'value')],
    [dash.dependencies.Input('bm-all-subjects', 'value')],
    [dash.dependencies.Input('bm-variables', 'value')],
    [dash.dependencies.Input('dybm-delay', 'value')],
    [dash.dependencies.Input('dybm-decay', 'value')],
    [dash.dependencies.Input('dybm-epochs', 'value')]
)

def check_train_dybm_status(subjects, use_all, variables, delay, decay, epochs):
    # Either individual subjects must be specified or all subjects flag must be enabled
    if (subjects is None or subjects == []) and (use_all is None or use_all == []):
        return True
    # At least two variables must be specified
    if variables is None or len(variables) < 2:
        return True
    # Delay must be equal or above 2
    if delay is None or delay < 2:
        return True
    # Decay must be between 0 and 1
    if decay is None or decay < 0 or decay > 1:
        return True
    if epochs is None or epochs < 1:
        return True
    return False

# Confirm Train button is enabled once output variables are provided
@app.callback(
    dash.dependencies.Output('dybm-nn-train-confirm', 'disabled'),
    [dash.dependencies.Input('dybm-nn-output-variables', 'value')],
    [dash.dependencies.Input('last-model', 'children')]
)

def check_train_dybm_nn_status(outputs, last_model):
    if outputs is None or outputs == []:
        return True
    # For Neural Networks, only one output variable must be specified
    if last_model == 'nn' and len(outputs) > 1:
        return True
    return False

# Confirm Train button trains a DyBM if last_model equals dybm
@app.callback(
    dash.dependencies.Output('dybm-weights', 'figure'),
    dash.dependencies.Output('dybm-perf-tab', 'children'),
    [dash.dependencies.Input('dybm-nn-train-confirm', 'n_clicks')],
    [dash.dependencies.Input('last-model', 'children')],
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('bm-subjects', 'value')],
    [dash.dependencies.Input('bm-all-subjects', 'value')],
    [dash.dependencies.Input('bm-variables', 'value')],
    [dash.dependencies.Input('dybm-delay', 'value')],
    [dash.dependencies.Input('dybm-decay', 'value')],
    [dash.dependencies.Input('dybm-epochs', 'value')],
    [dash.dependencies.Input('dybm-nn-output-variables', 'value')],
    [dash.dependencies.Input('dybm-nn-test-set', 'value')]
)

def train_dybm_dash(n_clicks, l_model, data, subjects, use_all, variables, delay, decay, epochs, out_vars, test_set):
    ctx = dash.callback_context
    if ctx.triggered and l_model == 'dybm':
        # Parse all triggers
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            if current_trigger == 'dybm-nn-train-confirm':
                model, new_vars, new_out, recalls = dybm_workflow(data, subjects, use_all, variables, delay, decay, 5, out_vars, epochs, test_set=test_set)
                weights = model.layers[0].layers[0].variables['W']
                final_weights = np.zeros((weights.shape[2], weights.shape[1]))

                for i in range(weights.shape[2]):
                    final_weights[i, :] = weights[:, :, i].mean(axis=0)

                # Weights heatmap
                weight_plot = px.imshow(final_weights, color_continuous_scale='RdBu_r', x=new_vars, y=new_out)

                # Performances plot
                perf_plot = px.line(recalls, template='ggplot2')
                perf_plot.update_layout(margin=fig_margin_layout, legend=fig_legend_layout,
                    xaxis=dict(
                        title='Epochs'
                    ),
                    yaxis=dict(
                        title='Recall'
                    ),
                )

                return weight_plot, [dcc.Graph(figure=perf_plot)]

    return no_update

# Train Neural Network option is enabled once all related inputs are valid
@app.callback(
    dash.dependencies.Output('train-nn', 'disabled'),
    [dash.dependencies.Input('bm-subjects', 'value')],
    [dash.dependencies.Input('bm-all-subjects', 'value')],
    [dash.dependencies.Input('bm-variables', 'value')],
    [dash.dependencies.Input('nn-window-size', 'value')],
    [dash.dependencies.Input('nn-dropout', 'value')],
    [dash.dependencies.Input('nn-epochs', 'value')]
)

def check_train_nn_status(subjects, use_all, variables, ws, dropout, epochs):
    # Either individual subjects must be specified or all subjects flag must be enabled
    if (subjects is None or subjects == []) and (use_all is None or use_all == []):
        return True
    # At least two variables must be specified
    if variables is None or len(variables) < 2:
        return True
    if ws is None or ws < 1:
        return True
    if dropout is None or dropout < 0 or dropout > 1:
        return True
    if epochs is None or epochs < 1:
        return True
    return False

# Confirm Train button trains a NN if last_model equals nn
@app.callback(
    dash.dependencies.Output('nn-res-tab', 'children'),
    dash.dependencies.Output('nn-perf-tab', 'children'),
    [dash.dependencies.Input('dybm-nn-train-confirm', 'n_clicks')],
    [dash.dependencies.Input('last-model', 'children')],
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('bm-subjects', 'value')],
    [dash.dependencies.Input('bm-all-subjects', 'value')],
    [dash.dependencies.Input('bm-variables', 'value')],
    [dash.dependencies.Input('nn-window-size', 'value')],
    [dash.dependencies.Input('nn-dropout', 'value')],
    [dash.dependencies.Input('nn-epochs', 'value')],
    [dash.dependencies.Input('dybm-nn-output-variables', 'value')],
    [dash.dependencies.Input('dybm-nn-test-set', 'value')]
)

def train_nn_dash(n_clicks, l_model, data, subjects, use_all, variables, ws, dropout, epochs, out_vars, test_set):
    ctx = dash.callback_context
    if ctx.triggered and l_model == 'nn':
        # Parse all triggers
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            if current_trigger == 'dybm-nn-train-confirm':
                perf, grads = nn_workflow(data, subjects, use_all, variables, out_vars, ws, dropout, epochs, test_set)

                # Gradient plot
                grad_plot = px.bar(x=variables, y=grads, template='ggplot2')
                grad_plot.update_layout(margin=fig_margin_layout, legend=fig_legend_layout,
                xaxis=dict(
                    title=''
                ),
                yaxis=dict(
                    title='Gradient'
                ))

                # Performances plot
                perf_plot = px.line(perf, template='ggplot2')
                perf_plot.update_layout(margin=fig_margin_layout, legend=fig_legend_layout,
                    xaxis=dict(
                        title='Epochs'
                    ),
                    yaxis=dict(
                        title='Recall'
                    ),
                )

                return [dcc.Graph(figure=grad_plot)], [dcc.Graph(figure=perf_plot)]

    return no_update, no_update


if __name__ == '__main__':
    app.run_server(debug=True)
