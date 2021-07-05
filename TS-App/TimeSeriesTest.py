import dash
from dash.dash import no_update
from Structure.helperfunctions import *
from Structure.modals import *

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

neural_network_options = [{'label': 'Bi-LSTM', 'value': 'Bi-LSTM'}, {'label': 'CNN', 'value': 'CNN'}]

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
                                id='upload-ts',
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
                            html.Div('No dataset has been selected', id='current-filename-ts'),
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
                            ], justify='between')
                        ]),
                        dbc.CardFooter(dbc.Button('Confirm', id='upload-ts-confirm', disabled=True,
                        style={'backgroundColor': '#58B088', 'border': 'none'}))
                    ], style={'borderColor': '#58B088'})
                ),
                dbc.Col(),
            ])
        ], id='upload-screen-ts', style={'height': '100vh'}),
        # Dashboard main page (after dataset has been imported)
        dbc.CardBody([
            # First "row" of panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Time Series', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            dcc.Graph(
                                id='ts',
                                style={'display': 'inline-block', 'width': '100%'}
                            ),
                            # Variable selection dropdown
                            dcc.Dropdown(
                                id='variable-selection',
                                multi=True,
                            )
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Options', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            dcc.Dropdown(id='chosen-variable', clearable=False, style={'marginBottom': '10px'}),
                            html.Div(id='differences-text', children='Dataset has not been differenced'),
                            html.Div(id='adf-test', children='ADF p-value'),
                            html.Div(id='missing-indices', children='Variable has no missing indices'),
                            html.Br(),
                            dbc.Button('First Difference', id='first-difference', n_clicks=0, style={'width': '100%'}),
                            html.Br(),
                            html.Br(),
                            autocorrelation_modal,
                            html.Br(),
                            seasonal_decomposition_modal,
                            html.Br(),
                            imputation_modal,
                            html.Br(),
                            confirm_revert_modal,
                            html.Br(),
                            dbc.Button('Export Dataset', id='export-ts-dataset', outline=True, color='dark', style={'width': '100%'}),
                            dcc.Download(id='download-ts-dataset'),
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Causality Network', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            cyto.Cytoscape(
                                id='causality-network',
                                layout={'name': 'circle'},
                                style={'width': '100%', 'height':'440px'},
                                stylesheet=[
                                    {'selector': 'node',
                                    'style': {
                                        'label': 'data(id)',
                                        'backgroundColor': 'black'
                                        }
                                    },
                                    {'selector': 'edge',
                                    'style': {
                                        'curve-style': 'bezier',
                                        'target-arrow-shape': 'vee',
                                    }
                                }]
                            ),
                            html.Br(),
                            causality_modal
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=5)
            ]),
            html.Br(),

            # Second "row" of panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Model Training', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            # Forecasting tabs
                            dbc.Tabs(
                                id='forecasting-tabs',
                                active_tab='linear-univariate',
                                children=[
                                    # SARIMAX tab
                                    dbc.Tab(
                                        label='Linear Univariate (SARIMAX)',
                                        tab_id='linear-univariate',
                                        children=[
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        'Autoregressive (p):',
                                                        dcc.Input(id='univariate-linear-autoregressive-parameter', type='number', min=0, placeholder='Autoregressive (p)', style={'width': '100%'}),
                                                    ])
                                                ]),
                                                dbc.Col([
                                                    html.Div([
                                                        'Integration (d):',
                                                        dcc.Input(id='integration-parameter', type='number', min=0, placeholder='Integration (d)', style={'width': '100%'}),
                                                    ])
                                                ])
                                            ]),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        'Moving Average (q):',
                                                        dcc.Input(id='moving-average-parameter', type='number', min=0, placeholder='Moving Average (q)', style={'width': '100%'}),
                                                    ])
                                                ]),
                                                dbc.Col([
                                                    html.Div([
                                                        'Seasonal (s):',
                                                        dcc.Input(id='seasonal-parameter', type='number', min=0, placeholder='Seasonal (s)', style={'width': '100%'}),
                                                    ])
                                                ])
                                            ]),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    uni_param_search_modal,
                                                ]),
                                                dbc.Col([
                                                    dbc.Button('Run SARIMAX', id='run-sarimax', n_clicks=0, style={'width': '100%'})
                                                ])
                                            ])
                                        ]
                                    ),
                                    # VAR tab
                                    dbc.Tab(
                                        label='Linear Multivariate (VAR)',
                                        tab_id='linear-multivariate',
                                        children=[
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        'Autoregressive (p):',
                                                        dcc.Input(id='multivariate-linear-autoregressive-parameter', type='number', min=0, placeholder='Autoregressive (p)', style={'width': '100%'}),
                                                    ])
                                                ]),
                                                # "Empty" Column to restrict space
                                                dbc.Col([
                                                    html.Div([])
                                                ])
                                            ]),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    multi_param_search_modal,
                                                ]),
                                                dbc.Col([
                                                    dbc.Button('Run VAR', id='run-var', n_clicks=0, style={'width': '100%'})
                                                ])
                                            ])
                                        ]
                                    ),
                                    # Neural Networks tab
                                    dbc.Tab(
                                        label='Neural Networks',
                                        tab_id='neural-networks',
                                        children=[
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        'Neural Network Model:',
                                                        dcc.Dropdown(id='neural-network-model', options=neural_network_options, value=neural_network_options[0]['value'], clearable=False),
                                                    ])
                                                ]),
                                                dbc.Col([
                                                    html.Div([
                                                        'Sliding Window Size:',
                                                        dcc.Input(id='order', type='number', min=0, placeholder='Sliding Window Size', style={'width': '100%'}),
                                                    ])
                                                ])
                                            ]),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        'Output Size:',
                                                        dcc.Input(id='output-size', type='number', min=1, placeholder='Output Size', style={'width': '100%'}),
                                                    ])
                                                ]),
                                                dbc.Col([
                                                    html.Div([
                                                        'Seed:',
                                                        dcc.Input(id='seed', type='number', min=0, max=(2 ** 16), placeholder='Random Seed', disabled=True, style={'width': '100%'}),
                                                    ])
                                                ])
                                            ]),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    nn_preferences_modal
                                                ]),
                                                dbc.Col([
                                                    dbc.Button('Run Neural Network', id='run-nn', n_clicks=0, style={'width': '100%'})
                                                ]),
                                                dbc.Col([
                                                    nn_results_modal
                                                ])
                                            ])
                                        ]
                                    )
                                ]
                            ),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        'Forecast Variables:',
                                        dcc.Dropdown(id='forecasting-variables', multi=True, style={'width': '100%'}),
                                    ])
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        'Train-Test Split:',
                                        dcc.Input(id='split', type='number', placeholder='Train-Test Split', style={'width': '100%'}),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Forecasting Window:',
                                        dcc.Input(id='forecast', type='number', placeholder='Forecasting Window', style={'width': '100%'}),
                                    ])
                                ])
                            ]),
                            html.Br(),
                            dbc.Checklist(
                                options=[
                                    {'label': 'Export model', 'value': 'export'}
                                ],
                                value=[],
                                id='export-model',
                                switch=True
                            ),
                            # Download SARIMAX results as txt
                            dcc.Download(id='download-sarimax-model'),
                            # Download VAR results as txt
                            dcc.Download(id='download-var-model'),
                            # Download Neural Network architecture as JSON
                            dcc.Download(id='download-nn-architecture')
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Training Results', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            dcc.Graph(
                                id='forecast-result',
                                figure=get_empty_plot('Please run a model first'),
                                style={'display': 'inline-block', 'width': '100%'}
                            ),
                            dcc.Dropdown(id='chosen-variable-forecast', clearable=False, disabled=True)
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=7)
            ]),
            html.Br(),

            # Third "row" of panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Residual Analysis', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            dcc.Graph(id='residual-autocorrelation', figure=get_empty_plot('Please run a model first'), style={'width': '100%'}),
                            dcc.RadioItems(
                                options=[
                                    {'label': 'Autocorrelation Function', 'value': 'acf'},
                                    {'label': 'Histogram', 'value': 'histogram'},
                                ],
                                value='acf',
                                id='res-method',
                                inputStyle={'margin-right': '5px'},
                                labelStyle={'margin-right': '20px'}
                            )
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5('Model Performances', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody([
                            dcc.Graph(id='model-status', figure=get_empty_plot('Please run a model first'), style={'width': '100%', 'height': '100%'})
                        ])
                    ], style={'height': '100%', 'borderColor': '#58B088'})
                ], width=6)
            ]),
        ], id='main-screen-ts', style={'display': 'none'})
    ], color='#ACF2D3', style={'border': 'none'}),

    # Hidden div holds the original dataset
    html.Div(id='dataset', style={'display': 'none'}),

    # Hidden div holds the current dataset
    html.Div(id='current-data', style={'display': 'none'}),

    # Hidden div holds the current number of differences
    html.Div(id='current-differences', children=0, style={'display': 'none'}),

    # Hidden div holds the data related to Neural Network hyperparameters
    html.Div(id='nn-hyperparameters', children=json.dumps({}), style={'display': 'none'}),

    # Hidden div holds the name of the last trained model
    html.Div(id='last-trained-model', children='', style={'display': 'none'}),

    # Hidden div holds the data, predictions, forecast and confidence intervals of last trained SARIMAX model
    html.Div(id='last-sarimax-training', children=json.dumps({}), style={'display': 'none'}),

    # Hidden div holds the data, predictions, forecast and confidence intervals of last trained VAR model
    html.Div(id='last-var-training', children=json.dumps({}), style={'display': 'none'}),

    # Hidden div holds the data, predictions, forecast and confidence intervals of last trained Neural Network model
    html.Div(id='last-nn-training', children=json.dumps({}), style={'display': 'none'}),

    # Hidden div holds the results of the last parameter search
    html.Div(id='last-param-search', style={'display': 'none'}),

    # Hidden div holds the results of the last imputation
    html.Div(id='last-imputation', style={'display': 'none'}),

    # Hidden div holds the results of each model
    html.Div(id='model-performances', children=json.dumps({}), style={'display': 'none'}),

    # Hidden div holds the last test size (used for resetting results)
    html.Div(id='last-test-size', children=0, style={'display': 'none'}),

    # Hidden div holds the training set and their respective predictions
    html.Div(id='training-set-res', children=pd.DataFrame().to_json(), style={'display': 'none'})

    # TODO: Hidden div holds the values removed after each difference
    # TODO: Causality networks:
    #       - Focus on a single variable (PCMCI)
    # TODO: Loading status for training/forecasting models

], style={'backgroundColor': '#ACF2D3', 'min-height': '100vh'})

# Verifies whether or not the upload is a valid csv file and updates Confirm button status and selection test
@app.callback(
    dash.dependencies.Output('upload-ts-confirm', 'disabled'),
    dash.dependencies.Output('current-filename-ts', 'children'),
    [dash.dependencies.Input('upload-ts', 'contents')],
    [dash.dependencies.Input('upload-ts', 'filename')]
)

def update_upload_ts_status(contents, file_input):
    ctx = dash.callback_context
    if ctx.triggered:
        valid_csv = check_file(file_input)
        # Upload is a valid .csv file
        if valid_csv:

            return False, file_input + ' has been selected'
        # Upload is not a valid .csv file
        else:
            print('Please upload a valid .csv file')
            return True, 'Upload is not a valid .csv file'
    return True, no_update

# Upload Confirm Button updates stored dataset
@app.callback(
    dash.dependencies.Output('dataset', 'children'),
    [dash.dependencies.Input('upload-ts', 'contents')],
    [dash.dependencies.Input('data-sep', 'value')],
    [dash.dependencies.Input('upload-ts-confirm', 'n_clicks')]
)

def update_dataset(contents, sep, n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            # If Upload Confirm Button was pressed
            if current_trigger == 'upload-ts-confirm':
                df = read_upload(contents, sep)
                return df.to_json()
    return no_update

# Upload Confirm button closes upload card and displays main page
@app.callback(
    dash.dependencies.Output('upload-screen-ts', 'style'),
    dash.dependencies.Output('main-screen-ts', 'style'),
    [dash.dependencies.Input('upload-ts-confirm', 'n_clicks')]
)

def change_screen_ts(n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        upload_screen_style = {'display': 'none'}
        main_screen_style = {'display': 'inline-block'}
        return upload_screen_style, main_screen_style
    return no_update, no_update

# Populate variable dropdowns with options and a default value once dataset has been uploaded
@app.callback(
    dash.dependencies.Output('variable-selection', 'options'),
    dash.dependencies.Output('variable-selection', 'value'),
    dash.dependencies.Output('chosen-variable', 'options'),
    dash.dependencies.Output('chosen-variable', 'value'),
    dash.dependencies.Output('forecasting-variables', 'options'),
    dash.dependencies.Output('forecasting-variables', 'value'),
    dash.dependencies.Output('causality-variable', 'options'),
    dash.dependencies.Output('causality-variable', 'value'),
    [dash.dependencies.Input('dataset', 'children')]
)

def populate_variable_select(data):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pd.read_json(data).sort_index()
        options = []
        for col in df.columns:
            options.append({'label': col, 'value': col})
        values = df.columns[:5]

        chosen_variable_options = [{'label': 'Dataset', 'value': 'data_all'}] + options
        return options, values, chosen_variable_options, 'data_all', options, values, chosen_variable_options, 'data_all'

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

# Button press updates text and plot
# Alternatively, variable selection updates plot
# Alternatively, Revert Changes button undoes every change made to the dataset
@app.callback(
    # Updates the "Number of differences" text
    dash.dependencies.Output('differences-text', 'children'),
    # Updates the time series plot
    dash.dependencies.Output('ts', 'figure'),
    # Updates the saved current data
    dash.dependencies.Output('current-data', 'children'),
    # Updates the number of differences
    dash.dependencies.Output('current-differences', 'children'),
    dash.dependencies.Output('imputation-variables', 'options'),
    [dash.dependencies.Input('first-difference', 'n_clicks')],
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('variable-selection', 'value')],
    [dash.dependencies.Input('current-differences', 'children')],
    [dash.dependencies.Input('dataset', 'children')],
    [dash.dependencies.Input('revert-confirm', 'n_clicks')],
    [dash.dependencies.Input('imputation-confirm', 'n_clicks')],
    [dash.dependencies.Input('last-imputation', 'children')]
)

def update_ts(n_clicks, children, value, num_dif, orig_data, undo, imp_n, last_imp):
    imp_vars = []
    if not children is None:
        imp_vars = get_missing_vars(children)
        df = pd.read_json(children).sort_index()
    elif not orig_data is None:
        imp_vars = get_missing_vars(orig_data)
        df = pd.read_json(orig_data).sort_index()

    if imp_vars == []:
        imp_vars = no_update

    ctx = dash.callback_context
    # Only with triggers
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]

            # If Dataset was uploaded or Revert Changes button was pressed
            if current_trigger == 'dataset' or current_trigger == 'revert-confirm':
                num_dif = 0
                text_dif = dbc.Row([dbc.Col('Nº of differences:', width=8), dbc.Col(dbc.Badge(num_dif, color='light', className='mr-1'), width=3)], justify='between')
                orig = pd.read_json(orig_data).sort_index()
                if len(value) < 1:
                    fig = get_empty_plot('Please select a variable')
                else:
                    fig = px.line(orig, y=value, template='ggplot2')
                    fig.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)
                imp_vars = get_missing_vars(orig_data)
                return text_dif, fig, orig_data, num_dif, imp_vars

            # If First Difference button was pressed
            elif current_trigger == 'first-difference':
                df = difference(df)
                text_dif = dbc.Row([dbc.Col('Nº of differences:', width=8), dbc.Col(dbc.Badge(num_dif + 1, color='light', className='mr-1'), width=3)], justify='between')
                if len(value) < 1:
                    fig = get_empty_plot('Please select a variable')
                else:
                    fig = px.line(df, y=value, template='ggplot2')
                    fig.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)
                return text_dif, fig, df.to_json(), num_dif + 1, imp_vars

            # User confirmed imputation results, dataset is updated
            elif current_trigger == 'imputation-confirm':
                if last_imp is None:
                    raise dash.exceptions.PreventUpdate()
                last_imputation = pd.read_json(last_imp).sort_index()
                # Override imputed columns
                for col in last_imputation.columns:
                    df[col] = last_imputation[col]
                # Number of differences text is not changed
                text_dif = no_update
                num_dif = no_update
                # Update imputation variable options
                imp_vars = []
                cols = get_missing_columns(df)
                for col in cols:
                    imp_vars.append({'label': col, 'value': col})
                if len(value) < 1:
                    fig = get_empty_plot('Please select a variable')
                else:
                    fig = px.line(df, y=value, template='ggplot2')
                    fig.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)
                return text_dif, fig, df.to_json(), num_dif, imp_vars

            # If variable selection was changed
            elif current_trigger == 'variable-selection':
                # Cannot plot empty dataset!
                if len(value) < 1:
                    fig = get_empty_plot('Please select a variable')
                else:
                    fig = px.line(df, y=value, template='ggplot2')
                    fig.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)
                    # Only alter plot, number of differences and current data remain unaltered
                return no_update, fig, no_update, no_update, imp_vars

            # By default, update nothing
            else:
                raise dash.exceptions.PreventUpdate()

    # Initial startup
    else:
        return no_update, no_update, no_update, no_update, no_update

# Export Dataset button triggers download of the current dataset in csv format
@app.callback(
    dash.dependencies.Output('download-ts-dataset', 'data'),
    dash.dependencies.Input('export-ts-dataset', 'n_clicks'),
    dash.dependencies.Input('current-data', 'children')
)

def export_current_dataset(n_clicks, data):
    ctx = dash.callback_context
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]
            if current_trigger == 'export-ts-dataset':
                df = pd.read_json(data).sort_index()
                return dcc.send_data_frame(df.to_csv, 'dataset_output.csv')

    raise dash.exceptions.PreventUpdate()

# Perform ADF test after each update on the dataset
# Alternatively, change variable statistics
# Do not display statistics until data has no missing values
@app.callback(
    # Updates the ADF test p-value
    dash.dependencies.Output('adf-test', 'children'),
    # Updates text with number of missing indices
    dash.dependencies.Output('missing-indices', 'children'),
    dash.dependencies.Output('acf', 'figure'),
    dash.dependencies.Output('pacf', 'figure'),
    dash.dependencies.Output('trend', 'figure'),
    dash.dependencies.Output('seasonal', 'figure'),
    dash.dependencies.Output('residual', 'figure'),
    dash.dependencies.Output('autocorrelation', 'disabled'),
    dash.dependencies.Output('seasonal-decomposition', 'disabled'),
    [dash.dependencies.Input('first-difference', 'n_clicks')],
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('chosen-variable', 'value')]
)

def update_adf(n_clicks, children, value):
    ctx = dash.callback_context
    if ctx.triggered:
        df = pd.read_json(children).sort_index()
        mi = 0
        disabled = False

        # If entire dataset is selected in dropdown
        if value == 'data_all':
            adf_msg = dbc.Row([dbc.Col('Stationary:', width=8), dbc.Col(dbc.Badge('Yes', color='success', className='mr-1'), width=3)], justify='between')
            var = df.columns[0]
            disabled = True

            # If there are no missing values
            if len(get_missing_columns(df)) == 0:
                for col in df.columns:
                    adf = adf_pvalue(df[col])
                    # If any p-value is above 0.05, adf message is changed
                    if adf > 0.05:
                        adf_msg = dbc.Row([dbc.Col('Stationary:', width=8), dbc.Col(dbc.Badge('No', color='danger', className='mr-1'), width=3)], justify='between')

            # If there are missing values, ADF test cannot be performed
            else:
                adf_msg = dbc.Row([dbc.Col('Stationary:', width=8), dbc.Col(dbc.Badge('Unk', color='warning', className='mr-1'), width=3)], justify='between')
                disabled = True

            # Display number of missing indices either way
            for col in df.columns:
                mi += len(get_missing_indices(df[col]))
            mi_msg = dbc.Row([dbc.Col('Missing values:', width=8), dbc.Col(get_mi_badge(mi), width=3)], justify='between')

        # If specific variable is selected
        else:
            var = value
            mi = len(get_missing_indices(df[var]))
            adf_msg = dbc.Row([dbc.Col('ADF p-value:', width=8), dbc.Col(get_adf_badge(df[var], mi), width=3)], justify='between')
            if mi != 0:
                disabled = True
            mi_msg = dbc.Row([dbc.Col('Missing values:', width=8), dbc.Col(get_mi_badge(mi), width=3)], justify='between')
        # If there are no missing values, get variable plots
        if len(get_missing_columns(df)) == 0:
            acf_data, pacf_data = get_acf_and_pacf(df[var], alpha=0.05)
            acf_points, acf_conf = acf_data
            fig_acf = create_acf_plot(acf_points, 'acf', conf_int=acf_conf)
            fig_pacf = create_acf_plot(pacf_data, 'pacf')
            fig_trend, fig_seasonal, fig_residual = setup_seasonal_decomposition(df, var)
        # If there are missing values, variable plots will be empty
        else:
            fig_acf = no_update
            fig_pacf = no_update
            fig_trend, fig_seasonal, fig_residual = no_update, no_update, no_update
        return adf_msg, mi_msg, fig_acf, fig_pacf, fig_trend, fig_seasonal, fig_residual, disabled, disabled
    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

# Changes to data or missing indices text may change
# Imputation, First Difference and Estimate Causality button status
# Imputation button is always opposite to the remaining two
@app.callback(
    dash.dependencies.Output('imputation', 'disabled'),
    dash.dependencies.Output('first-difference', 'disabled'),
    dash.dependencies.Output('causality', 'disabled'),
    [dash.dependencies.Input('missing-indices', 'children')],
    [dash.dependencies.Input('current-data', 'children')],
)
def imputation_status(msg, children):
    df = pd.read_json(children).sort_index()
    mi = get_missing_indices_multiple_cols(df)
    if len(mi) != 0:
        return False, True, True
    return True, False, False

# Autocorrelation Info button opens modal with
# autocorrelation and partial autocorrelation functions
@app.callback(
    dash.dependencies.Output('autocorrelation-modal', 'is_open'),
    [dash.dependencies.Input('autocorrelation', 'n_clicks')],
    [dash.dependencies.Input('autocorrelation-close', 'n_clicks')],
    [dash.dependencies.State('autocorrelation-modal', 'is_open')]
)

def toggle_autocorrelation_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# Seasonal Decomposition button opens modal with
# trend, residual and seasonal components
@app.callback(
    dash.dependencies.Output('seasonal-decomposition-modal', 'is_open'),
    [dash.dependencies.Input('seasonal-decomposition', 'n_clicks')],
    [dash.dependencies.Input('seasonal-decomposition-close', 'n_clicks')],
    [dash.dependencies.State('seasonal-decomposition-modal', 'is_open')]
)

def toggle_seasonal_decomposition_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# Imputation button opens model with imputation options
@app.callback(
    dash.dependencies.Output('imputation-modal', 'is_open'),
    [dash.dependencies.Input('imputation', 'n_clicks')],
    [dash.dependencies.Input('imputation-cancel', 'n_clicks')],
    [dash.dependencies.Input('imputation-confirm', 'n_clicks')],
    [dash.dependencies.State('imputation-modal', 'is_open')]
)

def toggle_imputation_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    else:
        return is_open

# Changing imputation method changes parameter label and placeholder
@app.callback(
    dash.dependencies.Output('imputation-parameter-name', 'children'),
    dash.dependencies.Output('imputation-parameter', 'placeholder'),
    [dash.dependencies.Input('imputation-method', 'value')]
)

def change_imputation_parameter_info(method):
    if method == 'spline':
        placeholder = 'Spline order'
    else:
        placeholder = 'Window size'
    return [placeholder + ':'], placeholder

# General callback for all imputation options
# Results are stored in a hidden div, the user must confirm in order for the changes
# to take place (separate callback)
# Note: uses original data as input in order to prevent circular dependencies
# As such, imputed variables cannot be used for further imputation
@app.callback(
    dash.dependencies.Output('imputation-results', 'figure'),
    dash.dependencies.Output('last-imputation', 'children'),
    # General inputs
    [dash.dependencies.Input('dataset', 'children')],
    [dash.dependencies.Input('imputation-variables', 'value')],
    # Imputation method imputs
    [dash.dependencies.Input('imputation-method', 'value')],
    [dash.dependencies.Input('imputation-parameter', 'value')],
    [dash.dependencies.Input('perform-imputation', 'n_clicks')]
)

def impute(children, imp_var, method, param, n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]

            if current_trigger == 'perform-imputation':
                df = pd.read_json(children).sort_index()

                # Variable must be selected
                if imp_var is None:
                    raise dash.exceptions.PreventUpdate()

                # Spline interpolation
                if method == 'spline':
                    # Spline order must be a number between 1 and 5
                    if param is None or param < 1 or param > 5:
                        raise dash.exceptions.PreventUpdate()
                    else:
                        data = df[imp_var]
                        mi = get_missing_indices(data)
                        si = spline_interpolation(data, order=param)
                        fig = create_imputation_results_plot(si, mi)
                        last_imputation = pandas.DataFrame()
                        last_imputation[imp_var] = si
                        return fig, last_imputation.to_json()

                # Local Average imputation
                elif method == 'local-average':
                    # Window size must be a positive number
                    if param is None or param < 1:
                        raise dash.exceptions.PreventUpdate()
                    else:
                        data = df[imp_var]
                        mi = get_missing_indices(data)
                        lai = nearest_mean_imputation(data, neighbourhood_size=param)
                        fig = create_imputation_results_plot(lai, mi)
                        last_imputation = pandas.DataFrame()
                        last_imputation[imp_var] = lai
                        return fig, last_imputation.to_json()

                # MLP imputation
                elif method == 'mlp':
                    # Test with param = 0
                    if param is None or param < 1:
                        raise dash.exceptions.PreventUpdate()
                    else:
                        data = df[imp_var]
                        mi = get_missing_indices(data)
                        # Note, unlike other methods, mlp_setup returns a DataFrame
                        mlpi = mlp_setup(data, ar_lag=param)
                        fig = create_imputation_results_plot(mlpi[imp_var].values, mi)
                        return fig, mlpi.to_json()
                else:
                    raise dash.exceptions.PreventUpdate()
    raise dash.exceptions.PreventUpdate()

# Revert Changes button toggles confirm action modal
@app.callback(
    dash.dependencies.Output('revert-changes-modal', 'is_open'),
    dash.dependencies.Input('revert-changes', 'n_clicks'),
    dash.dependencies.Input('revert-cancel', 'n_clicks'),
    dash.dependencies.Input('revert-confirm', 'n_clicks'),
    dash.dependencies.State('revert-changes-modal', 'is_open')
)

def toggle_confirm_revert_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    else:
        return is_open

# Estimate causality button opens modal with causality method options
@app.callback(
    dash.dependencies.Output('causality-modal', 'is_open'),
    dash.dependencies.Input('causality', 'n_clicks'),
    dash.dependencies.Input('causality-cancel', 'n_clicks'),
    dash.dependencies.Input('causality-confirm', 'n_clicks'),
    dash.dependencies.State('causality-modal', 'is_open')
)

def toggle_causality_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    else:
        return is_open

# Changing the causality method changes parameter label and placeholder
@app.callback(
    dash.dependencies.Output('causality-parameter-name', 'children'),
    dash.dependencies.Output('causality-parameter', 'placeholder'),
    [dash.dependencies.Input('causality-method', 'value')]
)

def change_causality_parameter_info(method):
    if method == 'gc':
        placeholder = 'Max lag'
    else:
        placeholder = 'Tau max'
    return [placeholder + ':'], placeholder

# Estimate a causality network of the dataset
@app.callback(
    dash.dependencies.Output('causality-network', 'elements'),
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('causality-method', 'value')],
    [dash.dependencies.Input('causality-variable', 'value')],
    [dash.dependencies.Input('causality-parameter', 'value')],
    [dash.dependencies.Input('causality-confirm', 'n_clicks')]
)

def estimate_causality_network(children, method, variable, param, n):
    ctx = dash.callback_context
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]

            # If user pressed Confirm button
            if current_trigger == 'causality-confirm':
                df = pd.read_json(children).sort_index()
                if method == 'gc':
                    if variable == 'data_all':
                        matrix = granger_causality_matrix(df, maxlag=param)
                        matrix = filter_causality_matrix(matrix)
                        elements = create_causality_elements(matrix, df.columns)
                        return elements
                    else:
                        p_values = granger_causality_by_variable(df, variable, maxlag=param)
                        matrix, related_variables = causality_by_variable_to_matrix(p_values)
                        elements = create_causality_elements(matrix, related_variables)
                        return elements
                else:
                    if variable == 'data_all':
                        link_matrix = run_pcmci(df, tau_max=param)
                        matrix = parse_link_matrix(link_matrix, df.columns)
                        elements = create_causality_elements(matrix, df.columns)
                        return elements
                    else:
                        print('Not implemented yet!')
                        raise dash.exceptions.PreventUpdate()

    raise dash.exceptions.PreventUpdate()

# Parameter Search button (in SARIMAX) opens modal with
# heatmap and plot of criterion values
@app.callback(
    dash.dependencies.Output('uni-param-search-modal', 'is_open'),
    [dash.dependencies.Input('uni-param-search', 'n_clicks')],
    [dash.dependencies.Input('uni-param-search-close', 'n_clicks')],
    [dash.dependencies.State('uni-param-search-modal', 'is_open')]
)

def toggle_uni_param_search_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# Univariate Parameter Search is only available if there is no missing data and exactly one variable is provided
# Multivariate Parameter Search is only available if there is no missing data and at least two variables have been selected
@app.callback(
    dash.dependencies.Output('uni-param-search', 'disabled'),
    dash.dependencies.Output('multi-param-search', 'disabled'),
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('forecasting-variables', 'value')]
)

def check_uni_param_search_modal_availability(data, available_vars):
    ctx = dash.callback_context
    if ctx.triggered:
        if available_vars is None or len(available_vars) == 0:
            return True, True
        elif data_has_missingness(data, available_vars):
            return True, True
        elif len(available_vars) == 1:
            return False, True
        else:
            return True, False
    return True, True

# Enable Univariate Parameter Search once maximum p and q parameters have been provided
@app.callback(
    dash.dependencies.Output('run-uni-param-search', 'disabled'),
    [dash.dependencies.Input('uni-limit_p', 'value')],
    [dash.dependencies.Input('uni-limit_q', 'value')],
    [dash.dependencies.Input('uni-itr', 'value')],
    [dash.dependencies.Input('uni-seasonality', 'value')]
)

def check_uni_param_search_availability(max_p, max_q, itr, s):
    # Maximum p and q parameters must be provided
    if max_p is None or max_q is None:
        return True
    if max_p < 1 or max_q < 1:
        return True
    # Integration, if any, must be a non-negative number
    if not itr is None and itr < 0:
        return True
    # Seasonality, if any, must be a non-negative number
    if not s is None and s < 0:
        return True
    return False

# Perform parameter search with given parameters
# Alternatively, change plots with new criterion
@app.callback(
    dash.dependencies.Output('parameter-heatmap', 'figure'),
    dash.dependencies.Output('uni-parameter-minimum-plot', 'figure'),
    dash.dependencies.Output('last-param-search', 'children'),
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('run-uni-param-search', 'n_clicks')],
    [dash.dependencies.Input('uni-limit_p', 'value')],
    [dash.dependencies.Input('uni-limit_q', 'value')],
    [dash.dependencies.Input('uni-itr', 'value')],
    [dash.dependencies.Input('uni-seasonality', 'value')],
    [dash.dependencies.Input('uni-criterion', 'value')],
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('last-param-search', 'children')]
)

def uni_param_search(children, n_clicks, limit_p, limit_q, itr, s, crt, val, last_param_search):
    # Provide at least one of each parameter
    if limit_p is None or limit_q is None or limit_p < 1 or limit_q < 1:
        raise dash.exceptions.PreventUpdate()

    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        # Button has been pressed
        if trigger == 'run-uni-param-search':
            df = pd.read_json(children).sort_index()

            # Runs parameter search with no seasonality by default
            if s is None:
                s = 0
            elif s < 0:
                raise dash.exceptions.PreventUpdate()

            # Runs parameter search with no integration by default
            if itr is None:
                itr = 0
            elif itr < 0:
                raise dash.exceptions.PreventUpdate()

            # Calculate heatmap and plot of minimums
            heatmaps = param_heatmap(df[val], limit_p, limit_q, itr, s=s)
            mins = param_min_search(heatmaps, limit_p, limit_q)

            # Create figures for heatmap and plot of minimums
            fig_heatmap = px.imshow(heatmaps[crt], color_continuous_scale='RdBu_r')
            fig_plot = px.line(y=mins[crt]['mins'], x=mins[crt]['parameters'], template='ggplot2')

            # Store heatmap and minimums (in json)
            to_store = {'heatmap': {}}
            for key in heatmaps.keys():
                # Numpy arrays must be converted to lists
                to_store['heatmap'][key] = heatmaps[key].tolist()
            to_store['mins'] = mins
            param_res = json.dumps(to_store)

            return fig_heatmap, fig_plot, param_res

        # New criterion was selected
        elif trigger == 'uni-criterion':
            if last_param_search is None:
                raise dash.exceptions.PreventUpdate()
            else:
                last_param_search = json.loads(last_param_search)
                fig_heatmap = px.imshow(last_param_search['heatmap'][crt], color_continuous_scale='RdBu_r')
                fig_plot = px.line(y=last_param_search['mins'][crt]['mins'], x=last_param_search['mins'][crt]['parameters'], template='ggplot2')
                return fig_heatmap, fig_plot, no_update

    # Button was not pressed and criteria was not updated
    raise dash.exceptions.PreventUpdate()

# Parameter search button (in VAR) opens modal with
# plot of criterion values
@app.callback(
    dash.dependencies.Output('multi-param-search-modal', 'is_open'),
    [dash.dependencies.Input('multi-param-search', 'n_clicks')],
    [dash.dependencies.Input('multi-param-search-close', 'n_clicks')],
    [dash.dependencies.State('multi-param-search-modal', 'is_open')]
)

def toggle_multi_param_search_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# Enable Univariate Parameter Search once maximum p parameter have been provided
@app.callback(
    dash.dependencies.Output('run-multi-param-search', 'disabled'),
    [dash.dependencies.Input('multi-limit-p', 'value')]
)

def check_multi_param_search_availability(max_p):
    return max_p is None or max_p < 2

# Perform parameter search with given maximum p
# Operation is relatively quick, results do not need to be stored
@app.callback(
    dash.dependencies.Output('multi-parameter-minimum-plot', 'figure'),
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('run-multi-param-search', 'n_clicks')],
    [dash.dependencies.Input('multi-limit-p', 'value')],
    [dash.dependencies.Input('forecasting-variables', 'value')]
)

def multi_param_search(children, n_clicks, limit_p, vals):
    if limit_p is None or limit_p < 2:
        raise dash.exceptions.PreventUpdate()
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'run-multi-param-search':
            df = pd.read_json(children).sort_index()
            df = df[vals]
            res = var_param_search(df, limit_p)
            vals = pandas.DataFrame()
            vals['aic'] = res['aic']
            vals['bic'] = res['bic']
            fig = px.line(vals, template='ggplot2')
            return fig
    raise dash.exceptions.PreventUpdate()

# Run SARIMAX option is only enabled once all parameters have valid values
@app.callback(
    dash.dependencies.Output('run-sarimax', 'disabled'),
    # SARIMAX parameters
    [dash.dependencies.Input('univariate-linear-autoregressive-parameter', 'value')],
    [dash.dependencies.Input('integration-parameter', 'value')],
    [dash.dependencies.Input('moving-average-parameter', 'value')],
    [dash.dependencies.Input('seasonal-parameter', 'value')],
    # General forecasting parameters
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('current-data', 'children')],
)

def check_sarimax_availability(p, d, q, s, available_vars, split, forecast_window, data):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        # Get dataframe of selected data
        df = pd.read_json(data).sort_index()
        df = df[available_vars]
        # p, q and d values must be provided
        if p is None or d is None or q is None:
            return True
        # p, q and d parameters must be natural numbers (including 0)
        if p < 0 or d < 0 or q < 0:
            return True
        # If s is provided, it must also be a natural number (including 0)
        elif not s is None and s < 0:
            return True
        # Split must be a number equal or above 10
        elif split is None or split < 10:
             return True
        # Exactly one variable must be selected
        if len(available_vars) != 1:
            return True
        # If there are missing indices in the selected data
        elif data_has_missingness(data, available_vars):
            return True
        # Forecast window (if any) must be a positive number
        elif not forecast_window is None and forecast_window < 1:
            return True
        else:
            return False
    return True

# Run VAR option is only enabled once all parameters have valid values
@app.callback(
    dash.dependencies.Output('run-var', 'disabled'),
    # VAR parameters
    [dash.dependencies.Input('multivariate-linear-autoregressive-parameter', 'value')],
    # General forecasting parameters
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('current-data', 'children')],
)

def check_var_availability(p, available_vars, split, forecast_window, data):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        # p parameter is non-optional and must be a non-negative integer
        if p is None or p < 0:
            return True
        # Split must be a number equal or above 10
        elif split is None or split < 10:
            return True
        # Multiple variables must be selected
        elif len(available_vars) < 2:
            return True
        # Forecast window (if any) must be a positive number
        elif not forecast_window is None and forecast_window < 1:
            return True
        # If there are missing indices in the selected data
        elif data_has_missingness(data, available_vars):
            return True
        else:
            return False
    return True

# Run Neural Network option is only enabled once all parameters have valid values
@app.callback(
    dash.dependencies.Output('run-nn', 'disabled'),
    [dash.dependencies.Input('neural-network-model', 'value')],
    [dash.dependencies.Input('order', 'value')],
    [dash.dependencies.Input('output-size', 'value')],
    [dash.dependencies.Input('nn-hyperparameters', 'children')],
    # General forecasting parameters
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('current-data', 'children')],
)

def check_nn_availability(nn_model, order, output_size, hyperparam, available_vars, split, forecast_window, data):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        # Check hyperparameters by model
        hyperparam = json.loads(hyperparam)
        # Bi-LSTM hyperparameters
        if nn_model == 'Bi-LSTM':
            # First layer nodes must be equal or above 1
            if hyperparam['lstm-first-layer-nodes'] is None or hyperparam['lstm-first-layer-nodes'] < 1:
                return True
            # Second layer nodes, if given, must be equal or above 1
            elif not hyperparam['lstm-second-layer-nodes'] is None and hyperparam['lstm-second-layer-nodes'] < 1:
                return True
        elif nn_model == 'CNN':
            # First layer filters must be equal or above 1
            if hyperparam['cnn-first-layer-filters'] is None or hyperparam['cnn-first-layer-filters'] < 1:
                return True
            # First layer kernels must be equal or above 1
            elif hyperparam['cnn-first-layer-kernel'] is None or hyperparam['cnn-first-layer-kernel'] < 1:
                return True
            # Second layer filters, if given must be equal or above 1
            elif not hyperparam['cnn-second-layer-filters'] is None and hyperparam['cnn-second-layer-filters'] < 1:
                return True
            # Second layer kernels, if given must be equal or above 1
            elif not hyperparam['cnn-second-layer-kernel'] is None and hyperparam['cnn-second-layer-kernel'] < 1:
                return True
        if not hyperparam['dropout'] is None:
            # Dropout, if given, must be between 0 and 1 (exclusive)
            if hyperparam['dropout'] < 0 or hyperparam['dropout'] >= 1:
                return True
            elif hyperparam['t'] is None or hyperparam['t'] < 1:
                return True
        # Number of epochs must be a number equal or above 10
        if hyperparam['epochs'] is None or hyperparam['epochs'] < 10:
            return True
        # Batch size must be a positive number
        if hyperparam['batch_size'] is None or hyperparam['batch_size'] < 1:
            return True

        # Output size must be a positive number
        if output_size is None or output_size < 1:
            return True
        # Sliding window size must be a non-negative number
        elif order is None or order < 0:
            return True

        # Split must be a number equal or above 10
        elif split is None or split < 10:
            return True
        # At least one variable must be selected
        elif len(available_vars) < 1:
            return True
        # If there are missing indices in the selected data
        elif data_has_missingness(data, available_vars):
            return True
        # Forecast window (if any) must be a positive number
        elif not forecast_window is None and forecast_window < 1:
            return True

        return False
    return True

# Run SARIMAX model with given parameters
@app.callback(
    dash.dependencies.Output('last-sarimax-training', 'children'),
    dash.dependencies.Output('download-sarimax-model', 'data'),
    # SARIMAX parameters
    [dash.dependencies.Input('univariate-linear-autoregressive-parameter', 'value')],
    [dash.dependencies.Input('integration-parameter', 'value')],
    [dash.dependencies.Input('moving-average-parameter', 'value')],
    [dash.dependencies.Input('seasonal-parameter', 'value')],
    # General forecasting parameters
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('run-sarimax', 'n_clicks')],
    [dash.dependencies.Input('export-model', 'value')],
)

def train_sarimax(p, d, q, s, data, available_vars, split, forecast_window, n_clicks, export_status):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]

            if current_trigger == 'run-sarimax':
                df = pd.read_json(data).sort_index()

                var = available_vars[0]
                df = df[[var]]

                # Seasonal (s) parameter is optional and may not be provided
                if s is None:
                    s = 0

                # If Export model option has been selected
                export = (export_status == ['export'])

                last_sarimax, results_string = sarimax_process(df, (p, d, q, s), split, var, forecast_window, export=export)

                download_content = no_update
                if export and not results_string is None:
                    download_content = dict(content=results_string, filename='results_sarimax.txt')

                return last_sarimax, download_content

    # If button has not been pressed
    raise dash.exceptions.PreventUpdate()

# Run VAR model with given parameters
@app.callback(
    dash.dependencies.Output('last-var-training', 'children'),
    dash.dependencies.Output('download-var-model', 'data'),
    # VAR parameters
    [dash.dependencies.Input('multivariate-linear-autoregressive-parameter', 'value')],
    # General forecasting parameters
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('run-var', 'n_clicks')],
    [dash.dependencies.Input('export-model', 'value')]
)

def train_var(p, data, available_vars, split, forecast_window, n_clicks, export_status):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]

            if current_trigger == 'run-var':
                df = pd.read_json(data).sort_index()

                df = df[available_vars]

                # If Export model option has been selected
                export = (export_status == ['export'])

                last_var, results_string = var_process(df, p, split, available_vars, forecast_window, export=export)

                download_content = no_update
                if export and not results_string is None:
                    download_content = dict(content=results_string, filename='results_var.txt')

                return last_var, download_content

    # If button has not been pressed
    raise dash.exceptions.PreventUpdate()

# Run Neural Network with given parameters and updates results modal
@app.callback(
    dash.dependencies.Output('last-nn-training', 'children'),
    dash.dependencies.Output('nn-results', 'disabled'),
    dash.dependencies.Output('loss-results-plot', 'figure'),
    dash.dependencies.Output('training-set-res', 'children'),
    dash.dependencies.Output('training-set-var', 'options'),
    dash.dependencies.Output('training-set-var', 'value'),
    dash.dependencies.Output('download-nn-architecture', 'data'),
    # Neural Network parameters
    [dash.dependencies.Input('neural-network-model', 'value')],
    [dash.dependencies.Input('order', 'value')],
    [dash.dependencies.Input('output-size', 'value')],
    [dash.dependencies.Input('seed', 'value')],
    # General forecasting parameters
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('run-nn', 'n_clicks')],
    [dash.dependencies.Input('nn-hyperparameters', 'children')],
    [dash.dependencies.Input('export-model', 'value')],
)

def train_nn(model, window_size, output_size, seed, data, available_vars, split, forecast_window, n_clicks, hyperparam, export_status):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        for trigger in ctx.triggered:
            current_trigger = trigger['prop_id'].split('.')[0]

            if current_trigger == 'run-nn':

                df = pd.read_json(data).sort_index()
                hyperparam_data = json.loads(hyperparam)

                df = df[available_vars]

                # If Export model option has been selected
                export = (export_status == ['export'])

                last_nn, nn_res, model_arch_json = nn_process(df, model, window_size, split, available_vars, forecast_window, hyperparam_data, output_size=output_size, seed=seed, export=export)

                loss_data = pandas.DataFrame([nn_res['loss'], nn_res['val_loss']]).transpose()
                loss_data.columns = ['loss', 'val_loss']

                # Change to display selected variable
                training_set_res = pandas.DataFrame()
                for i in range(len(available_vars)):
                    training_set_res[available_vars[i]] = np.array(nn_res['training'])[:, i]
                    # Due to multi-output, results may differ in length
                    # Offset is represented by null values (ignored by plots)
                    training_set_res = training_set_res.join(pandas.DataFrame(np.array(nn_res['training_res'][:, :, i]), columns=[available_vars[i] + '_pred']))

                loss_plot = px.line(loss_data, template='ggplot2')
                loss_plot.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)

                training_set_res = training_set_res.to_json()

                var_options = []
                for variable in available_vars:
                    var_options.append({'label': variable, 'value': variable})

                download_content = no_update
                if export and not model_arch_json is None:
                    download_content = dict(content=model_arch_json, filename='results_nn.json')

                return last_nn, False, loss_plot, training_set_res, var_options, available_vars[0], download_content

    # If button has not been pressed
    raise dash.exceptions.PreventUpdate()

# Change in selected training set variable changes related plot
@app.callback(
    dash.dependencies.Output('training-set-fit', 'figure'),
    [dash.dependencies.Input('training-set-res', 'children')],
    [dash.dependencies.Input('training-set-var', 'value')]
)

def update_training_set_res_plot(training_set_res, var):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        training_set_res = pd.read_json(training_set_res)

        training_set_plot = px.line(training_set_res, y=[var, var + '_pred'], template='ggplot2')
        training_set_plot.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)

        return training_set_plot
    raise dash.exceptions.PreventUpdate()

# Variable dropdown in model train results plot is enabled after a model is trained
@app.callback(
    dash.dependencies.Output('chosen-variable-forecast', 'disabled'),
    [dash.dependencies.Input('last-sarimax-training', 'children')],
    [dash.dependencies.Input('last-var-training', 'children')],
    [dash.dependencies.Input('last-nn-training', 'children')]
)

def enable_trained_variable_dropdown(last_sarimax, last_var, last_nn):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'sarimax' in trigger or 'var' in trigger or 'nn' in trigger:
            return False
    raise dash.exceptions.PreventUpdate()

# Updates all results info after any model has been trained
@app.callback(
    dash.dependencies.Output('model-performances', 'children'),
    dash.dependencies.Output('last-test-size', 'children'),
    dash.dependencies.Output('chosen-variable-forecast', 'options'),
    dash.dependencies.Output('chosen-variable-forecast', 'value'),
    dash.dependencies.Output('last-trained-model', 'children'),
    # Sarimax related inputs
    [dash.dependencies.Input('last-sarimax-training', 'children')],
    # VAR related inputs
    [dash.dependencies.Input('last-var-training', 'children')],
    # Neural Network related inputs
    [dash.dependencies.Input('last-nn-training', 'children')],
    # General inputs
    [dash.dependencies.Input('model-performances', 'children')],
    [dash.dependencies.Input('last-test-size', 'children')],
    [dash.dependencies.Input('last-trained-model', 'children')]
)

def update_all_results(last_sarimax, last_var, last_nn, performances, last_test, last_model):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        # Sarimax was trained
        if 'sarimax' in trigger:
            # Load stored SARIMAX data
            sarimax_data = json.loads(last_sarimax)

            var = sarimax_data['var']
            error = sarimax_data['error']
            residuals = np.array(sarimax_data['residuals']).flatten()

            var_options = [{'label': var, 'value': var}]
            value = var
            nobs = sarimax_data['nobs']

            results, last_test = update_model_results('linear-univariate', [var], performances, error, residuals, last_test, nobs)

            return results, last_test, var_options, value, 'sarimax'

        elif 'var' in trigger:
            var_data = json.loads(last_var)

            predicted_variables = var_data['var']
            var_options = []
            for variable in predicted_variables:
                var_options.append({'label': variable, 'value': variable})
            value = predicted_variables[0]
            residuals = np.array(var_data['residuals'])
            nobs = var_data['nobs']

            results, last_test = update_model_results('linear-multivariate', predicted_variables, performances, var_data['error'], residuals, last_test, nobs)
            return results, last_test, var_options, value, 'var'

        elif 'nn' in trigger:
            nn_data = json.loads(last_nn)

            predicted_variables = nn_data['var']
            var_options = []
            for variable in predicted_variables:
                var_options.append({'label': variable, 'value': variable})
            value = predicted_variables[0]
            residuals = np.array(nn_data['residuals'])
            nobs = nn_data['nobs']

            results, last_test = update_model_results('neural-networks', predicted_variables, performances, nn_data['error'], residuals, last_test, nobs)
            return results, last_test, var_options, value, 'nn'

    raise dash.exceptions.PreventUpdate()

@app.callback(
    dash.dependencies.Output('forecast-result', 'figure'),
    [dash.dependencies.Input('chosen-variable-forecast', 'value')],
    # Sarimax related inputs
    [dash.dependencies.Input('last-sarimax-training', 'children')],
    # VAR related inputs
    [dash.dependencies.Input('last-var-training', 'children')],
    # Neural Network related inputs
    [dash.dependencies.Input('last-nn-training', 'children')],
    [dash.dependencies.Input('last-trained-model', 'children')]
)

def update_results_figure(display_variable, last_sarimax, last_var, last_nn, last_model):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        # Have to search all triggers for 'chosen-variable-forecast'
        for trigger in ctx.triggered:
            if trigger['prop_id'].split('.')[0] == 'chosen-variable-forecast':
                offset = 0
                if last_model == 'sarimax':
                    model_data = json.loads(last_sarimax)
                elif last_model == 'var':
                    model_data = json.loads(last_var)
                elif last_model == 'nn':
                    model_data = json.loads(last_nn)
                    # There is a difference in testing set sizes between neural networks and linear models
                    # Offset moves the results horizontally to the right
                    offset = 1
                else:
                    raise dash.exceptions.PreventUpdate()

                df = pd.read_json(model_data['data']).sort_index()
                pred = pd.read_json(model_data['predicted']).sort_index()[display_variable].values.flatten()
                fig_data = create_forecasting_plot(df, display_variable, model_data['split'] + offset, pred, forecast=model_data['forecast'], conf_int=model_data['conf_int'])
                return fig_data

    raise dash.exceptions.PreventUpdate()

# Preferences button opens Neural Network preferences modal
@app.callback(
    dash.dependencies.Output('nn-preferences-modal', 'is_open'),
    [dash.dependencies.Input('nn-preferences', 'n_clicks')],
    [dash.dependencies.Input('nn-preferences-close', 'n_clicks')],
    [dash.dependencies.State('nn-preferences-modal', 'is_open')]
)

def toggle_nn_preferences_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# Changes to any Neural Network preference updates the stored hyperparameters
@app.callback(
    dash.dependencies.Output('nn-hyperparameters', 'children'),
    [dash.dependencies.Input('nn-hyperparameters', 'children')],
    # LSTM related hyperparameters
    [dash.dependencies.Input('lstm-first-layer-nodes', 'value')],
    [dash.dependencies.Input('lstm-first-layer-activation', 'value')],
    [dash.dependencies.Input('lstm-second-layer-nodes', 'value')],
    [dash.dependencies.Input('lstm-second-layer-activation', 'value')],
    # CNN related hyperparameters
    [dash.dependencies.Input('cnn-first-layer-filters', 'value')],
    [dash.dependencies.Input('cnn-first-layer-kernel', 'value')],
    [dash.dependencies.Input('cnn-first-layer-activation', 'value')],
    [dash.dependencies.Input('cnn-second-layer-filters', 'value')],
    [dash.dependencies.Input('cnn-second-layer-kernel', 'value')],
    [dash.dependencies.Input('cnn-second-layer-activation', 'value')],
    # General parameters
    [dash.dependencies.Input('dropout', 'value')],
    [dash.dependencies.Input('t-parameter', 'value')],
    [dash.dependencies.Input('nn-epochs', 'value')],
    [dash.dependencies.Input('batch-size', 'value')]
)

def update_nn_hyperparameters(hyperparam, lstm_n1, lstm_a1, lstm_n2, lstm_a2, cnn_f1, cnn_k1, cnn_a1, cnn_f2, cnn_k2, cnn_a2, d, t, epochs, batch_size):
    hyperparam = json.loads(hyperparam)
    hyperparam['lstm-first-layer-nodes'] = lstm_n1
    hyperparam['lstm-first-layer-activation'] = lstm_a1
    hyperparam['lstm-second-layer-nodes'] = lstm_n2
    hyperparam['lstm-second-layer-activation'] = lstm_a2
    hyperparam['cnn-first-layer-filters'] = cnn_f1
    hyperparam['cnn-first-layer-kernel'] = cnn_k1
    hyperparam['cnn-first-layer-activation'] = cnn_a1
    hyperparam['cnn-second-layer-filters'] = cnn_f2
    hyperparam['cnn-second-layer-kernel'] = cnn_k2
    hyperparam['cnn-second-layer-activation'] = cnn_a2
    hyperparam['dropout'] = d
    hyperparam['t'] = t
    hyperparam['epochs'] = epochs
    hyperparam['batch_size'] = batch_size
    return json.dumps(hyperparam)

# Results button opens Neural Network results modal
@app.callback(
    dash.dependencies.Output('nn-results-modal', 'is_open'),
    [dash.dependencies.Input('nn-results', 'n_clicks')],
    [dash.dependencies.Input('nn-results-close', 'n_clicks')],
    [dash.dependencies.State('nn-results-modal', 'is_open')]
)

def toggle_nn_results_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    else:
        return is_open

# New model performance or variable performance selection changes plot with correlation of residuals
@app.callback(
    dash.dependencies.Output('residual-autocorrelation', 'figure'),
    [dash.dependencies.Input('model-performances', 'children')],
    [dash.dependencies.Input('model-status', 'hoverData')],
    [dash.dependencies.Input('res-method', 'value')]
)

def update_residual_plot(performances, hover_data, res_method):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        perf_data = json.loads(performances)
        model = None
        var = None

        # Check if any model performance bar has been selected or if residual plot was changed
        for trigger in ctx.triggered:
            if trigger['prop_id'].split('.')[0] == 'model-status' or \
            (trigger['prop_id'].split('.')[0] == 'res-method' and not hover_data is None):
                label = hover_data['points'][0]['label']
                [model, var] = label.split(' - ')

        # If any bar was hovered
        if not (model is None) or not (var is None):
            residuals = pd.read_json(perf_data[model]['residuals']).sort_index()
            if res_method == 'acf':
                acf_data, pacf_data = get_acf_and_pacf(residuals[var], alpha=None)
                fig = create_acf_plot(acf_data, 'res-acf', nobs=perf_data[model]['nobs'])
            else:
                fig = px.histogram(residuals[var], template='ggplot2')

            fig.update_layout(legend=fig_legend_layout, margin=fig_margin_layout)

            return fig

    raise dash.exceptions.PreventUpdate()

# Update to stored model performances changes model error bar plot
@app.callback(
    dash.dependencies.Output('model-status', 'figure'),
    [dash.dependencies.Input('model-performances', 'children')]
)

def update_model_status_plot(performances):
    perf_data = json.loads(performances)
    # Initial startup
    if perf_data == {}:
        raise dash.exceptions.PreventUpdate()
    # At least one model has been trained
    else:
        fig = get_performance_bar_plot(perf_data)
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)
