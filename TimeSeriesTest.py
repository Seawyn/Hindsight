import dash
from dash.dash import no_update
from Structure.helperfunctions import *
from Structure.modals import *

# Check csv input
if len(sys.argv) > 1:
    filepath = sys.argv[1]
    valid_csv = check_file(filepath)
    if not valid_csv:
        raise ValueError('Please input a valid csv file')
else:
    filepath = '../Datasets/Iowa_Income.csv'


external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

initial_fig, fig_acf, fig_pacf, df, options, values = setup(filepath)
fig_trend, fig_seasonal, fig_residual = setup_seasonal_decomposition(df, values[0])

neural_network_options = [{'label': 'Bi-LSTM', 'value': 'Bi-LSTM'}, {'label': 'CNN', 'value': 'CNN'}]

app.layout = html.Div(children=[
    dbc.Card([
        dbc.CardBody([
            # First "row" of panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='ts',
                                figure=initial_fig,
                                config={'displayModeBar': False},
                                style={'display': 'inline-block'}
                            ),
                            # Variable selection dropdown
                            dcc.Dropdown(
                                id='variable-selection',
                                options=options,
                                value=values,
                                multi=True,
                            )
                        ])
                    ], style={'height': '100%'})
                ], width=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Dropdown(id='chosen-variable', options=[{'label': 'Dataset', 'value': 'data_all'}] + options, value='data_all', clearable=False, style={'marginBottom': '10px'}),
                            dbc.Button('First Difference', id='first-difference', color='info', n_clicks=0, style={'marginBottom': '10px', 'width': '250px'}),
                            html.Div(id='differences-text', children='Dataset has not been differenced'),
                            html.Div(id='adf-test', children='ADF p-value'),
                            html.Div(id='missing-indices', children='Variable has no missing indices'),
                            autocorrelation_modal,
                            html.Br(),
                            seasonal_decomposition_modal,
                            html.Br(),
                            imputation_modal,
                            html.Br(),
                            dbc.Button('Revert Changes', id='revert-changes', color='info', n_clicks=0, style={'marginBottom': '10px', 'width': '250px'}),
                        ])
                    ], style={'height': '100%'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            cyto.Cytoscape(
                                id='causality-network',
                                layout={'name': 'circle'},
                                style={'width': '600px', 'height':'500px'},
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
                            dcc.RadioItems(
                                options=[
                                    {'label': 'Granger Causality', 'value': 'gc'},
                                    {'label': 'PCMCI', 'value': 'pcmci'}
                                ], id='causality-method', value='gc', labelStyle={'display': 'inline-block'}, style={'display': 'inline-block'}
                            ),
                            dbc.Button('Estimate Causality', id='causality', color='info', n_clicks=0)
                        ])
                    ])
                ], width=5)
            ]),
            html.Br(),

            # Second "row" of panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # Forecasting tabs
                            dcc.Tabs(
                                id='forecasting-tabs',
                                value='linear-univariate',
                                children=[
                                    # SARIMAX tab
                                    dcc.Tab(
                                        label='Linear Univariate (SARIMAX)',
                                        value='linear-univariate',
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
                                                    dbc.Button('Run SARIMAX', id='run-sarimax', color='info', n_clicks=0, style={'width': '100%'})
                                                ])
                                            ])
                                        ]
                                    ),
                                    # VAR tab
                                    dcc.Tab(
                                        label='Linear Multivariate (VAR)',
                                        value='linear-multivariate',
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
                                                    dbc.Button('Run VAR', id='run-var', color='info', n_clicks=0, style={'width': '100%'})
                                                ])
                                            ])
                                        ]
                                    ),
                                    # Neural Networks tab
                                    dcc.Tab(
                                        label='Neural Networks',
                                        value='neural-networks',
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
                                                        dcc.Input(id='seed', type='number', min=0, max=(2 ** 16), placeholder='Random Seed', style={'width': '100%'}),
                                                    ])
                                                ])
                                            ]),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    nn_preferences_modal
                                                ]),
                                                dbc.Col([
                                                    dbc.Button('Run Neural Network', id='run-nn', color='info', n_clicks=0, style={'width': '100%'})
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
                                        dcc.Dropdown(id='forecasting-variables', options=options, value=values, multi=True, style={'width': '100%'}),
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
                            ])
                        ])
                    ], style={'height': '100%'})
                ], width=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='forecast-result',
                                figure=get_empty_plot('Please run a model first'),
                                style={'display': 'inline-block'}
                            ),
                            dcc.Dropdown(id='chosen-variable-forecast', clearable=False, disabled=True)
                        ])
                    ], style={'height': '100%'})
                ])
            ]),
            html.Br(),

            # Third "row" of panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='residual-autocorrelation', figure=get_empty_plot('Please run a model first'), style={'display': 'inline-block'}),
                            dcc.RadioItems(
                                options=[
                                    {'label': 'Autocorrelation Function', 'value': 'acf'},
                                    {'label': 'Histogram', 'value': 'histogram'},
                                ],
                                value='acf',
                                id='res-method'
                            )
                        ])
                    ], style={'height': '100%'})
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='model-status', figure=get_empty_plot('Please run a model first'))
                        ])
                    ], style={'height': '100%'})
                ], width=6)
            ]),
        ])
    ], color='#FCFCC6'),

    # Hidden div holds the original dataset
    html.Div(id='dataset', children=df.to_json(), style={'display': 'none'}),

    # Hidden div holds the current dataset
    html.Div(id='current-data', children=df.to_json(), style={'display': 'none'}),

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
    #       - Add PCMCI
    #       - Parameter selection: order
    #       - Focus on a single variable
    # TODO: Ask for Dataset upon startup
    # TODO: Export model and/or dataset
    # TODO: Neural Networks:
    #       - Confidence interval
    #       - Multiple output
    #       - Multiple models
    # TODO: Loading status for training/forecasting models
    # TODO: Change label position in each plot

])

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
    [dash.dependencies.Input('revert-changes', 'n_clicks')],
    [dash.dependencies.Input('imputation-confirm', 'n_clicks')],
    [dash.dependencies.Input('last-imputation', 'children')]
)

def update_ts(n_clicks, children, value, num_dif, orig_data, undo, imp_n, last_imp):
    # Obtain dataset from stored json data and sort by index
    df = pd.read_json(children).sort_index()
    # Get columns with missing values
    imp_vars = []
    cols = get_missing_columns(df)
    for col in cols:
        imp_vars.append({'label': col, 'value': col})
    ctx = dash.callback_context
    # Only with triggers
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        # If First Difference button was pressed
        if trigger == 'first-difference':
            df = difference(df)
            text_dif = 'Dataset has been differenced {} times'.format(num_dif + 1)
            if len(value) < 1:
                fig = get_empty_plot('Please select a variable')
            else:
                fig = px.line(df, y=value)
            return text_dif, fig, df.to_json(), num_dif + 1, imp_vars

        # If Revert Changes button was pressed
        elif trigger == 'revert-changes':
            text_dif = 'Dataset has not been differenced'
            num_dif = 0
            orig = pd.read_json(orig_data).sort_index()
            if len(value) < 1:
                fig = get_empty_plot('Please select a variable')
            else:
                fig = px.line(orig, y=value)
            return text_dif, fig, orig_data, num_dif, imp_vars

        # User confirmed imputation results, dataset is updated
        elif trigger == 'imputation-confirm':
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
                fig = px.line(df, y=value)
            return text_dif, fig, df.to_json(), num_dif, imp_vars

        # If variable selection was changed
        else:
            # Cannot plot empty dataset!
            if len(value) < 1:
                fig = get_empty_plot('Please select a variable')
            else:
                fig = px.line(df, y=value)
                # Only alter plot, number of differences and current data remain unaltered
            return no_update, fig, no_update, no_update, imp_vars

    # Initial startup
    else:
        return no_update, no_update, no_update, no_update, imp_vars

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
    df = pd.read_json(children).sort_index()
    mi = 0
    disabled = False
    # If entire dataset is selected in dropdown
    if value == 'data_all':
        adf_msg = 'Dataset is stationary'
        var = df.columns[0]
        disabled = True
        # If there are no missing values
        if len(get_missing_columns(df)) == 0:
            for col in df.columns:
                adf = adf_pvalue(df[col])
                if adf > 0.05:
                    adf_msg = 'Dataset is not stationary'
        # If there are missing values, ADF test cannot be performed
        else:
            adf_msg = 'Unable to perform ADF test'
            disabled = True
        # Display number of missing indices either way
        for col in df.columns:
            mi += len(get_missing_indices(df[col]))
        mi_msg = 'Dataset has {} missing indices'.format(mi)
    # If specific variable is selected
    else:
        var = value
        mi = len(get_missing_indices(df[var]))
        if mi == 0:
            adf_msg = 'ADF p-value: ' + str(adf_pvalue(df[var]))
        else:
            adf_msg = 'Unable to perform ADF test'
            disabled = True
        mi_msg = 'Variable has {} missing indices'.format(mi)
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
    # Spline interpolation inputs
    [dash.dependencies.Input('spline-interpolation', 'n_clicks')],
    [dash.dependencies.Input('spline-order', 'value')],
    # Local Average interpolation inputs
    [dash.dependencies.Input('window-size', 'value')],
    [dash.dependencies.Input('local-average-interpolation', 'n_clicks')],
    # MLP interpolation inputs
    [dash.dependencies.Input('mlp-imputation-ar-lag', 'value')],
    [dash.dependencies.Input('mlp-imputation', 'n_clicks')]
)

def impute(children, imp_var, spl_n, spl_order, window_size, la_n, mlp_ar, mlp_n):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        df = pd.read_json(children).sort_index()
        # Variable must be selected
        if imp_var is None:
            raise dash.exceptions.PreventUpdate()

        # Spline interpolation
        if trigger == 'spline-interpolation':
            # Spline order must be a number between 1 and 5
            if spl_order is None or spl_order < 1 or spl_order > 5:
                raise dash.exceptions.PreventUpdate()
            else:
                data = df[imp_var]
                si = spline_interpolation(data, order=spl_order)
                fig = px.line(si)
                last_imputation = pandas.DataFrame()
                last_imputation[imp_var] = si
                return fig, last_imputation.to_json()

        # Local Average imputation
        elif trigger == 'local-average-interpolation':
            # Window size must be a positive number
            if window_size is None or window_size < 1:
                raise dash.exceptions.PreventUpdate()
            else:
                data = df[imp_var]
                lai = nearest_mean_imputation(data, neighbourhood_size=window_size)
                fig = px.line(lai)
                last_imputation = pandas.DataFrame()
                last_imputation[imp_var] = lai
                return fig, last_imputation.to_json()

        # MLP imputation
        elif trigger == 'mlp-imputation':
            # Test with mlp_ar = 0
            if mlp_ar is None or mlp_ar < 1:
                raise dash.exceptions.PreventUpdate()
            else:
                data = df[imp_var]
                # Note, unlike other methods, mlp_setup returns a DataFrame
                mlpi = mlp_setup(data, ar_lag=mlp_ar)
                fig = px.line(mlpi[imp_var].values)
                return fig, mlpi.to_json()
        else:
            raise dash.exceptions.PreventUpdate()
    raise dash.exceptions.PreventUpdate()

# Estimate a causality network of the dataset
@app.callback(
    dash.dependencies.Output('causality-network', 'elements'),
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('causality-method', 'value')],
    [dash.dependencies.Input('causality', 'n_clicks')]
)

def estimate_causality_network(children, method, n):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        # If user pressed Estimate Causality button
        if trigger == 'causality':
            df = pd.read_json(children).sort_index()
            if method == 'gc':
                matrix = granger_causality_matrix(df, maxlag=5)
                matrix = filter_causality_matrix(matrix)
                elements = create_causality_elements(matrix, df.columns)
                return elements
        else:
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
            fig_plot = px.line(y=mins[crt]['mins'], x=mins[crt]['parameters'])

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
                fig_plot = px.line(y=last_param_search['mins'][crt]['mins'], x=last_param_search['mins'][crt]['parameters'])
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
            fig = px.line(vals)
            return fig
    raise dash.exceptions.PreventUpdate()

# Run SARIMAX model with given parameters
@app.callback(
    dash.dependencies.Output('last-sarimax-training', 'children'),
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
    [dash.dependencies.Input('run-sarimax', 'n_clicks')]
)

def train_sarimax(p, d, q, s, data, available_vars, split, forecast_window, n_clicks):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'run-sarimax':
            df = pd.read_json(data).sort_index()

            var = available_vars[0]
            df = df[[var]]

            last_sarimax = sarimax_process(df, (p, d, q, s), split, var, forecast_window)

            return last_sarimax

    # If button has not been pressed
    raise dash.exceptions.PreventUpdate()

# Run VAR model with given parameters
@app.callback(
    dash.dependencies.Output('last-var-training', 'children'),
    # VAR parameters
    [dash.dependencies.Input('multivariate-linear-autoregressive-parameter', 'value')],
    # General forecasting parameters
    [dash.dependencies.Input('current-data', 'children')],
    [dash.dependencies.Input('forecasting-variables', 'value')],
    [dash.dependencies.Input('split', 'value')],
    [dash.dependencies.Input('forecast', 'value')],
    [dash.dependencies.Input('run-var', 'n_clicks')]
)

def train_var(p, data, available_vars, split, forecast_window, n_clicks):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'run-var':
            df = pd.read_json(data).sort_index()

            df = df[available_vars]

            last_var = var_process(df, p, split, available_vars, forecast_window)

            return last_var

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
)

def train_nn(model, window_size, output_size, seed, data, available_vars, split, forecast_window, n_clicks, hyperparam):
    ctx = dash.callback_context
    # Prevent update upon startup
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'run-nn':

            df = pd.read_json(data).sort_index()
            hyperparam_data = json.loads(hyperparam)

            df = df[available_vars]

            last_nn, nn_res = nn_process(df, model, window_size, split, available_vars, forecast_window, hyperparam_data, seed=seed)

            loss_data = pandas.DataFrame([nn_res['loss'], nn_res['val_loss']]).transpose()
            loss_data.columns = ['loss', 'val_loss']

            # Change to display selected variable
            training_set_res = pandas.DataFrame()
            for i in range(len(available_vars)):
                training_set_res[available_vars[i]] = np.array(nn_res['training'])[:, i]
                training_set_res[available_vars[i] + '_pred'] =  np.array(nn_res['training_res'][:, 0, 0])

            print(training_set_res)

            loss_plot = px.line(loss_data)

            training_set_res = training_set_res.to_json()

            var_options = []
            for variable in available_vars:
                var_options.append({'label': variable, 'value': variable})

            return last_nn, False, loss_plot, training_set_res, var_options, available_vars[0]

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

        training_set_plot = px.line(training_set_res, y=[var, var + '_pred'])

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
                if last_model == 'sarimax':
                    model_data = json.loads(last_sarimax)
                elif last_model == 'var':
                    model_data = json.loads(last_var)
                elif last_model == 'nn':
                    model_data = json.loads(last_nn)
                else:
                    raise dash.exceptions.PreventUpdate()

                df = pd.read_json(model_data['data']).sort_index()
                pred = pd.read_json(model_data['predicted']).sort_index()[display_variable].values.flatten()
                fig_data = create_forecasting_plot(df, display_variable, model_data['split'], pred, forecast=model_data['forecast'], conf_int=model_data['conf_int'])
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
    [dash.dependencies.Input('forecast-strategy', 'value')]
)

def update_nn_hyperparameters(hyperparam, lstm_n1, lstm_a1, lstm_n2, lstm_a2, cnn_f1, cnn_k1, cnn_a1, cnn_f2, cnn_k2, cnn_a2, d, f_str):
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
    hyperparam['forecast-strategy'] = f_str
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
                fig = px.histogram(residuals)
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
