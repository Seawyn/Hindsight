from Structure.helperfunctions import *

activation_functions_options = [{'label': 'Sigmoid', 'value': 'sigmoid'}, {'label': 'Tanh', 'value': 'tanh'},
                                {'label': 'ReLU', 'value': 'relu'}, {'label': 'ELU', 'value': 'elu'},
                                {'label': 'Swish', 'value': 'swish'}]

forecast_strategy_options = [{'label': 'Recursive', 'value': 'recursive'}, {'label': 'Direct', 'value': 'direct'},
                            {'label': 'MIMO', 'value': 'mimo'}, {'label': 'DIRMO/MISMO', 'value': 'dirmo'}]

autocorrelation_modal = html.Div([
    dbc.Button('Autocorrelation Info', id='autocorrelation', color='success', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Autocorrelation and Partial Autocorrelation Functions'),
        dbc.ModalBody([
            dcc.Graph(id='acf', figure=get_empty_plot('Please select a variable'), config={'displayModeBar': False}),
            dcc.Graph(id='pacf', figure=get_empty_plot('Please select a variable'), config={'displayModeBar': False})
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='autocorrelation-close', className='ml-auto')
        )
    ], id='autocorrelation-modal', size='lg'),
])

seasonal_decomposition_modal = html.Div([
    dbc.Button('Seasonal Decomposition', id='seasonal-decomposition', color='success', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Seasonal Decomposition'),
        dbc.ModalBody([
            dcc.Graph(id='trend', figure=get_empty_plot('Please select a variable'), config={'displayModeBar': False}),
            dcc.Graph(id='seasonal', figure=get_empty_plot('Please select a variable'), config={'displayModeBar': False}),
            dcc.Graph(id='residual', figure=get_empty_plot('Please select a variable'), config={'displayModeBar': False})
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='seasonal-decomposition-close', className='ml-auto')
        )
    ], id='seasonal-decomposition-modal', size='lg'),
])

imputation_modal = html.Div([
    dbc.Button('Imputation', id='imputation', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Imputation'),
        dbc.ModalBody([
            'Variable to impute:',
            dcc.Dropdown(id='imputation-variables'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    'Method:',
                    dcc.Dropdown(id='imputation-method', options=[
                        {'label': 'Spline Interpolation', 'value': 'spline'},
                        {'label': 'Local Average', 'value': 'local-average'},
                        {'label': 'MLP', 'value': 'mlp'}
                    ], value='spline', clearable=False)
                ]),
                dbc.Col([
                    html.Div(id='imputation-parameter-name'),
                    dcc.Input(id='imputation-parameter', type='number', style={'width': '100%'})
                ])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(width=4),
                dbc.Col(width=4),
                dbc.Col([
                    dbc.Button('Impute', id='perform-imputation', outline=True, color='dark', style={'width': '80%'}),
                ], width=4)
            ]),
            html.Br(),
            dcc.Graph(id='imputation-results', figure=get_empty_plot('Imputation results will be displayed here'))
        ]),
        dbc.ModalFooter(children=[
            dbc.Button('Cancel', id='imputation-cancel', className='ml-auto'),
            dbc.Button('Confirm', id='imputation-confirm', className='ml-auto')
        ])
    ], id='imputation-modal', size='lg'),
])

confirm_revert_modal = html.Div([
    dbc.Button('Revert Changes', id='revert-changes', outline=True, color='danger', n_clicks=0, style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Confirm Revert Changes'),
        dbc.ModalBody([
            'Are you sure you want to revert changes?',
            html.Br(),
            html.A('This operation cannot be undone.', style={'fontWeight': 'bold'}),
        ]),
        dbc.ModalFooter([
            dbc.Button('Cancel', id='revert-cancel', className='ml-auto'),
            dbc.Button('Confirm', id='revert-confirm', color='danger', className='ml-auto'),
        ])
    ], id='revert-changes-modal')
])

causality_modal = html.Div([
    dbc.Button('Estimate Causality', id='causality'),
    dbc.Modal([
        dbc.ModalHeader('Estimate Causality'),
        dbc.ModalBody([
            'Method:',
            dcc.Dropdown(id='causality-method', clearable=False, options=[
                {'label': 'Granger Causality', 'value': 'gc'},
                {'label': 'PCMCI', 'value': 'pcmci'}
            ], value='gc'),
            html.Br(),
            'Variable:',
            dcc.Dropdown(id='causality-variable', clearable=False),
            html.Br(),
            html.Div(id='causality-parameter-name', children=['Max lag:']),
            dcc.Input(id='causality-parameter', type='number')
        ]),
        dbc.ModalFooter(children=[
            dbc.Button('Cancel', id='causality-cancel', className='ml-auto'),
            dbc.Button('Confirm', id='causality-confirm', className='ml-auto')
        ])
    ], id='causality-modal')
])

uni_param_search_modal = html.Div([
    dbc.Button('Parameter Search', id='uni-param-search', color='success', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Parameter Search (SARIMA)'),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    'Max p:',
                    dcc.Input(id='uni-limit_p', type='number', placeholder='Limit of p'),
                ], width=4),
                dbc.Col([
                    'Max q:',
                    dcc.Input(id='uni-limit_q', type='number', placeholder='Limit of q'),
                ], width=4),
                dbc.Col([
                    'Integration (Optional):',
                    dcc.Input(id='uni-itr', type='number', placeholder='Integration'),
                ], width=4)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    'Seasonality (Optional):',
                    dcc.Input(id='uni-seasonality', type='number', placeholder='Seasonality'),
                ], width=4),
                dbc.Col(width=4),
                dbc.Col([
                    html.Br(),
                    dbc.Button('Run parameter search', id='run-uni-param-search'),
                ], width=4)
            ]),
            html.Br(),
            dcc.RadioItems(
                options=[
                    {'label': 'AIC', 'value': 'aic'},
                    {'label': 'AICc', 'value': 'aicc'},
                    {'label': 'BIC', 'value': 'bic'}
                ], id='uni-criterion',
                value='aic',
                inputStyle={'marginRight': '5px'},
                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
            ),
            dcc.Graph(id='parameter-heatmap', figure=get_empty_plot('Criteria heatmap will be displayed here')),
            dcc.Graph(id='uni-parameter-minimum-plot', figure=get_empty_plot('Minimum criteria plot will be displayed here'))
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='uni-param-search-close', className='ml-auto')
        )
    ], id='uni-param-search-modal', size='lg')
])

multi_param_search_modal = html.Div([
    dbc.Button('Parameter Search', id='multi-param-search', color='success', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Parameter Search (VAR)'),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    'Max p:',
                    dcc.Input(id='multi-limit-p', type='number', placeholder='Limit of p'),
                ], width=4),
                dbc.Col(width=4),
                dbc.Col([
                    html.Br(),
                    dbc.Button('Run parameter search', id='run-multi-param-search'),
                ], width=4)
            ]),
            html.Br(),
            dcc.Graph(id='multi-parameter-minimum-plot', figure=get_empty_plot('Minimum criteria plot will be displayed here'))
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='multi-param-search-close', className='ml-auto')
        )
    ], id='multi-param-search-modal', size='lg')
])

nn_preferences_modal = html.Div([
    dbc.Button('Preferences', id='nn-preferences', color='success', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Neural Network Preferences'),
        dbc.ModalBody([
            dcc.Tabs(
                id='nn-preferences-model',
                value='lstm',
                children=[
                    dcc.Tab(
                        label='LSTM',
                        value='lstm',
                        children=[
                            html.A('First Layer', style={'fontWeight': 'bold'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        'Number of nodes:',
                                        dcc.Input(id='lstm-first-layer-nodes', type='number', min=1, placeholder='Number of nodes', value=20, style={'width': '100%'}),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Activation Function:',
                                        dcc.Dropdown(id='lstm-first-layer-activation', options=activation_functions_options, value='relu', clearable=False),
                                    ])
                                ])
                            ]),
                            html.Br(),
                            html.A('Second Layer (Optional)', style={'fontWeight': 'bold'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        'Number of nodes:',
                                        dcc.Input(id='lstm-second-layer-nodes', type='number', min=1, placeholder='Number of nodes', value=20, style={'width': '100%'}),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Activation Function:',
                                        dcc.Dropdown(id='lstm-second-layer-activation', options=activation_functions_options, value='relu', clearable=False),
                                    ])
                                ])
                            ])
                        ]
                    ),
                    dcc.Tab(
                        label='CNN',
                        value='cnn',
                        children=[
                            'First Layer',
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        'Number of filters:',
                                        dcc.Input(id='cnn-first-layer-filters', type='number', min=1, placeholder='Number of filters', value=32),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Number of kernels:',
                                        dcc.Input(id='cnn-first-layer-kernel', type='number', min=1, placeholder='Kernel size', value=4),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Activation Function:',
                                        dcc.Dropdown(id='cnn-first-layer-activation', options=activation_functions_options, value='relu', clearable=False),
                                    ])
                                ])
                            ]),
                            html.Br(),
                            'Second Layer (Optional)',
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        'Number of filters:',
                                        dcc.Input(id='cnn-second-layer-filters', type='number', min=1, placeholder='Number of filters', value=16),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Number of kernels:',
                                        dcc.Input(id='cnn-second-layer-kernel', type='number', min=1, placeholder='Kernel size', value=2),
                                    ])
                                ]),
                                dbc.Col([
                                    html.Div([
                                        'Activation Function:',
                                        dcc.Dropdown(id='cnn-second-layer-activation', options=activation_functions_options, value='relu', clearable=False),
                                    ])
                                ])
                            ])
                        ]
                    )
                ]
            ),
            html.Br(),
            html.A('General', style={'fontWeight': 'bold'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        'Dropout (Optional):',
                        dcc.Input(id='dropout', type='number', min=0, max=1, value=0.2, placeholder='Dropout', style={'width': '100%'}),
                    ])
                ]),
                dbc.Col([
                    html.Div([
                        't parameter:',
                        dcc.Input(id='t-parameter', type='number', min=1 , value=5, style={'width': '100%'})
                    ])
                ])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        'Batch size:',
                        dcc.Input(id='batch-size', type='number', min=1, value=10, style={'width': '100%'})
                    ])
                ]),
                dbc.Col([
                    html.Div([
                        'Epochs:',
                        dcc.Input(id='nn-epochs', type='number', min=10, value=50, style={'width': '100%'})
                    ])
                ])
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='nn-preferences-close', className='ml-auto')
        )
    ], id='nn-preferences-modal', size='lg')
])

nn_results_modal = html.Div([
    dbc.Button('Results', id='nn-results', color='success', disabled=True, style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Neural Network Training Results'),
        dbc.ModalBody([
            'The following plot shows results for normalized input:',
            dcc.Graph(id='loss-results-plot', figure=get_empty_plot('No Neural Network results found')),
            html.Br(),
            dcc.Dropdown('training-set-var', clearable=False),
            dcc.Graph(id='training-set-fit', figure=get_empty_plot('No Neural Network results found')),
            html.Br(),
            dcc.Graph(id='grad-info', figure=get_empty_plot('No Neural Network results found')),
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='nn-results-close', className='ml-auto')
        )
    ], id='nn-results-modal', size='lg')
])
