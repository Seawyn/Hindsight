from Structure.helperfunctions import *

activation_functions_options = [{'label': 'Sigmoid', 'value': 'sigmoid'}, {'label': 'Tanh', 'value': 'tanh'},
                                {'label': 'ReLU', 'value': 'relu'}, {'label': 'ELU', 'value': 'elu'},
                                {'label': 'Swish', 'value': 'swish'}]

forecast_strategy_options = [{'label': 'Recursive', 'value': 'recursive'}, {'label': 'Direct', 'value': 'direct'},
                            {'label': 'MIMO', 'value': 'mimo'}, {'label': 'DIRMO/MISMO', 'value': 'dirmo'}]

autocorrelation_modal = html.Div([
    dbc.Button('Autocorrelation Info', id='autocorrelation', color='info', style={'width': '250px'}),
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
    dbc.Button('Seasonal Decomposition', id='seasonal-decomposition', color='info', style={'width': '250px'}),
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
    dbc.Button('Imputation', id='imputation', color='info', style={'width': '250px'}),
    dbc.Modal([
        dbc.ModalHeader('Imputation'),
        dbc.ModalBody([
            dcc.Dropdown(id='imputation-variables', style={'marginBottom': '25px'}),
            dcc.Tabs(
                id='imputation-tabs',
                value='spline-interpolation-tab',
                children=[
                    # Spline interpolation tab
                    dcc.Tab(
                        label='Spline Interpolation',
                        value='spline-interpolation-tab',
                        children=[
                            dcc.Input(id='spline-order', type='number', placeholder='Spline Order'),
                            dbc.Button('Interpolate', id='spline-interpolation', color='info')
                        ]
                    ),
                    dcc.Tab(
                        label='Local Average',
                        value='local-average-tab',
                        children=[
                            dcc.Input(id='window-size', type='number', placeholder='Window Size'),
                            dbc.Button('Interpolate', id='local-average-interpolation', color='info')
                        ]
                    ),
                    dcc.Tab(
                        label='MLP',
                        value='mlp-tab',
                        children=[
                            dcc.Input(id='mlp-imputation-ar-lag', type='number', placeholder='Sliding Window size'),
                            dbc.Button('Interpolate', id='mlp-imputation', color='info')
                        ]
                    )
                ]
            ),
            dcc.Graph(id='imputation-results', figure=get_empty_plot('Imputation results will be displayed here'))
        ]),
        dbc.ModalFooter(children=[
            dbc.Button('Cancel', id='imputation-cancel', className='ml-auto'),
            dbc.Button('Confirm', id='imputation-confirm', className='ml-auto')
        ])
    ], id='imputation-modal', size='lg'),
])

uni_param_search_modal = html.Div([
    dbc.Button('Parameter Search', id='uni-param-search', color='info', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Parameter Search (SARIMA)'),
        dbc.ModalBody([
            dcc.Input(id='uni-limit_p', type='number', placeholder='Limit of p'),
            dcc.Input(id='uni-limit_q', type='number', placeholder='Limit of q'),
            dcc.Input(id='uni-itr', type='number', placeholder='Integration'),
            dcc.Input(id='uni-seasonality', type='number', placeholder='Seasonality'),
            dbc.Button('Run parameter search', id='run-uni-param-search', color='info'),
            dcc.RadioItems(
                options=[
                    {'label': 'AIC', 'value': 'aic'},
                    {'label': 'AICc', 'value': 'aicc'},
                    {'label': 'BIC', 'value': 'bic'}
                ], id='uni-criterion', value='aic', labelStyle={'display': 'inline-block'}
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
    dbc.Button('Parameter Search', id='multi-param-search', color='info', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Parameter Search (VAR)'),
        dbc.ModalBody([
            dcc.Input(id='multi-limit-p', type='number', placeholder='Limit of p'),
            dbc.Button('Run parameter search', id='run-multi-param-search', color='info'),
            dcc.Graph(id='multi-parameter-minimum-plot', figure=get_empty_plot('Minimum criteria plot will be displayed here'))
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='multi-param-search-close', className='ml-auto')
        )
    ], id='multi-param-search-modal', size='lg')
])

nn_preferences_modal = html.Div([
    dbc.Button('Preferences', id='nn-preferences', color='info', style={'width': '100%'}),
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
                            'First Layer',
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
                            'Second Layer (Optional)',
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
            'General',
            dbc.Row([
                dbc.Col([
                    html.Div([
                        'Dropout (Optional):',
                        dcc.Input(id='dropout', type='number', min=0, max=1, placeholder='Dropout', style={'width': '100%'}),
                    ])
                ]),
                dbc.Col([
                    html.Div([
                        'Forecast Strategy:',
                        dcc.Dropdown(id='forecast-strategy', options=forecast_strategy_options, value='recursive', clearable=False)
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
    dbc.Button('Results', id='nn-results', color='info', disabled=True, style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Neural Network Training Results'),
        dbc.ModalBody([
            'The following plot shows results for normalized input:',
            dcc.Graph(id='loss-results-plot', figure=get_empty_plot('No Neural Network results found')),
            dcc.Dropdown('training-set-var', clearable=False),
            dcc.Graph(id='training-set-fit', figure=get_empty_plot('No Neural Network results found')),
        ]),
        dbc.ModalFooter(
            dbc.Button('Close', id='nn-results-close', className='ml-auto')
        )
    ], id='nn-results-modal', size='lg')
])
