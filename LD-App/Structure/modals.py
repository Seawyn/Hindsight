from Structure.helperfunctions import *

dataframe_modal = html.Div(children=[
    dbc.Button('Data Table Options', id='data-table-options', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Data Table Options'),
        dbc.ModalBody([
            'Selected Columns:',
            dcc.Dropdown(id='current-columns', multi=True, clearable=False),
            html.Br(),
            'Selected Subjects:',
            dcc.Dropdown(id='current-subject', options=[{'label': 'All', 'value': 'all'}], value='all', clearable=False),
            html.Br(),
            html.Hr(),
            html.A('Find and replace', style={'fontWeight': 'bold'}),
            html.Br(),
            'Variable:',
            dcc.Dropdown(id='find-replace-variable'),
            html.Br(),
            dbc.Row(children=[
                dbc.Col(children=[
                    'Value:',
                    dcc.Dropdown(id='variable-value-to-replace')
                ]),
                dbc.Col(children=[
                    'Replacement:',
                    dbc.Input(id='variable-value-replacement', type='number')
                ])
            ]),
            html.Br(),
            dbc.Row(children=[
                dbc.Col(),
                dbc.Col(),
                dbc.Col(dbc.Button('Replace', id='replace-confirm', disabled=True, style={'width': '100%'}))
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button('Confirm', id='data-table-options-close', className='ml-auto')
        ),
    ], id='data-table-modal', size='lg'),
    dbc.Tooltip('Contains variable and subject options for changing the data table', target='data-table-options'),
])

ld_imputation_modal = html.Div(children=[
    dbc.Button('Imputation', id='ld-imputation', style={'width': '100%'}),
    dbc.Modal([
        dbc.ModalHeader('Variable Imputation'),
        dbc.ModalBody([
            html.Div(children=[
                'Subjects:',
                dbc.Row(children=[
                    dbc.Col(children=[
                        dcc.Dropdown(id='ld-imputation-subjects', multi=True)
                    ], width=8),
                    dbc.Col(children=[
                        dcc.Checklist(
                            options=[{'label': 'Apply to all subjects', 'value': 'apply'}],
                            style={'marginTop': '10px'},
                            id='all-subjects'
                        )
                    ], width=4),
                ]),
                html.Br(),
                'Variables:',
                dcc.Dropdown(
                    id='ld-imputation-variables',
                    multi=True
                ),
                html.Br(),
                'Method:',
                dcc.Dropdown(
                    id='imputation-method',
                    options=[{'label': 'LOCF', 'value': 'locf'}, {'label': 'MissForest', 'value': 'missforest'}],
                    value='locf',
                    clearable=False
                ),
                html.Div(children=[
                    html.Br(),
                    'Discrete variables:',
                    dcc.Dropdown(id='missforest-discrete-variables', multi=True),
                ], id='discrete-variables-div', style={'display': 'none'})
            ]),
        ]),
        dbc.ModalFooter(children=[
            dbc.Button('Confirm', id='ld-imputation-confirm', className='ml-auto'),
            dbc.Button('Cancel', id='ld-imputation-cancel', className='ml-auto')
        ])
    ], id='ld-imputation-modal', size='lg'),
    dbc.Tooltip('Contains imputation methods for missing data in longitudinal data', target='ld-imputation'),
])
