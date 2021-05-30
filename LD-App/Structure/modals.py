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
                        dcc.Dropdown(id='ld-imputation-subjects')
                    ], width=8),
                    dbc.Col(children=[
                        dcc.Checklist(
                            options=[{'label': 'Apply to all subjects', 'value': 'apply'}],
                            style={'marginTop': '10px'}
                        )
                    ], width=4),
                ])
            ]),
            html.Br(),
            dcc.Tabs(
                id='ld-imputation-tabs',
                value='simple',
                children=[
                    dcc.Tab(
                        label='Simple methods',
                        value='simple',
                        children=[
                            html.Br(),
                            'Method:',
                            dcc.Dropdown(
                                id='bias-imputation',
                                options=[{'label': 'LOCF', 'value': 'locf'}],
                                value='locf',
                                clearable=False
                            ),
                            html.Br(),
                            'Variables:',
                            dcc.Dropdown(
                                id='simple-imputation-variables',
                            )
                        ]
                    ),
                    dcc.Tab(
                        label='MissForest',
                        value='missforest',
                        children=[
                            html.Br(),
                            'Input variables:',
                            dcc.Dropdown(id='moss-forest-variables'),
                        ]
                    )
                ]
            )
        ]),
        dbc.ModalFooter(children=[
            dbc.Button('Confirm', id='ld-imputation-confirm', className='ml-auto'),
            dbc.Button('Cancel', id='ld-imputation-cancel', className='ml-auto')
        ])
    ], id='ld-imputation-modal', size='lg'),
    dbc.Tooltip('Contains imputation methods for missing data in longitudinal data', target='ld-imputation'),
])
