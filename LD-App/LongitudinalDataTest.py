import dash
from dash.dash import no_update
import dash_table
from Structure.modals import *

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    dbc.Card([
        # Import Dataset (only upon startup)
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
                            html.Div('No dataset has been selected', id='current-filename')
                        ]),
                        dbc.CardFooter(dbc.Button('Confirm', id='upload-confirm', disabled=True, style={'backgroundColor': '#58B088', 'border': 'none'})),
                    ])
                ),
                dbc.Col()
            ])
        ], id='upload-screen'),
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
                            html.H5('Things and stuff', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            'Testing new row'
                        ])
                    ])
                ),
                dbc.Col(
                    dbc.Card(children=[
                        dbc.CardHeader(
                            html.H5('Things and stuff: the long awaited sequel', style={'color': '#FFFFFF'}),
                            style={'backgroundColor': '#333333'}
                        ),
                        dbc.CardBody(children=[
                            'Now featuring chapters!'
                        ])
                    ])
                )
            ])
        ], id='main-screen', style={'display': 'none'}),
    ], color='#ACF2D3', style={'height': '100vh'}),

    # Hidden div holds the original dataset
    html.Div(id='dataset', style={'display': 'none'}),

    # Hidden div holds the current instance of the dataset
    html.Div(id='current-data', style={'display': 'none'}),

    # TODO: Data imputation
    # TODO: Restricted Boltzmann Machine / Dynamic Boltzmann Machine
    # TODO: Dynamic Bayesian Network
])

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
            print('Please upload a valid .csv file')
            return True, 'Upload is not a valid .csv file', None
    return True, no_update, no_update

# Upload Confirm button closes upload Card and displays main page
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

# Uploading dataset alters the current dataset instance
@app.callback(
    dash.dependencies.Output('current-data', 'children'),
    [dash.dependencies.Input('dataset', 'children')]
)

def update_current_data(data):
    ctx = dash.callback_context
    if ctx.triggered:
        # df = pandas.read_json(data).sort_index()
        return data
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
            if current_trigger == 'data-table-options-close':
                df = pandas.read_json(data).sort_index()
                # Display a subset of the columns to prevent heavy load
                current_df = df[cur_cols]
                if cur_ind != 'all':
                    current_df = get_subject(current_df, cur_ind, df.columns[0])

                degrees_missingness = column_missingness(current_df)
                tooltip_header = get_column_info(current_df)

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

if __name__ == '__main__':
    app.run_server(debug=True)
