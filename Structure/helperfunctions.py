import os
import sys
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import json
import plotly.express as px
import plotly.graph_objs as go
from Utils.tsu import *
from Utils.mtsu import *
from Utils.mdu import *
from Utils.nnu import *

# Checks if file exists and if is a valid .csv file
def check_file(filename):
    file = filename.split('/')[-1]
    if len(file.split('.')) != 2:
        return False
    (name, type) = file.split('.')
    is_csv = (type == 'csv') and (len(name) > 0)
    return os.path.exists(filename) and is_csv

# Update variable selection dropdown with variable names
# Plots 5 first variables by default
# TODO: Change to callback
def populate_var_select(df):
    options = []
    cols = list(df.columns)
    for col in cols:
        options.append({'label': col, 'value': col})
    return options, cols[:5]

def create_acf_plot(data, plot_name, conf_int=None, nobs=None):
    if plot_name == 'acf':
        title = 'Autocorrelation Function'
        gap = 0.5
    elif plot_name == 'res-acf':
        title = 'Residual Autocorrelation Function'
        if not nobs is None:
            gap = 2 / math.sqrt(nobs)
        # This shouldn't happen
        else:
            gap = 0.1
    else:
        title = 'Partial Autocorrelation Function'
        gap = 0.2

    x = list(np.arange(len(data)))
    y = list(data)
    if conf_int is None:
        center_conf = np.zeros(len(data))
        upper_conf = list(center_conf + gap)
        lower_conf = list(center_conf - gap)
    # Uses Statsmodels' implementation of Bartlett's Formula
    else:
        upper_conf = list(conf_int[:, 1] - data)
        lower_conf = list(conf_int[:, 0] - data)

    fig = go.Figure([
        # Plot of data
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            showlegend=False
        ),
        # Alpha interval (at 0.05)
        go.Scatter(
            x=x+x[::-1],
            y=upper_conf+lower_conf[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            hoverinfo='skip',
            line=dict(width=0),
            showlegend=False
        )
    ])

    fig.update_layout(
        width=750,
        height=350,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=25,
            pad=4
        ),
        title=title,
        yaxis_range=[-1.2, 1.2]
    )

    return fig

def setup(filepath):
    df = read_dataset(filepath, header=0)
    options, values = populate_var_select(df)
    fig = px.line(df, y=values)
    if len(get_missing_columns(df)) == 0:
        # Plots Autocorrelation and Partial Autocorrelation Functions of the first variable
        acf_data, pacf_data = get_acf_and_pacf(df[values[0]])
        fig_acf = create_acf_plot(acf_data, 'acf')
        fig_pacf = create_acf_plot(pacf_data, 'pacf')
    else:
        # Returns empty plots
        fig_acf = get_empty_plot('There are missing values')
        fig_pacf = get_empty_plot('There are missing values')
    return fig, fig_acf, fig_pacf, df, options, values

def setup_seasonal_decomposition(df, var):
    if len(get_missing_columns(df)) == 0:
        # Plots Seasonal Decomposition
        res = get_seasonal_decomposition(df[var])
        fig_trend = px.line(res.trend)
        fig_seasonal = px.line(res.seasonal)
        fig_residual = px.line(res.resid)
    else:
        # Returns empty plots
        fig_trend = get_empty_plot('There are missing values')
        fig_seasonal = get_empty_plot('There are missing values')
        fig_residual = get_empty_plot('There are missing values')
    return fig_trend, fig_seasonal, fig_residual

def create_forecasting_plot(df, var, split, predicted, forecast=None, conf_int=None):
    data = df[var]
    x_pred = np.arange(split, len(data))

    fig = go.Figure([
        # Observed time series
        go.Scatter(
            x = np.arange(len(data)),
            y = data.values,
            mode='lines',
            name='observed'
        ),
        # Predicted time series
        go.Scatter(
            x = x_pred,
            y = predicted,
            mode='lines',
            name='predicted'
        )
    ])

    # If there is a forecast window
    if not (forecast is None) and not (conf_int is None):
        forecast = pd.read_json(forecast).sort_index()[var].values
        conf_int = pd.read_json(conf_int).sort_index()[[var + '_lower_conf', var + '_upper_conf']].values.transpose()

        # conf_int = [lower_conf, upper_conf]
        x_for = list(np.arange(len(data), len(data) + len(predicted)))
        conf_int = conf_int.tolist()
        forecast_plot = go.Scatter(
            x = x_for,
            y = forecast,
            mode='lines',
            name='forecast'
        )
        upper_conf_plot = go.Scatter(
            x = x_for,
            y = conf_int[1],
            mode='lines',
            hoverinfo='skip',
            line=dict(width=0),
            showlegend=False
        )
        lower_conf_plot = go.Scatter(
            x = x_for,
            y = conf_int[0],
            mode='lines',
            fillcolor='rgba(0, 100, 80, 0.2)',
            fill='tonexty',
            hoverinfo='skip',
            line=dict(width=0),
            showlegend=False
        )
        fig.add_trace(forecast_plot)
        fig.add_trace(upper_conf_plot)
        fig.add_trace(lower_conf_plot)

    return fig

def process_predicted(pred, vars):
    pred_data = pd.DataFrame(pred).transpose()
    pred_data.columns = vars
    return pred_data

def process_forecast_results(forecast_results, vars):
    lower_vars = []
    upper_vars = []
    for var in vars:
        lower_vars.append(var + '_lower_conf')
        upper_vars.append(var + '_upper_conf')
    forecast = pd.DataFrame(forecast_results[0])
    forecast.columns = vars
    lower_conf = pd.DataFrame(forecast_results[1])
    lower_conf.columns = lower_vars
    upper_conf = pd.DataFrame(forecast_results[2])
    upper_conf.columns = upper_vars
    return forecast, lower_conf.join(upper_conf)

def get_conf_int(data, var, from_json=True):
    if from_json:
        conf_int_data = pd.read_json(data).sort_index()
    else:
        conf_int_data = data
    vars = [var + '_lower_conf', var + '_upper_conf']
    return conf_int_data[vars].transpose().values

def get_new_dropdown_options(available_vars):
	content = []
	for val in available_vars:
	    new_el = {}
	    new_el['label'] = val
	    new_el['value'] = val
	    content.append(new_el)
	return content

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
    return fig

def create_causality_elements(matrix, variables):
    elements = []

    if len(variables) == 1:
        return dcc.Graph(figure=get_empty_plot('Dataset is univariate', width=600))

    # Add variables as nodes
    for var in variables:
        elements.append({'data': {'id': var, 'label': var}})

    # Add causality relations as edges
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                elements.append({'data': {'source': variables[j], 'target': variables[i]}})

    return elements

# Trains a SARIMAX model and outputs a dictionary with all data (in json format)
def sarimax_process(df, params, split, var, forecast_window):
    model_fit, pred, error, residuals = predict_sarimax(df, params, split, var, status=False)
    pred_data = pd.DataFrame(pred)
    pred_data.columns = [var]
    pred_data = pred_data.to_json()

    if not forecast_window is None:
        # Create, process and store predictions and confidence intervals as json data
        forecast, conf_int = forecast_sarimax(model_fit, df, forecast_window)
        forecast_data = pd.DataFrame(forecast)
        forecast_data.columns = [var]
        forecast_data = forecast_data.to_json()
        conf_int_data = pd.DataFrame(conf_int).transpose()
        conf_int_data.columns = [var + '_lower_conf', var + '_upper_conf']
        conf_int_data = conf_int_data.to_json()
    else:
        # If there is no Forecasting window, related data is empty
        forecast_data = None
        conf_int_data = None

    nobs = model_fit.nobs

    # Create dictionary and convert to json
    last_sarimax = get_training_dict(df.to_json(), split, var, pred_data, forecast_data, conf_int_data, error, residuals.tolist(), nobs)

    return last_sarimax

# Trains a VAR model and outputs a dictionary with all data (in json format)
def var_process(df, p, split, vars, forecast_window):
    model_fit, pred, error, residuals = predict_var(df, p, split)
    pred_data = process_predicted(pred, vars).to_json()

    if not forecast_window is None:
        # Create, process and store predictions and confidence intervals as json data
        forecast_results = forecast_var(model_fit, df, forecast_window)
        forecast_data, conf_int_data = process_forecast_results(forecast_results, vars)
        forecast_data = forecast_data.to_json()
        conf_int_data = conf_int_data.to_json()
    else:
        # If there is no Forecasting window, related data is replaced by empty dataframes
        forecast_data = None
        conf_int_data = None

    nobs = model_fit.nobs

    # Create dictionary and convert to json
    last_var = get_training_dict(df.to_json(), split, vars, pred_data, forecast_data, conf_int_data, error, residuals.tolist(), nobs)

    return last_var

# Trains a Neural Network model and outputs a dictionary with all data (in json format)
def nn_process(df, model, window_size, split, available_vars, forecast_window, hyperparam_data, seed=None):
    res, seed, errors, residuals, history, pr_train, y_train = neural_network_regression(model, df, window_size, split, hyperparam_data, output_size=len(available_vars))
    pred_data = res.to_json()

    # Still have to implement forecast data
    forecast_data = None
    conf_int_data = None

    # Rough estimate
    nobs = len(df)

    last_nn = get_training_dict(df.to_json(), split, available_vars, pred_data, forecast_data, conf_int_data, errors, residuals.tolist(), nobs, seed=seed)
    nn_res = {'loss': history.history['loss'], 'val_loss': history.history['val_loss'], 'training': y_train, 'training_res': pr_train}

    return last_nn, nn_res

# Creates a dictionary with all training results and converts to json format
def get_training_dict(df, split, var, pred_data, forecast_data, conf_int_data, error, residuals, nobs, seed=None):
    tr_dict = {}
    tr_dict['data'] = df
    tr_dict['split'] = split
    tr_dict['var'] = var
    tr_dict['predicted'] = pred_data
    tr_dict['forecast'] = forecast_data
    tr_dict['conf_int'] = conf_int_data
    tr_dict['error'] = error
    tr_dict['residuals'] = residuals
    tr_dict['nobs'] = nobs
    tr_dict['seed'] = seed
    tr_dict = json.dumps(tr_dict)
    return tr_dict

# For a given model, update results based on obtained residuals
def update_model_results(model, var, performances, error, residuals, last_test, nobs):
    # Check if results should be reset (if test size changed)
    if len(residuals) != last_test:
        results = {}
        last_test = len(residuals)
    else:
        results = json.loads(performances)

    # If model not been trained before
    if model not in results.keys():
        results[model] = {}
        results[model]['error'] = {}
        res_df = pd.DataFrame()

    # If model has been trained before
    else:
        res_df = pd.read_json(results[model]['residuals']).sort_index()

    residuals = pd.DataFrame(residuals, columns=var)
    for variable in var:
        results[model]['error'][variable] = error[variable]
        res_df[variable] = residuals[variable]

    results[model]['residuals'] = res_df.to_json()
    results[model]['nobs'] = nobs
    results = json.dumps(results)
    return results, last_test

# Interprets Ljung-Box test p-values and outputs percentage of lags below alpha
def interpret_ljung_box(residuals):
    p_values = acorr_ljungbox(residuals, return_df=False)[1]
    lags_under_alpha = 0

    for i in range(len(p_values)):
        if p_values[i] < 0.05:
            lags_under_alpha += 1

    percentage_under_alpha = lags_under_alpha // len(p_values)
    return percentage_under_alpha

# Returns bar plot with performances of each model color coded by Ljung-Box test result
def get_performance_bar_plot(perf_data):
    x_labels = []
    errors = []
    percentages_under_alpha = []
    all_vars = []

    # Loop through all trained models
    for key in perf_data.keys():
        residuals = pd.read_json(perf_data[key]['residuals']).sort_index()
        # If time series is multivariate, append all errors and run Ljung-Box test
        model_errors = perf_data[key]['error']
        for var in model_errors.keys():
            x_labels.append(key + ' - ' + var)
            errors.append(model_errors[var])
            percentages_under_alpha.append(interpret_ljung_box(residuals[var]))

    # Create bar plot
    fig = go.Figure(data=[go.Bar(
        x = x_labels,
        y = errors,
        marker=dict(
            cmin=0,
            cmax=1,
            color=percentages_under_alpha,
            colorscale="RdYlGn",
            reversescale=True,
            colorbar=dict(
                title='Lags under alpha (%)',
            )
        )
    )])
    fig.update_layout(
        title='Model Performances',
        yaxis=dict(
            title='Mean Absolute Error (MAE)'
        ),
        xaxis_tickangle=20
    )
    return fig
