import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas
from statsmodels.tsa.stattools import grangercausalitytests
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

# Applies the Granger Causality test between two variables
# Specify runs Granger Causality test for each time lag up to maxlag
def granger_causality_test(data, maxlag, verbose, specify=False):
	test_results = grangercausalitytests(data, maxlag=maxlag, verbose=False)
	p_values = []
	for lag in range(1, maxlag + 1):
		p_values.append(test_results[lag][0]['ssr_chi2test'][1])
	# Returns all p-values (causality per lag)
	if specify:
		return p_values
	# Returns smallest p-value (if there is any causality)
	else:
		return min(p_values)

# Calculates and outputs a matrix consisting of a Granger Causality test
# for each pair of variables, where values below 0.05 indicate that the
# variable of the column can be used to predict the variable of the row
def granger_causality_matrix(data, maxlag=10, plot=False):
    cols = len(data.columns)
    matrix = np.zeros([cols, cols])
    for i, row in enumerate(data.columns):
        for j, col in enumerate(data.columns):
            matrix[i, j] = granger_causality_test(data[[row, col]], maxlag, False)
    if plot:
        plt.imshow(matrix)
        plt.colorbar()
        plt.show()
    return matrix

# Receives a causality matrix as input and outputs a matrix where
# values below 0.05 are replaced by 1 and 0 otherwise
def filter_causality_matrix(matrix, plot=False):
    filtered_matrix = np.zeros(matrix.shape)
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        if matrix[i, j] < 0.05:
            filtered_matrix[i, j] = 1
    if plot:
        plt.imshow(filtered_matrix, cmap='Greens')
        plt.colorbar()
        plt.show()
    return filtered_matrix

# Given a dataset and a variable, outputs a dataframe where each column
# stores the Granger Causality test result for each variable
# Rows are sorted by values and there are no values equal or above 0.05
def granger_causality_by_variable(data, var, maxlag=10, maxval=0.05):
    if maxval > 0.05:
        raise ValueError('maxval above 0.05')
    vals = pandas.DataFrame([], index=[var])
    for col in data.columns:
        val = granger_causality_test(data[[var, col]], maxlag, False)
        if val < maxval:
            vals[col] = val
    vals = vals.sort_values(by=var, axis=1)
    return vals

# Receives dataframe output of granger_causality_by_variable and creates a network matrix
def causality_by_variable_to_matrix(data):
	col_size = len(data.columns) + 1
	mt = np.zeros((col_size, col_size))
	# Set all connections of each column to row as 1
	# Remaining connections are 0
	mt[0, 1:] = np.zeros(len(data.columns)) + 1
	return mt, [data.index[0]] + list(data.columns)

# Sets up PCMCI
def pcmci_setup(data):
	dataframe = pp.DataFrame(data.values, var_names=list(data.columns))
	parcorr = ParCorr(significance='analytic')
	pcmci = PCMCI(
		dataframe=dataframe,
		cond_ind_test=parcorr,
		verbosity=1)
	return pcmci

# Runs Tigramite's implementation of the PCMCI Algorithm
def run_pcmci(data, tau_max=8, plot_res=False):
	var_names = list(data.columns)
	pcmci = pcmci_setup(data)
	# correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
	results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)
	q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
	link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix, val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
	if plot_res:
		tp.plot_graph(val_matrix=results['val_matrix'], link_matrix=link_matrix, var_names=var_names, link_colorbar_label='cross-MCI', node_colorbar_label='auto-MCI')
		plt.show()
	return link_matrix

# Receives link matrix from PCMCI and returns a causality network
def parse_link_matrix(link_matrix, keys):
	n = len(keys)
	causal_net = np.zeros((n, n))
	# For each variable entry in the link_matrix
	for i in range(n):
		# For each variable relatioship (row)
		for j in range(len(link_matrix[i])):
			if True in link_matrix[i][j]:
				causal_net[j, i] = 1
	return causal_net
