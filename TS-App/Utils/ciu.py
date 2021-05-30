import matplotlib.pyplot as plt
import numpy as np
import pandas
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

def pcmci_setup(data):
	dataframe = pp.DataFrame(data.values, var_names=list(data.columns))
	parcorr = ParCorr(significance='analytic')
	pcmci = PCMCI(
		dataframe=dataframe,
		cond_ind_test=parcorr,
		verbosity=1)
	return pcmci

def run_pcmci(data, plot_res=False):
	var_names = list(data.columns)
	pcmci = pcmci_setup(data)
	correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
	results = pcmci.run_pcmci(tau_max=8, pc_alpha=None)
	q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
	link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix, val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
	if plot_res:
		tp.plot_graph(val_matrix=results['val_matrix'], link_matrix=link_matrix, var_names=var_names, link_colorbar_label='cross-MCI', node_colorbar_label='auto-MCI')
		plt.show()
	return link_matrix

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
