import os
import pandas
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Converts Time Series DataFrame to Longitudinal Data and saves to csv file
def convert_to_tdbn_format(df):
    pd = pandas.DataFrame()
    for i in range(len(df)):
        for j in range(len(df.columns)):
            pd['X' + str(j) + '__' + str(i)] = [df[df.columns[j]][i]]
    pd.to_csv('temp.csv', sep=',')

# Calls tDBM Algorithm through the Shell
def execute_tdbn(filename, markov_lag=1, num_parents=1, use_mdl=True):
	command = 'java -jar tDBN-0.1.4.jar -i ' + filename
	command += ' -m ' + str(markov_lag)
	command += ' -p ' + str(num_parents)
	if not use_mdl:
		command += ' -s ll' 
	command += ' -o TempDBNResult.txt'
	os.system(command)

# Create networks from output
def parse_output(filename):
	with open(filename, 'r') as f:
		# Get lines (including empty lines)
		data = f.read().split('\n')

	networks = {'n1': []}
	print("Created Network 1")
	current_net_id = 1
	last_line = None
	for i in range(len(data)):
		if data[i] != '':
			networks['n' + str(current_net_id)].append(data[i])
		else:
			if last_line != '':
				if data[i + 1] != '':
					current_net_id += 1
					networks['n' + str(current_net_id)] = []
					print("Created Network", str(current_net_id))
				# Reached end of file
				else:
					break
		last_line = data[i]
	return networks

# Returns parent and node from line
def parse_line(line):
	elements = line.split(' -> ')
	return elements[0], elements[1]

# Creates networks from dictionary
def networks_from_output(networks):
	graphs = {}
	for key in networks.keys():
		current_net = networks[key]
		graph = nx.DiGraph()
		for line in current_net:
			orig, dest = parse_line(line)
			graph.add_edge(orig, dest)
		graphs[key] = graph
	return graphs

# Plots each graph in graphs
def plot_graphs(graphs):
	for key in graphs.keys():
		pos = nx.spring_layout(graphs[key])
		nx.draw(graphs[key], pos, node_size=750, font_size=8, with_labels=True, node_color='#aaaaaa')
		plt.show()

# If output exists, parse output, create networks and deletes output
def check_output():
	if not os.path.exists('TempDBNResult.txt'):
		raise ValueError("TempDBNResult.txt not found! Maybe DBN was not called?")
	else:
		networks = parse_output('TempDBNResult.txt')
		graphs = networks_from_output(networks)
		plot_graphs(graphs)
		os.remove('TempDBNResult.txt')

# Discretizes the data using Piecewise Aggregate Approximation
def paa(df, size):
    new_df = pandas.DataFrame()
    for col in df.columns:
        new_array = []
        for i in range(0, len(df), size):
            new_array.append(np.average(df[col].values[i:i + size]))
        new_df[col] = new_array
    return new_df
