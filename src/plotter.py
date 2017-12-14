#!/usr/bin/env python3
# A script for plotting the results obtained in the various experiments.
# Check the main function to define which files it will look for results in.

# Copyright (C) 2017 University of Coimbra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Original author and maintainer: Gonçalo S. Martins (gondsm@gmail.com)

import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import rc
import matplotlib.patches as mpatches
import itertools
import numpy as np
import yaml
from yaml import CLoader as Loader, CDumper as Dumper
import pickle
import time
import sys
from pprint import pprint
import random


def normalize(lst):
	""" Normalizes a list of numbers """
	return [float(i)/sum(lst) for i in lst]


def calc_reward(states, v_s, transition_dist):
	""" Calculate the SVR reward for illustrative purposes """
	reward = 0.0
	for i,state in enumerate(states):
		reward += transition_dist[i]*v_s[state]
	return reward


def plot_distribution_example():
	""" Generate a few figures that exemplify the usage of SVR and our
	learning mechanism.
	""" 
	# Define list of possible states
	it = list(itertools.product([1,2], [1,2]))
	dist = [1]*len(it)
	uniform_dist = [1]*len(it)

	# Value function
	v_s = {(1,1): 10, (1,2): 5, (2,1): 2, (2,2): 20}

	# Define labels and stuff
	x_label = "$S'$"
	x_label_rewards = "Action $a$"
	y_label = "$P(S'|S=(1,1),A=1)$"
	y_label_rewards = "$R(a, S = (1,1))$"
	fig_size_dist = (6,3)
	fig_size_reward = (6,4)
	reward_color = "green"
	prob_color = "cyan"

	# Define figure parameters
	y_range = (0, 0.5)
	y_range_rewards = (0, 75)
	col_width = 0.7
	left_padding = (1-col_width)/2
	xtick_padding = 0.5

	# Before new information
	# Plot probability
	plt.figure(figsize=fig_size_dist)
	plt.ylim(y_range)
	plt.bar([val + left_padding for val in range(0, len(dist))], normalize(dist), width=col_width, color=prob_color)
	plt.xlabel(x_label)
	plt.ylabel(y_label)	
	plt.xticks([val + xtick_padding for val in range(0, len(dist))], it)
	plt.tight_layout()
	plt.savefig("0_prob.pdf")

	# Plot reward
	actions = range(1,3)
	reward_vec = [calc_reward(it, v_s, uniform_dist) for a in actions]
	plt.figure(figsize=fig_size_reward)
	plt.ylim(y_range_rewards)
	plt.bar([val + left_padding for val in range(len(reward_vec))], reward_vec, width=col_width, color=reward_color)
	plt.xlabel(x_label_rewards)
	plt.ylabel(y_label_rewards)	
	plt.xticks([val + xtick_padding for val in range(0, len(reward_vec))], actions)
	plt.tight_layout()
	plt.savefig("0_action.pdf")
	
	# Gain information
	dist[3]+=1

	# After new information
	# Plot probability
	plt.figure(figsize=fig_size_dist)
	plt.ylim(y_range)
	plt.bar([val + left_padding for val in range(0, len(dist))], normalize(dist), width=col_width, color=prob_color)
	plt.xlabel(x_label)
	plt.ylabel(y_label)	
	plt.xticks([val + xtick_padding for val in range(0, len(dist))], it)
	plt.tight_layout()
	plt.savefig("1_prob.pdf")

	# Plot reward
	actions = range(1,3)
	reward_vec = [calc_reward(it, v_s, uniform_dist) for a in actions]
	reward_vec[0] = calc_reward(it, v_s, dist)
	plt.figure(figsize=fig_size_reward)
	plt.ylim(y_range_rewards)
	plt.bar([val + left_padding for val in range(len(reward_vec))], reward_vec, width=col_width, color=reward_color)
	plt.xlabel(x_label_rewards)
	plt.ylabel(y_label_rewards)	
	plt.xticks([val + xtick_padding for val in range(0, len(reward_vec))], actions)
	plt.tight_layout()
	plt.savefig("1_action.pdf")


def plot_timeseries_data(filename, outfile=None, entropy=False):
	""" Plot some nice timeseries obtained from the simulations. """
	# Read raw data
	# raw_data is a vector of vectors, each vector contains a full run of the algorithm
	raw_data = []
	extension = filename.split(".")[1]
	if extension == "txt":
		with open(filename) as data_file:
			for line in data_file:
				if line[0] != '#':
					raw_data.append([float(elem) for elem in line.split()])
	elif extension == "pkl":
		yaml_data = pickle.load(open(filename, "rb"))
		if entropy:
			raw_data = [d["entropies"] for d in yaml_data]
		else:
			raw_data = [d["rewards"] for d in yaml_data]
	else:
		print("Error: I don't know this file extension!")
		return

	# Split for iteration
	# Data will be a vector of vectors, with data [i] corresponding to all points at iteration i.
	data = raw_data
	#for i in range(len(raw_data[0])):
	#	data.append([elem[i] for elem in raw_data])

	# Calculate cumulative reward
	cum_reward = []
	if entropy:
		cum_reward = raw_data
	else:
		for trial in data:
			l_cum_reward = []
			for i in range(len(trial)):
				l_cum_reward.append(sum(trial[0:i]))
			cum_reward.append(l_cum_reward)

	# Calculate min, max and average
	avg_cum_reward = []
	std_cum_reward = []
	max_cum_reward = []
	min_cum_reward = []
	for i in range(len(cum_reward[0])):
		vec = [elem[i] for elem in cum_reward]
		avg_cum_reward.append(np.mean(vec))
		std_cum_reward.append(np.std(vec))
		max_cum_reward.append(max(vec))
		min_cum_reward.append(min(vec))

	# Calc + and - std vectors
	avg_minus_3std = [avg_cum_reward[i] - 3* std_cum_reward[i] for i in range(len(avg_cum_reward))]
	avg_plus_3std = [avg_cum_reward[i] + 3* std_cum_reward[i] for i in range(len(avg_cum_reward))]
	avg_minus_2std = [avg_cum_reward[i] - 2* std_cum_reward[i] for i in range(len(avg_cum_reward))]
	avg_plus_2std = [avg_cum_reward[i] + 2* std_cum_reward[i] for i in range(len(avg_cum_reward))]

	# Decide which extra plots will be used (yay for "everything is a reference"))
	extra_border_max = avg_plus_2std
	extra_border_min = avg_minus_2std

	# Decide colors
	if entropy:
		std_color = "orange"
		line_color = "red"
	else:
		std_color = "cyan"
		line_color = "blue"

	# Make encapsulating patch
	points = [[i,val] for i,val in enumerate(extra_border_min)]
	points.extend([[i,val] for i,val in reversed(list(enumerate(extra_border_max)))])
	ply = mpatches.Polygon(points, alpha=0.1, facecolor=std_color)

	# And plot
	plt.figure(figsize=(6.4, 3.2))
	plt.hold(True)
	plt.plot(avg_cum_reward, label="avg", color=line_color)
	plt.plot(extra_border_max, '--', label="max", color=std_color)
	plt.plot(extra_border_min, '--', label="min", color=std_color)
	ax = plt.gca()
	ax.add_patch(ply)
	plt.xlabel("Iterations")
	if entropy:
		plt.ylabel("Average $T$ Entropy")
	else:
		plt.ylabel("Cumulative Reward")
	if not entropy:
		plt.yscale('symlog')
	plt.tight_layout()
	plt.grid()
	#plt.legend()
	if outfile:
		plt.savefig(outfile)
		plt.close()
	else:
		plt.show()


def calc_state_histogram(data):
	""" Calculates the statistical data needed to plot the state histogram """
	# List of vectors containing the count according to state value for each trial
	# vecs[i][0] contains the number of iterations on the most valuable state in
	# trial i, and so on
	vecs = []

	# This needs to be tackled for each trial
	for entry in data:
		# Get the series of states into a local (for readability)
		series = entry["states"]
		# Get a list of states and of values
		states = [s for s in entry["v_s"]]
		values = [entry["v_s"][s] for s in entry["v_s"]]
		# States ordered from most to least valuable
		ordered_states = [x for _,x in sorted(zip(values,states), reverse=True)]
		# Create occurrence vector
		count = [series.count(list(s)) for s in ordered_states]
		# Append to our list
		vecs.append(count)

	# Stats vectors
	sum_vec = [sum([vecs[i][j] for i in range(len(vecs))]) for j in range(len(vecs[0]))]
	avg_vec = [elem/len(vecs) for elem  in sum_vec]
	std_vec = [np.std([vecs[i][j] for i in range(len(vecs))]) for j in range(len(vecs[0]))]
	box_vec = [[vecs[i][j] for i in range(len(vecs))] for j in range(len(vecs[0]))]
	top_3_vec = [sum([vecs[i][j] for j in range(3)])/sum(vecs[i]) for i in range(len(vecs))]

	# vec[i] contains a value for rank state [i]
	# top_3_vec contains the fraction of iterations spent in the top 3 states for each run i
	return [sum_vec, avg_vec, std_vec, box_vec, top_3_vec]


def plot_state_histogram(filename, outfile=None):
	""" Plots information on what states the system spent the most time on. """
	# Read raw data
	# raw_data is a vector of vectors, each vector contains a full run of the algorithm
	data = []
	extension = filename.split(".")[1]
	if extension == "pkl":
		data = pickle.load(open(filename, "rb"))
	else:
		print("Error: I don't know this file extension!")
		return

	# Get vectors
	[sum_vec, avg_vec, std_vec, box_vec, _] = calc_state_histogram(data)

	# Define figure parameters
	col_width = 0.7
	left_padding = (1-col_width)/2
	offset = 0.5
	# Initial and final colors for the bars
	initial_color = [26/255, 17/255, 1.0, 1.0]
	final_color = [0.5, 0.0, 0.0, 1.0]
	# A small lambda to calculate our intermediate colors
	# It depends on a ton of local vars, I'm not sure that's cool
	custom_color = lambda t: [initial_color[i] + t/len(avg_vec)*(final_color[i]-initial_color[i]) for i in range(len(initial_color))]

	# And plot bars
	plt.figure(figsize=(6.4, 3.2))
	plt.hold(True)
	for i,elem in enumerate(avg_vec):
		plt.bar(i+left_padding+offset, elem, width=col_width, color=custom_color(i))
	plt.xlabel("State Ranks")
	plt.ylabel("Iterations")
	plt.gca().yaxis.grid()
	plt.xticks(range(1,len(avg_vec) + 1), range(1,len(avg_vec) + 1))
	plt.tight_layout()

	# And plot boxes
	# plt.figure()
	# plt.hold(True)
	# plt.boxplot(box_vec)

	# Show plot of save it
	if outfile:
		plt.savefig(outfile)
		plt.close()
	else:
		plt.show()

	#print(data[0]["states"])
	#print(data[0]["v_s"][tuple(data[0]["states"][0])])


def convert_log_to_pickle(filename):
	""" Convert a yaml log built during a simulation to a pickle file that can be read and written much faster. 
	This was not able to be performed with log files over 4008000 lines in length, since they basically exploded
	in memory. Beware. (or use the C++ implementation of the parser)
	"""
	# Inform
	proper_name = filename.split(".")[0]
	print("Converting {} file.".format(proper_name))

	# Read data from file
	start = time.time()
	print("Reading yaml... ", end="")
	sys.stdout.flush()
	with open(filename, "r") as yaml_file:
		data = yaml.load(yaml_file, Loader=Loader)
	print("done in {} seconds".format(time.time()-start))

	# Dump to pickle
	start = time.time()
	print("Dumping pickle... ", end="")
	pickle.dump(data, open(proper_name + ".pkl", "wb"))
	print("done in {} seconds".format(time.time()-start))

	# Try reading from pickle
	start = time.time()
	print("Reading pickle... ", end="")
	data = pickle.load(open(proper_name + ".pkl", "rb"))
	print("done in {} seconds".format(time.time()-start))


def calculate_table_entries(filename):
	""" Calculate the average +- std execution time as well as the avg +- std cumulative reward obtained. 
	Input file must be a pickle converted from a yaml log by the convert_log_to_pickle() function.
	"""
	# Load data
	data = pickle.load(open(filename, "rb"))

	# Calculate the statistics we need from the data
	[_, _, _, _, top_3_vec] = calc_state_histogram(data)

	# Build execution time and cumulative reward vectors
	# LIST COMPREHENSIONS RULE
	exec_time_vec = [d["execution_time_ms"] for d in data]
	cum_reward_vec = [sum(d["rewards"]) for d in data]
	entropy_vec = [d["entropies"][-1] for d in data]

	# Print out the good stuff
	print("Filename:", filename)
	cum_reward_mean = np.round(np.mean(cum_reward_vec), 3)
	cum_reward_std = np.round(np.std(cum_reward_vec), 3)
	exec_time_mean = np.round(np.mean(exec_time_vec), 3)
	exec_time_std = np.round(np.std(exec_time_vec), 3)
	top_3_mean = np.round(np.mean(top_3_vec)*100, 3)
	top_3_std = np.round(np.std(top_3_vec)*100, 3)
	entropy_mean = np.round(np.mean(entropy_vec), 3)
	entropy_std = np.round(np.std(entropy_vec), 3)
	#print("Cumulative Reward:")
	#print("Avg: {}, Std: {}".format(cum_reward_mean, cum_reward_std))
	#print("Execution Time:")
	#print("Avg: {}, Std: {}".format(exec_time_mean, exec_time_std))
	# Results table format: Final Cum Reward & Execution time & Iterations in top 3 & Final avg entropy
	print("Copyable:")
	print("${}\pm{}$ & ${}\pm{}$ & ${}\\%\pm{}$ & ${}\pm{}$".format(cum_reward_mean, cum_reward_std, exec_time_mean, exec_time_std, top_3_mean, top_3_std, entropy_mean, entropy_std))
	print()


def write_test_yaml():
	""" Utility function to illustrate a YAML, for designing the julia YAML writer. """
	with open("test.yaml", "w") as yaml_file:
		data = dict()
		data["states"] = [[1,2], [2,3], [1,2]]
		data["rewards"] = [2.3454, 2.345, 12.3421]
		data["actions"] = [1,2,3,4,3,2]
		v_s = {(1,2):3}
		data["v_s"] = v_s
		yaml.dump([data], yaml_file, default_flow_style=False)


def read_test_yaml(filename):
	""" Utility function to read a YAML, for testing the julia YAML writer. """
	with open(filename, "r") as yaml_file:
		data = yaml.load(yaml_file)
		print(data)


def plot_connection_graph(connection_matrix, state):
	""" Draw the connectivity graph for all agents.

	It takes the position of each agent, and draw a circle.
	Then, it connects the circles with lines depending on the connection
	strength.
	"""
	# A small lambda function to calculate our intermediate colors
	bad_conn_color = [1.0, 0.0, 0.0, 1.0]
	good_conn_color = [0.0, 1.0, 0.0, 1.0]
	custom_color = lambda t: [bad_conn_color[i] + t*(good_conn_color[i]-bad_conn_color[i]) for i in range(len(bad_conn_color))]

	# TODO: get connection matrix and positions from input args

	# Define bogus data
	num_agents = 10
	positions = []
	for i in range(num_agents):
		positions.append([10*random.random(), 10*random.random()])

	connection_matrix = []
	for i in range(num_agents):
		connection_matrix.append([])
		for j in range(num_agents):
			connection_matrix[i].append(random.random())

	# Draw circles
	# https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
	# https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
	fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
	for i in range(len(positions)):
		ax.add_artist(plt.Circle(positions[i], 0.1, color='k', zorder=10))
		
	# Draw lines
	# TODO: Lines somehow appear on top
	conn_thresh = 0.3
	for i in range(len(connection_matrix)):
		for j in range(len(connection_matrix[0])):
			if i != j:
				if connection_matrix[i][j] > conn_thresh:
					plt.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], color=custom_color(connection_matrix[i][j]))

	# Set axes limits and hide them
	ax.set_aspect('equal', 'box')
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# Show/save the figure
	plt.show()


def plot_task_assignments(task_history):
	""" This function receives a task_history of the form
	[
		[agent, task, timestamp],
		[agent, task, timestamp],
		...
	]

	which is then plotted. All values are ints, and time is in
	seconds.

	TODO: Deal with out-of-order tasks
	TODO: Deal with overlap
	"""
	# Constants
	bar_h = 0.8
	bar_s = 1-bar_h
	agent_colors = ['b', 'r', 'g', 'k']

	# Define bogus data
	data = [
		[1, 1, 10],
		[3, 2, 5],
		[2, 3, 11],
		[1, 2, 15],
		[3, 1, 15],
		[2, 1, 23],
		[3, 3, 23],
	]
	end_time = 25

	# Convert data to bars
	bars = []
	agents = set([a[0] for a in data])
	# Populate the list of bars agent-by-agent for simplicity
	for agent in agents:
		agent_actions = [a for a in data if a[0] == agent]
		# For all elements except the last
		for i in range(len(agent_actions)-1):
			bar = [agent_actions[i][0],
			       agent_actions[i][1],
			       agent_actions[i][2],
			       agent_actions[i+1][2] - agent_actions[i][2],
			       ]
			bars.append(bar)
		# Last element needs exception case
		# Use end-time to calculate the duration of the last element
		bar = [agent_actions[-1][0],
		       agent_actions[-1][1],
		       agent_actions[-1][2],
		       end_time - agent_actions[-1][2],
		       ]
		bars.append(bar)


	# Start new fig
	fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
	plt.hold(True)

	# Plot
	#ax.barh(y, width, height, left
	temp = list(agents)
	for bar in bars:
		ax.barh(bar[1]-bar_h/2, bar[3], bar_h, bar[2], color=agent_colors[bar[0]-1], label="Agent {}".format(bar[0]) if bar[0] in temp else "")
		try:
			temp.remove(bar[0])
		except ValueError:
			pass

	# Create a Legend
	plt.legend()

	# Show plot
	plt.show()


if __name__ == "__main__":

	#plot_connection_graph(None, None)
	plot_task_assignments(None)
	exit()

	# Configure matplotlib
	rc('text', usetex=True)

	# Plot the learning distribution example:
	#plot_distribution_example()

	# Define relevant files
	# Final test cases start here
	# File names are defined with no extension on purpose.
	other_files = [
	                 # "results/random_sarsop_svr_100_1_0_1000",
	                 # "results/random_sarsop_isvr_100_1_0_1000",
	                 # "results/random_sarsop_svr_100_20_0_1000",
	                 # "results/random_sarsop_isvr_100_20_0_1000",
	                 # "results/random_qmdp_svr_100_1_0_1000",
	                 # "results/random_qmdp_svr_100_20_0_1000",
	                 # "results/random_qmdp_isvr_100_1_0_1000",
	                 # "results/random_qmdp_isvr_100_20_0_1000",
	                 # "results/random_qmdp_msvr_100_1_0_1000",
	                 # "results/random_qmdp_msvr_100_20_0_1000",
	                 # "results/random_sarsop_msvr_100_1_0_1000",
	                 # "results/random_sarsop_msvr_100_20_0_1000"
	                 #"results/hri_results",
	                 #"results/random_qmdp_svr_100_1_0_100_new_struct"
	                 "results/random_qmdp_svr_100_5_0_100_new_struct",
	                 "results/random_qmdp_svr_100_20_0_100_new_struct",
	                 "results/random_qmdp_svr_100_-1_0_100_new_struct"
	              ]
	all_files = []
	# Feel free to add more stuff here
	all_files.extend(other_files)

	# Convert files to pickle:
	for f in all_files:
		convert_log_to_pickle(f+".yaml")

	# Calculate the stuff we want for the table:
	for f in all_files:
		calculate_table_entries(f+".pkl")

	# Plot stuff
	for f in all_files:
		plot_timeseries_data(f+".pkl", f+"_reward.pdf")

	for f in all_files:
		plot_timeseries_data(f+".pkl", f+"_entropy.pdf", entropy=True)

	for f in all_files:
		plot_state_histogram(f+".pkl", f+"_hist.pdf")
	