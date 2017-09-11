#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import rc
import matplotlib.patches as mpatches
import itertools
import numpy as np
import yaml

def normalize(lst):
	return [float(i)/sum(lst) for i in lst]


def calc_reward(states, v_s, transition_dist):
	reward = 0.0
	for i,state in enumerate(states):
		reward += transition_dist[i]*v_s[state]
	return reward


def plot_distribution_example():
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


def plot_timeseries_data(filename, outfile=None):
	""" Plot some nice timeseries obtained from the simulations. """
	# Read raw data
	# raw_data is a vector of vectors, each vector contains a full run of the algorithm
	raw_data = []
	with open(filename) as data_file:
		for line in data_file:
			if line[0] != '#':
				raw_data.append([float(elem) for elem in line.split()])

	# Split for iteration
	# Data will be a vector of vectors, with data [i] corresponding to all points at iteration i.
	data = []
	for i in range(len(raw_data[0])):
		data.append([elem[i] for elem in raw_data])

	# Calculate cumulative reward
	cum_reward = []
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

	# Make encapsulating patch
	points = [[i,val] for i,val in enumerate(extra_border_min)]
	points.extend([[i,val] for i,val in reversed(list(enumerate(extra_border_max)))])
	ply = mpatches.Polygon(points, alpha=0.1, facecolor="green")

	# And plot
	plt.figure(figsize=(6.4, 3.2))
	plt.hold(True)
	plt.plot(avg_cum_reward, label="avg")
	plt.plot(extra_border_max, label="max", color="green")
	plt.plot(extra_border_min, label="min", color="green")
	ax = plt.gca()
	ax.add_patch(ply)
	plt.xlabel("Iterations")
	plt.ylabel("Cumulative Reward")
	plt.yscale('symlog')
	plt.tight_layout()
	plt.grid()
	#plt.legend()
	if outfile:
		plt.savefig(outfile)
	else:
		plt.show()


def write_test_yaml():
	with open("test.yaml", "w") as yaml_file:
		data = dict()
		data["states"] = [[1,2], [2,3], [1,2]]
		data["rewards"] = [2.3454, 2.345, 12.3421]
		data["actions"] = [1,2,3,4,3,2]
		yaml.dump([data], yaml_file, default_flow_style=False)

def read_test_yaml():
	with open("test.txt", "r") as yaml_file:
		print(yaml.load(yaml_file))


if __name__ == "__main__":
	rc('text', usetex=True)
	#plot_distribution_example()
	#plot_timeseries_data("p0i1000.txt", "p0i1000_2std.pdf")
	#plot_timeseries_data("p1i1000.txt", "p1i1000_2std.pdf")
	#plot_timeseries_data("p5i1000.txt", "p5i1000_2std.pdf")
	#write_test_yaml()
	read_test_yaml()