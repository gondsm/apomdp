#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools

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



if __name__ == "__main__":
	rc('text', usetex=True)
	plot_distribution_example()