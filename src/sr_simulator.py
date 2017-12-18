#!/usr/bin/env python

# Futures
from __future__ import print_function
from __future__ import division

# ROS Imports
import rospy
import rospkg
from apomdp.msg import action
from apomdp.msg import obs
from apomdp.msg import shared_data
from apomdp.srv import Act
import apomdp.srv
import pickle

# STD Imports
import yaml
from threading import Lock
import math
import random
import copy
import time


# Global Variables
state = [] 					# Maintains the state of the simulated world
connection_matrix = [] 		# Maintains the connectivity of the agents
connectivity_constant = 0	# Connectivity constant (see problem.yaml)
shared_data_pubs = [] 		# Maintains publishers for each individual agent
global_lock = [] 			# Mutex for controlling critical sections

def initialize_system(common_data_filename, team_config_filename, problem_config_file):
	""" Receives a file name and initializes the global variables according to
	the information contained therein.

	The function assumes the file is YAML
	"""
	# TODO: Implement
	# Inform
	rospy.loginfo("Initializing system. Files loaded:\n{}\n{}".format(common_data_filename, team_config_filename))

	# Read data from files
	common_data = []
	team_config = []
	problem_data = []
	with open(common_data_filename) as data_file:
		common_data = yaml.load(data_file)
	with open(team_config_filename) as data_file:
		team_config = yaml.load(data_file)
	with open(problem_config_file) as data_file:
		problem_data = yaml.load(data_file)

	# Inform
	rospy.loginfo("Common information: {}".format(common_data))
	rospy.loginfo("Team configuration: {}".format(team_config))
	rospy.loginfo("Problem: {}".format(problem_data))

	# Build state vector
	global state
	# Get first state
	state = problem_data["initial_state"]

	# Set connectivity_constant
	global connectivity_constant
	connectivity_constant = problem_data["connectivity_constant"]

	# Create first connection matrix
	global connection_matrix
	connection_matrix = calc_connection_matrix(state, connectivity_constant)
	print(connection_matrix)

	# TODO: think of other stuff to have here


def calc_connection_matrix(state, c):
	""" Given the current state of the system, this function generates the
	associated connectivity matrix containing the probability of message
	delivery to and from every agent.
	"""
	# Allocate the Matrix
	n_agents = len(state["Agents"])
	connection_matrix = [[0 for x in range(n_agents)] for y in range(n_agents)]

	# Calculate matrix values
	for i in range(n_agents):
		for j in range(n_agents):
			# Import agent coordinates to local variables
			a1 = state["Agents"][i]
			a2 = state["Agents"][j]
			# Calculate dist between agents
			d = math.sqrt((a1[0]-a2[0])**2 + (a1[1]-a2[1])**2)
			# Apply inverse-square law
			try:
				connection_matrix[i][j] = c*(1/(d**2))
			except ZeroDivisionError:
				# If the agents have distance zero, probability is maximum
				connection_matrix[i][j] = 1.0

	# Return matrix
	return connection_matrix


def generate_observation(state, action, agent):
	""" Given an end state and the action that got us there, this function
	generates the observation corresponding to the agent specified.
	"""
	# TODO: Implement
	rospy.loginfo("Generating an observation for new state for action {} of agent {}.".format(action, agent))
	pass


def transition(state, action, agent_id):
	""" Updates the state of the world according to the action that
	was received. Returns the updated state of the world.
	"""
	# Get the position of the agent
	x,y = state["Agents"][agent_id]

	# Copy state to output variable
	new_state = copy.deepcopy(state)

	# Process each possible action
	# If action is put out fire
	if action == 0:
		# If there is fire in the current position, it gets put out
		if state["World"][x][y][1] == 1:
			new_state["World"][x][y][1] = 0
	# If action is remove debris
	elif action == 1:
		# If there is debris in the current position, it gets removed
		if state["World"][x][y][2] == 1:
			new_state["World"][x][y][2] = 0
	# If action is extract person
	elif action == 2:
		# If there is a victim in the current position, they get extracted
		if state["World"][x][y][3] == 1:
			new_state["World"][x][y][3] = 0
	# If action is move
	elif action > 2:
		# Determine where the agent wants to move
		a = action - 3 # "Pull" the action value to index the map correctly
		target_x = a // len(state["World"])
		target_y = a - target_x*len(state["World"])
		# Move them there if possible
		if state["World"][target_x][target_y][0] == 0:
			new_state["Agents"][agent_id][0] = target_x
			new_state["Agents"][agent_id][0] = target_y
		# Drop them on the way if not
		# TODO

	# Return the newly-constructed state
	return new_state


def broadcast(msg):
	""" Callback fuction for the /broadcast topic which propagates messages
	among agents according to connectivity.
	"""
	rospy.loginfo("I'm broadcasting data from agent {}!".format(msg.agent_id))

	# For each other agent in the team
	n_agents = len(state["Agents"])
	for a in range(n_agents):
		# Agents won't broadcast to themselves
		if msg.agent_id != a:
			# Get probability of delivery
			prob = connection_matrix[msg.agent_id][a]
			# If so, publish the message in the topic
			if random.random() < prob:
				shared_data_pubs[a].publish(msg)


def receive_action(req):
	""" Callback function for the act service which gets an action from
	a specific agent and returns the observation to that agent.
	"""
	# Lock the global var mutex
	global_lock.acquire()

	# Inform
	rospy.loginfo("I have received an action!")

	# Update the state of the world
	rospy.loginfo("Updating world state!")
	global state
	state = transition(state, req.a.action, req.a.agent_id)

	# Update the connectivity
	global connection_matrix
	connection_matrix = calc_connection_matrix(state, connectivity_constant)

	# Release the global var mutex
	global_lock.release()

	# Generate an observation of the new state
	observation = generate_observation(state, req.a.action, req.a.agent_id)

	# Pack it into a response message
	# TODO
	res = apomdp.srv.ActResponse()

	# Return the response
	rospy.loginfo("Returning observation to agent.")
	return res


if __name__ == "__main__":
	# Initialize ROS node
	rospy.init_node("sr_simulator")
	
	# Initialize state
	rospack = rospkg.RosPack()
	common_filename = rospack.get_path('apomdp') + "/config/common.yaml"
	team_filename = rospack.get_path('apomdp') + "/config/team.yaml"
	problem_filename = rospack.get_path('apomdp') + "/config/problem.yaml"
	initialize_system(common_filename, team_filename, problem_filename)

	# Launch servers etc
	rospy.Subscriber("broadcast", shared_data, broadcast)
	rospy.Service('act', Act, receive_action)

	# Start the publishers for each agent
	n_agents = len(state["Agents"])
	for a in range(n_agents):
		shared_data_pubs.append(rospy.Publisher('shared_data/{}'.format(a), shared_data, queue_size=10))

	# Initialize mutexes
	global_lock = Lock()

	# Wait for stuff to happen
	rospy.spin()