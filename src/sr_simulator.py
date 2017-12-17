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

# STD Imports
import yaml


# Global Variables
state = [] # Maintains the state of the simulated world
connection_matrix = [] # Maintains the connectivity of the agents

def initialize_system(common_data_filename, team_config_filename):
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
	with open(common_data_filename) as data_file:
		common_data = yaml.load(data_file)
	with open(team_config_filename) as data_file:
		team_config = yaml.load(data_file)

	# Inform
	rospy.loginfo("Common information: {}".format(common_data))
	rospy.loginfo("Team configuration: {}".format(team_config))

	# Build state vector
	global state
	# TODO

	# Select first state
	# TODO


	# Create first connection matrix
	global connection_matrix
	connection_matrix = calc_connection_matrix(state)

	# TODO: think of other stuff to have here


def calc_connection_matrix(state):
	""" Given the current state of the system, this function generates the
	associated connectivity matrix containing the probability of message
	delivery to and from every agent.
	"""
	# TODO: Implement
	pass


def generate_observation(state, action, agent):
	""" Given an end state and the action that got us there, this function
	generates the observation corresponding to the agent specified.
	"""
	# TODO: Implement
	rospy.loginfo("Generating an observation for new state for action {} of agent {}.".format(action, agent))
	pass


def update_state(state, action):
	""" Updates the state of the world according to the action that
	was received. Returns the updated state of the world.
	"""
	# TODO: implement
	pass


def broadcast(msg):
	""" Callback fuction for the /broadcast topic which propagates messages
	among agents according to connectivity.
	"""
	# TODO: Implement based on connectivity
	rospy.loginfo("I'm broadcasting data from agent {}!".format(msg.agent_id))


def receive_action(req):
	""" Callback function for the act service which gets an action from
	a specific agent and returns the observation to that agent.
	"""
	# Inform
	rospy.loginfo("I have received an action!")

	# Update the state of the world
	rospy.loginfo("Updating world state!")
	global state
	state = update_state(state, req.a.action)

	# Update the connectivity
	global connection_matrix
	connection_matrix = calc_connection_matrix(state)

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
	rospy.Subscriber("broadcast", shared_data, broadcast)
	rospy.Service('act', Act, receive_action)

	# Initialize state
	rospack = rospkg.RosPack()
	common_filename = rospack.get_path('apomdp') + "/config/common.yaml"
	team_filename = rospack.get_path('apomdp') + "/config/team.yaml"
	initialize_system(common_filename, team_filename)

	# Wait for stuff to happen
	rospy.spin()