#!/usr/bin/env python

"""
This script implements a ROS node for simulating a search and rescue scenario. The node is
responsible for maintaining the (true) state of the world, receiving actions from the agents
replying with noisy observations. This essentially done based on the assumptions that are specified
in the config/*.yaml files.

The node is also responsible for broadcasting messages to agents, introducing noise in the
communication under the form of lost messages. This is done using the inverse-square law to
model a noisy channel with message loss.
"""

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
# The only functions that touch these variables are the ROS callbacks and the
# initialization function.
# The ROS callbacks use a mutex to ensure that there is no simultaneous
# access to the variables.
state = []                  # Maintains the state of the simulated world
connection_matrix = []      # Maintains the connectivity of the agents
connectivity_constant = 0   # Connectivity constant (see problem.yaml)
shared_data_pubs = []       # Maintains publishers for each individual agent
global_lock = []            # Mutex for controlling critical sections
log_dict = dict()           # A dictionary containing the full logs of the execution
nodes_location = dict()     # A dictionary containing the locations of nodes (see common.yaml)
node_connectivity = dict()  # A dictionary containing how nodes are connected (see common.yaml)
agent_abilities = []        # A dictionary containing the agents' abilities (see common.yaml)
n_actions = 0               # The action space (see common.yaml)


# Logging functions
def log_initial_state(log_dict, initial_state, initial_comm_matrix):
    """ Logs the initial state to the logging dictionary provided. """
    log_dict["initial_state"] = initial_state
    log_dict["initial_connection_matrix"] = initial_comm_matrix
    log_dict["transitions"] = []
    log_dict["communications"] = []


def log_transition(log_dict, action, agent, final_connection_matrix, final_state, observation):
    """ 
    Logs a transition to the logging dict provided.
    Since the initial state is logged elsewhere, this function logs which
    agent took which action and the *impact* of the action, meaning it logs the
    final connection matrix and the final state *only*. The previous values for
    both of these can always be determined from the previous entry in the log.
    """
    log_dict["transitions"].append({"action": action,
                                    "agent": agent,
                                    "observation": observation,
                                    "time": time.time(),
                                    "final_connection_matrix": final_connection_matrix,
                                    "final_state": final_state,})


def log_communication(log_dict, sender, deliveries):
    """ Logs """
    log_dict["communications"].append(
        {"sender": sender, "delivered_to": deliveries, "time": time.time()}
        )


def dump_log(filename, log):
    """ Dumps the logs to the yaml file. """
    with open(filename, "w") as out_file:
        yaml.dump(log, out_file, default_flow_style=False)


# Simulation functions
def initialize_system(common_data_filename, problem_config_filename):
    """ Receives a file name and initializes the global variables according to
    the information contained therein.

    The function assumes the file is YAML
    """
    # Inform
    rospy.loginfo("Initializing system. Files loaded:\n{}\n{}".format(
        common_data_filename, problem_config_filename)
                 )

    # Read data from files
    common_data = []
    problem_data = []
    with open(common_data_filename) as data_file:
        common_data = yaml.load(data_file)
    with open(problem_config_filename) as data_file:
        problem_data = yaml.load(data_file)

    # Inform
    rospy.loginfo("Common information: {}".format(common_data))
    rospy.loginfo("Problem: {}".format(problem_data))

    # Build state vector from first state
    global state
    state = problem_data["initial_state"]

    global nodes_location
    nodes_location = common_data["nodes_location"]
    print("nodes_location",nodes_location)

    global node_connectivity
    node_connectivity = common_data["node_connectivity"]
    print ("node_connectivity", node_connectivity)

    global agent_abilities
    agent_abilities = common_data["agent_abilities"]

    global n_actions
    n_actions = len(node_connectivity) + len(agent_abilities[1])

    # Set connectivity_constant
    global connectivity_constant
    connectivity_constant = problem_data["connectivity_constant"]

    # Create first connection matrix
    global connection_matrix
    connection_matrix = calc_connection_matrix(state, connectivity_constant)
    print("connection_matrix", connection_matrix)

    # Log first state
    log_initial_state(log_dict, state, connection_matrix)


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
            a1 = state["Agents"][i+1][0]
            a2 = state["Agents"][j+1][0]
            # Calculate dist between agents
            d = math.sqrt((a1-a2)**2 + (a1-a2)**2)
            print(d)
            # Apply inverse-square law
            try:
                connection_matrix[i][j] = c*(1/(d**2))
            except ZeroDivisionError:
                # If the agents have distance zero, probability is maximum
                connection_matrix[i][j] = 1.0

    # Return matrix
    return connection_matrix


def generate_observation(state, action, agent, noisy=True):
    """ Given an end state and the action that got us there, this function
    generates the observation corresponding to the agent specified.

    If noisy is True, then the observation is corrupted with noise for cells
    that are far away from the agent. The farther the cell, the higher the
    noise in the cells.
    """
    # TODO: Re-think the probability of noise to be lower near the agent and higher
    # farther away. (As it stands it's completely linear)
    # Inform
    rospy.loginfo("Generating an observation for new state for action {} of agent {}.".format(action, agent))
    
    # Copy the current state to the observation
    obs = copy.deepcopy(state)

    # Corrupt observation with noise
    # For each of the cells, we get the distance of the agent to the cell.
    # The larger the cell, the higher the probability of noise.
    # Then, for each bit in the cell, we sample a uniform distribution.
    # If the result in lower than the probability of noise, then we randomly
    # attribute a value to the bit.
    node = state["Agents"][agent] # Get node location of the agent

    # Ǵet the position of the node
    position_x = nodes_location[node][0]
    position_y = nodes_location[node][1]

    # Corrupt bits as a function of distance
    # TODO: fix this calculation, it appears to be using indices and not
    # actual position values (probably wrongly adapted from previous code)
    for i in range(len(state["World"])):
        for j in range(len(state["World"][i])):
            # Calculate distance/probability of noise for the cell
            dist = math.sqrt((i-position_x)**2 + (j-position_y)**2)
            prob = dist / len(state["World"]) # Normalized distance = probability of noise
            # Corrupt bits one by one
            if random.random() < prob:
                obs["World"][i][j] = random.randint(0, 1)

    # Return the observation
    return obs


def transition(state, action, agent_id):
    """ Updates the state of the world according to the action that
    was received. Returns the updated state of the world.
    """
    action = 0
    # Get the node of the agent
    node = state["Agents"][agent_id]

    # Copy state to output variable
    new_state = copy.deepcopy(state)

    # Process each possible action
    # If action is put out fire
    # TODO: completely rework this according to the new action definition:
    # [fire, victim, path, <move to node i>]
    if action == 0:
        # If there is fire in the current position, it gets put out
        if state["World"][node][0] == 1:
            new_state["World"][node][0] = 0
    # If action is remove debris
    elif action == 1:
        # If there is a person in the current position, they get removed
        if state["World"][node][1] == 1:
            new_state["World"][node][1] = 0
    # If action is extract person
    elif action == 2:
        # If there is debris in the current position, they get extracted
        if state["World"][node][2] == 1:
            new_state["World"][node][2] = 0
    # If action is move
    elif action > 2 and action != n_actions-2:
        new_node = node_connectivity[node][action]
        # TODO: change according to new actions
        if new_node == 0:
            new_state["Agents"][agent_id] = state["World"][node]
        # Direction will take you to another node
        else:
            new_state["Agents"][agent_id] = state["World"][new_state]

    print("agent in node: ", state["Agents"][agent_id])
    print("action taken:", action)
    print("new state: ", new_state["Agents"][agent_id])

    # Return the newly-constructed state
    return new_state


# ROS Message/Service Handlers
def broadcast(msg):
    """ Callback fuction for the /broadcast topic which propagates messages
    among agents according to connectivity.
    """
    rospy.loginfo("I'm broadcasting data from agent {}!".format(msg.agent_id))

    # Lock the global var mutex
    global_lock.acquire()

    # A list of the agents that the communication was delivered to
    deliveries = []

    # For each other agent in the team
    n_agents = len(state["Agents"])
    for a in range(n_agents):
        # TODO: check if agent indices start at 0 or 1
        # Agents won't broadcast to themselves
        if msg.agent_id != a:
            # Get probability of delivery
            prob = connection_matrix[msg.agent_id][a]
            # If so, publish the message in the topic
            if random.random() < prob:
                shared_data_pubs[a].publish(msg)
                deliveries.append(a)

    # Log this communication
    log_communication(log_dict, msg.agent_id, deliveries)

    # Release the global var mutex
    global_lock.release()


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

    # Generate an observation of the new state
    observation = generate_observation(state, req.a.action, req.a.agent_id)

    # Log the transition
    log_transition(log_dict, req.a.action, req.a.agent_id, connection_matrix, state, observation)

    # Release the global var mutex
    global_lock.release()

    # Pack it into a response message
    obs_string = yaml.dump(observation)
    res = apomdp.srv.ActResponse()
    res.o.obs = obs_string

    # Return the response
    rospy.loginfo("Returning observation to agent.")
    return res


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node("sr_simulator")

    # Initialize state
    rospack = rospkg.RosPack()
    common_filename = rospack.get_path('apomdp') + "/config/common.yaml"
    problem_filename = rospack.get_path('apomdp') + "/config/problem.yaml"
    initialize_system(common_filename, problem_filename)

    # Define log file location
    log_filename = rospack.get_path('apomdp') + "/config/{}_sim_log.yaml".format(int(time.time()))
    rospy.loginfo("Logs will be saved in {}.".format(log_filename))

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

    # Dump logs to file
    rospy.loginfo("Dumping logs!")
    dump_log(log_filename, log_dict)
