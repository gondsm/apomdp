#!/usr/bin/env julia

# The main goal of this script is to : 
# 1) Serves as an aPOMDP interface through: 
#	 1) Re-calculating Policies
#	 2) Choosing Actions 
# 2) Serves as a ROS interface through: 
#	1) Process Observations 
#	2) Communication with other agents 
# 3) Control the calculations: 
#	1) Fuse Information (beliefs from every agent)
# 	2) Fuse Transistions 

# Copyright 

# License 

# Imports, includes et al
include("./apomdp.jl")
using RobotOS
import YAML
import JSON
import Base

# Create our ROS types
@rosimport apomdp.srv: Act 
@rosimport apomdp.msg: shared_data
rostypegen()
using apomdp.srv
using apomdp.msg

# Global vars
# These need to be global in order to be written to by the callback
# TODO: Figure out inter-thread communication
global beliefs_vector
global transitions_vector
global pomdp
beliefs_vector = []
transitions_vector = []


# Agent-specific actions
# Function that returns the state values for this particular agent
# It receives a state index and returns an int.
function get_v_s(state, state_lut)
    # Convert to bpompd state
    bpomdp_state = idx_to_state(state[1], state_lut)

    # Calculate the state's value
    value = 0
    for node in keys(bpomdp_state["World"])
        value -= sum(bpomdp_state["World"][node])
    end

    # And return it
    return value
end 


# Function that returns the cost vector for all actions for this agent
function calc_cost_vector(node_locations, agent_id, n_actions, belief, pomdp, agent_abilities, state_lut) 
    # C_i(a,s) = Ab(a) * Cc(a,s)
    # The final cost of each action in each state is the product of the fixed
    # action cost (abilities or Ab) times the current cost (Cc) of the action
    # in the current state

    # Determine the current state
    index = argmax_belief(pomdp, belief)

    # Convert state index to actual state
    # After this, we should have a state dictionary, as in the simulator
    # The first call gets a state vector in the aPOMDP style, the second
    # converts that to a bPOMDP-style dict.
    state = idx_to_state(index, state_lut)

    # Calculate the cost vector 
    cost_vector = zeros(Float32, n_actions)
    
    # Actually calculate the cost
    # For non-movement actions, the cost will be the inverse of the agent's
    # abilities: if an action is desired, its cost is 0, if it is undesireable
    # then its cost is 1, and if impossible its cost is 10.
    for (i, ability) in enumerate(agent_abilities[agent_id])
        if  ability == -1
            cost_vector[i] = 1000
        elseif  ability == 0
            cost_vector[i] = 1
        elseif  ability == 1
            cost_vector[i] = 0
        end
    end

    # For movement actions, the cost will be equal to the Euclidean distance
    # between the current node and the target node.
    n_abilities = length(agent_abilities[agent_id])
    for i in n_abilities+1:n_actions
        # Retrieve current and target nodes
        curr_node = state["Agents"][agent_id][1]
        target_node = i - n_abilities

        # Retrieve their locations
        curr_loc = node_locations[curr_node]
        target_loc = node_locations[target_node]

        # Calculate Euclidean dist
        dist = norm(curr_loc - target_loc)

        # Done!
        cost_vector[i] = dist
    end

    # And we're done costing!
    return cost_vector
end 


# ROS-related functions
# call simulation (function ?? or just from the main)
function act(action, agent_id, service_client)
    # Build ROS message
    ros_action = ActRequest()
    ros_action.a.action = action
    ros_action.a.agent_id = agent_id
    fieldnames(ros_action)

    # Call ROS service
    o = service_client(ros_action)

    # And return the observation    
    return YAML.load(o.o.obs)
end


# publish to broadcast topic (function ?? or just from the main) 
function share_data(beliefs_vector, transitions_vector, publisher, agent_id)
    # Create ROS message 
    msg = shared_data()
    msg.agent_id = agent_id

    # Pack the belief and transistions in one ROS message 
    # TODO: refurbish when messages stabilize. YAML?
    msg.b_s = JSON.json(beliefs_vector[agent_id].dist)
    msg.T = JSON.json(transitions_vector[agent_id])
    
    # Publish message 
    publish(publisher, msg)
end


# callback function for shared data (belief, transistion..etc)
function share_data_cb(msg)
    # every time we recieve new msg we need to update the vectors only with latest information (no repetition)
    println("I got a message from agent_$(msg.agent_id)")

    # Parse other agent's belief into the vector
    new_belief = apomdpDistribution(pomdp)
    new_belief.dist = convert(Array{Float64,1}, JSON.parse(msg.b_s))
    beliefs_vector[msg.agent_id] = new_belief
    println("The other agent believes we're in state:")
    println(argmax_belief(pomdp, beliefs_vector[msg.agent_id]))

    # TODO: Parse other agent's belief into the vector
    # (there's no way Julia will just let this one slide)
    transitions_vector[msg.agent_id] = JSON.parse(msg.T)
end

# Functions for converting to/from the states and indices using the LUT
function state_to_idx(state, state_lut)
    idx = 0
    for (i, s) in enumerate(state_lut)
        if s == state
            idx = i
        end
    end
    return idx
end

function idx_to_state(idx, state_lut)
    return state_lut[idx]
end


# And a main function
function main(agent_id)
    # Initialize ROS node
    println("Initializing bPOMDP - agent_$(agent_id)")
    tic()
    init_node("agent_$(agent_id)")
    start_time = now()

    # Create the service client object
    service_client = ServiceProxy{Act}("act")

    # Read configuration
    config_file = expanduser("~/catkin_ws/src/apomdp/config/common.yaml")
    state_lut_file = expanduser("~/catkin_ws/src/apomdp/config/state_lut.yaml")
    println("Reading configuration from ", config_file)
    config = YAML.load(open(config_file))

    # Parse configuration
    n_agents = config["n_agents"]
    agents_structure = config["agents_structure"]
    world_structure = config["world_structure"]
    node_locations = config["node_locations"]
    node_connectivity = config["node_connectivity"]
    agent_abilities = config["agent_abilities"]
    n_nodes = length(node_locations)
    n_actions = n_nodes + length(agent_abilities[1])
    println("Read a configuration file:")
    println("\tn_actions: ", n_actions)
    println("\tn_agents: ", n_agents)
    println("\tn_nodes: ", n_nodes)
    println("\tagents_structure: ", agents_structure)
    println("\tworld_structure: ", world_structure)
    println("\tnode_locations: ", node_locations)
    println("\tnode_connectivity: ", node_connectivity)
    println("\tagent_abilities: ", agent_abilities)

    # Read state LUT
    println("Reading state LUT from ", state_lut_file)
    global state_lut
    state_lut = YAML.load(open(state_lut_file))
    #println(state_lut)
    n_states = length(state_lut)
    println("Read an LUT with ", n_states, " states.")

    # Allocate belief and transition vectors
    global beliefs_vector
    global transitions_vector

    beliefs_vector = Dict()
    transitions_vector = Dict()

    for i in 1:n_agents
        beliefs_vector[i] = nothing
        transitions_vector[i] = nothing
    end

    # Create a pomdp object. This contains the structure of the problem,
    # and has to be used in almost all calls to apomdp.jl in order to
    # inform the functions of the problem structure
    # All agents need to build aPOMDP structs that are the same, so that
    # they inform the aPOMDP primitives in the same way and ensure
    # consistency.
    print("Creating aPOMDP object... ")
    global pomdp
    pomdp = aPOMDP(n_states, n_actions, get_v_s, state_lut)

    # Initialize this agent's belief
    beliefs_vector[agent_id] = initialize_belief(pomdp)
    transitions_vector[agent_id] = initialize_transitions(pomdp)

    # Subscribe to the agent's shared_data topic
    # Create publisher, and a subscriber 
    pub = Publisher{shared_data}("broadcast", queue_size=10)
    sub = Subscriber{shared_data}("shared_data/$(agent_id)", share_data_cb, queue_size=10)

    # Inform on initialization time
    elapsed = toq()
    println("Initialization completed in $elapsed seconds.")
    println()

    # Start main loop
    println("Going into execution loop!")
    iter = 0
    while ! is_shutdown()
        # Inform
        println("Iteration $iter")

        # Get action costs for the current state
        println("Calculating cost vector")
        c_vector = calc_cost_vector(
            node_locations,
            agent_id,
            n_actions,
            beliefs_vector[agent_id],
            pomdp,
            agent_abilities,
            state_lut
            )

        # Fuse new transition with everyone else's
        fused_T = fuse_transitions(pomdp, transitions_vector)

        # Solve
        println("Calculating new policy")
        policy = get_policy(pomdp, fused_T, c_vector, get_v_s)

        # Call fuse belief function
        println("Fusing beliefs")
        fused_b = fuse_beliefs(pomdp, beliefs_vector)

        # Decision_making (action selection)
        action = get_action(pomdp, policy, fused_b)

        # Act and receive an observation 
        println("Applying action $action")
        observation = act(action, agent_id, service_client)
        index = state_to_idx(observation, state_lut)
        println("Got an index: ", index)

        # Update belief
        println("Updating local belief") 
        #TODO: rethink if we should use the local transition or the fused one
        previous_b = beliefs_vector[agent_id] # save the previous belief 
        beliefs_vector[agent_id] = update_belief(pomdp, index, action, beliefs_vector[agent_id], transitions_vector[agent_id])

        # Learn
        transitions_vector[agent_id] = learn(pomdp, beliefs_vector[agent_id], action, previous_b, transitions_vector[agent_id])

        # Publish something to "broadcast" topic
        # TODO: decide when to do this  
        println("Sharing data with other agents")
        share_data(beliefs_vector, transitions_vector, pub, agent_id)

        # End-of-loop
        println("================================================")
        iter += 1
        sleep(1)
    end

    # Inform
    println("Agent exiting!")

    # TODO: Write apomdp logs
end


# If this module is called as main
if size(ARGS)[1] == 0
    println("For now, I need to receive an int on the command line to use as agent ID.")
    exit(-1)
else
    id = parse(Int64,ARGS[1])
    if id < 1
        println("Agent indices start at 1.")
    else
        main(id)
    end
end