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
beliefs_vector = []
transitions_vector = []


# Agent-specific actions
#this function will return the v(s)
function get_v_s(state)
    # Algorithm:
    # Convert apomdp state to bpomdp state (where each var has meaning)
    # Apply the equation:
    # v(s) = -2n_victim * -n_depries * n_fire

    # For now, we'll return 0, and all states will have the same value
    return 0
end 


# this function will retun cost vector for all actions for this agent
function calc_cost_vector(node_connectivity, n_agents, world_structure, agent_id, n_actions, belief, pomdp, agent_abilities) 
    # C_i(a,s) = Ab(a) * Cc(a,s)
    # The final cost of each action in each state is the product of the fixed
    # action cost (abilities or Ab) times the current cost (Cc) of the action
    # in the current state

    # Steps
    # Get abilites vector (from global var?)
    # Calculate the current cost for each individual action
    # Get the final vector as the vector product of the previous two

    # TODO: re-implement
    return zeros(Int8, n_actions)

    # Get the ability 
    println("=======agent_abilities=======")
    for a in agent_abilities
        println(a)
    end
    
    # TODO: get the state from belief using argmax that has the highest probability 
    # Question: is it one belief vector of beliefs ? if beliefs.  will get highest state  index from all vectors, and then how used all to cal cost?
    # We will assume that the index of the state we got from the belief  
    index = 11 

    # Getting this index we pass it to function to retrieve the state
    state = state_from_index(pomdp,index)
    println("=========state=========")
    println(state)
    #println(state[1])

    #take the cells states from the state 
    size_world_structure = size(world_structure,1)
    
    #println("index",size_world_structure+(size_world_structure*(state[agent_id]-1)))#for testing
   
    # calculate the cost vector 
    cost_vector = zeros(Int8, n_actions)
    
    for i in 1:n_actions       
        node_connectivity_index = 1
        #actions of movement 
        if i>size_world_structure && i<n_actions
            println("i: ",i)
            if node_connectivity[agent_id][node_connectivity_index] != 0 #can move to a node 
                cost_vector[i] = 1*agent_abilities[agent_id][node_connectivity_index]
            else 
                cost_vector[i] = 0     
            end 
        elseif i == n_actions #action is stop 
           cost_vector[i] = 0
        else  #first three actions fire, victim_debris 
            for j in 1:size(world_structure,1)
                if i == j #action and the world_structure match state 
                    println("i: ",i)
                    cost_vector[i]=(state[size_world_structure+(size_world_structure*(state[agent_id]-1))])*agent_abilities[agent_id][i]#+(j-1)
                end 
            end
        end    

        node_connectivity_index+=1
    end 
    println("cost_vector: ",cost_vector)

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
    msg.b_s = JSON.json(beliefs_vector[agent_id])
    msg.T = JSON.json(transitions_vector[agent_id])
    
    # Publish message 
    publish(publisher, msg)
end


# callback function for shared data (belief, transistion..etc)
function share_data_cb(msg)
    # every time we recieve new msg we need to update the vectors only with latest information (no repetition)
    println("I got a message from agent_$(msg.agent_id)")
    beliefs_vector[msg.agent_id] = JSON.parse(msg.b_s)
    transitions_vector[msg.agent_id] = JSON.parse(msg.T)
end


# And a main function
function main(agent_id)
    # TODO: make this consistent (1-indexed)
    agent_index = agent_id

    # Initialize ROS node
    println("Initializing bPOMDP - agent_$(agent_id)")
    tic()
    init_node("agent_$(agent_id)")
    start_time = now()

    # Create the service client object
    service_client = ServiceProxy{Act}("act")

    # Read configuration
    config_file = expanduser("~/catkin_ws/src/apomdp/config/common.yaml")
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
    println("\tn_actions: ",n_actions)
    println("\tn_agents: ",n_agents)
    println("\tn_nodes: ",n_nodes)
    println("\tagents_structure: ",agents_structure)
    println("\tworld_structure: ",world_structure)
    println("\tnode_locations: ",node_locations)
    println("\tnode_connectivity: ",node_connectivity)
    println("\tagent_abilities: ",agent_abilities)

    # Show state structure
    state_structure = convert_structure(n_agents, n_nodes, agents_structure, world_structure)
    println("This corresponds to aPOMDP state structure $state_structure.")
    elapsed = toq()
    println("Config read in $elapsed seconds.")
    println()

    # Allocate belief and transition vectors
    global beliefs_vector
    global transitions_vector
    beliefs_vector = [nothing for a in 1:n_agents]
    transitions_vector = [nothing for a in 1:n_agents]

    # Create a pomdp object. This contains the structure of the problem,
    # and has to be used in almost all calls to apomdp.jl in order to
    # inform the functions of the problem structure
    # All agents need to build aPOMDP structs that are the same, so that
    # they inform the aPOMDP primitives in the same way and ensure
    # consistency.
    print("Creating aPOMDP object... ")
    tic()
    pomdp = aPOMDP(n_actions, n_agents, agents_structure, n_nodes, world_structure, node_locations, node_connectivity, get_v_s)
    elapsed = toq()
    println("Done in $elapsed seconds.")

    # Subscribe to the agent's shared_data topic
    # Create publisher, and a subscriber 
    pub = Publisher{shared_data}("broadcast", queue_size=10)
    sub = Subscriber{shared_data}("shared_data/$(agent_id)", share_data_cb, queue_size=10)

    # Start main loop
    println("Going into execution loop!")
    iter = 0
    while ! is_shutdown()
        # Inform
        println("Iteration $iter")

        # Get the cost 
        println("Calculating cost vector")
        c_vector = calc_cost_vector(node_connectivity,
            n_agents,
            world_structure,
            agent_id,
            n_actions,
            beliefs_vector[agent_index],
            pomdp,
            agent_abilities
            )

        # Solve
        println("Calculating new policy")
        #policy = get_policy(pomdp, fused_T, c_vector)

        # Call fuse belief function
        println("Fusing beliefs")
        #fused_b = fuse_beliefs(pomdp, beliefs_vector)

        # Decision_making (action selection)
        #action = get_action(pomdp, policy, fused_b)
        action = rand(0:n_actions-1)

        # Act and receive an observation 
        println("Applying action $action")
        observation = act(action, agent_id, service_client)
        # TODO: Get apomdp state (index)
        #temp_s = state_b_to_a(pomdp, observation)
        #index = POMDPs.state_index(pomdp, temp_s)
        #temp_s = state_from_index(pomdp,index)
        index = 1

        # Update belief
        println("Updating local belief") 
        #TODO: rethink if we should use the local transition or the fused one
        previous_b = beliefs_vector[agent_index] # save the previous belief 
        beliefs_vector[agent_index] = update_belief(pomdp, index, action, beliefs_vector[agent_index], transitions_vector[agent_index])
        
        # Fuse beliefs
        fused_b = fuse_beliefs(pomdp, beliefs_vector)

        # Learn
        # TODO: decide whether to use the fused_b or local belief to learn
        transitions_vector[agent_index] = learn(pomdp, beliefs_vector[agent_index], action, previous_b, transitions_vector[agent_index])
        
        # Fuse new transition with everyone else's
        fused_T = fuse_transitions(pomdp, transitions_vector)

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