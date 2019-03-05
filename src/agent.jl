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
global beliefs_vector
global transitions_vector
global observed_transitions
global pomdp
beliefs_vector = []             # A vector containing the apomdpDistribution beliefs of each agents_structure
transitions_vector = []         # A vector containing transitions of all agents
observed_transitions = Dict()   # A simpler representation of transitions to ease propagation


# Agent-specific actions
# Function that returns the state values for this particular agent
# It receives a state index and returns an int.
function get_v_s(state, state_lut)
    # Convert to bpompd state
    bpomdp_state = idx_to_state(state[1], state_lut)

    # Calculate the state's value
    value = 0
    for node in keys(bpomdp_state["World"])
        value -= 10*sum(bpomdp_state["World"][node])
    end

    # And return it
    return value
end 


# Function that returns the cost vector for all actions for this agent
function calc_cost_vector(node_locations, agent_id, n_actions, belief, pomdp, agent_abilities, state_lut) 
    # Determine the current state
    index = argmax_belief(pomdp, belief)

    # Convert state index to actual state
    # After this, we should have a state dictionary, as in the simulator
    state = idx_to_state(index, state_lut)

    # Calculate the cost vector 
    cost_vector = zeros(Float32, n_actions)
    
    # Actually calculate the cost
    # For non-movement actions, the cost will be the inverse of the agent's
    # abilities: if an action is desired, its cost is 0, if it is undesireable
    # then its cost is 1, and if impossible its cost is 100.
    for (i, ability) in enumerate(agent_abilities[agent_id])
        if  ability == -1
            cost_vector[i] = 100
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
    msg.T = JSON.json(observed_transitions)
    
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

    # Parse other agent's belief into the vector
    read_transitions = Dict()
    for (key, val) in JSON.parse(msg.T)
        new_key = map(x->parse(Int64,x),split(key[2:end-1], ","))
        read_transitions[new_key] = val
    end
    transitions_vector[msg.agent_id] = read_transitions
end

# Functions for converting to/from the states and indices using the LUT
function state_to_idx(state, state_lut)
    idx = 0
    for (i, s) in enumerate(state_lut)
        if s == state
            idx = i
        end
    end
    if idx == 0
        println("ERROR: somehow, we got a state not in the LUT:")
        println(state)
        println(typeof(state))
    end
    return idx
end

function idx_to_state(idx, state_lut)
    return state_lut[idx]
end

# This function adds transitions to the agent's observed transitions
function learn_from_folder(learning_folder, agent_id, state_lut)
    println("Learning from folder: $learning_folder")
    learning_files = [f for f in readdir(learning_folder) if contains(f, "sim_log.yaml")]
    n_learned = 0
    for file in learning_files
        data = YAML.load(open(joinpath(learning_folder,file)))
        trans = data["transitions"]
        last_state = state_to_idx(data["initial_state"], state_lut)
        for t in trans
            action = t["action"]
            final_state = state_to_idx(trans[1]["final_state"], state_lut)
            if final_state == 0 || last_state == 0
                println("ERROR: got an invalid state while learning from the folder!")
            end
            try
                push!(observed_transitions[last_state, action], final_state)
                n_learned += 1
            catch BoundsError
                observed_transitions[last_state, action] = []
                push!(observed_transitions[last_state, action], final_state)
                n_learned += 1
            end
            last_state = final_state
        end
        println("Loaded transitions from $file")
    end

    println("Learned $n_learned transitions from file.")

    transitions_vector[agent_id] = observed_transitions

end

# And a main function
function main(agent_id, rand_actions=false, learning_folder=nothing)
    # Initialize ROS node
    println("Initializing bPOMDP - agent_$(agent_id)")
    tic()
    init_node("agent_$(agent_id)")
    start_time = now()

    # Read configuration
    config_file = expanduser("~/catkin_ws/src/apomdp/config/common.yaml")
    state_lut_file = expanduser("~/catkin_ws/src/apomdp/config/state_lut.yaml")
    log_file = expanduser("~/catkin_ws/src/apomdp/results/$(start_time)_agent$(agent_id).yaml")
    println("Reading configuration from ", config_file)
    println("Writing agent logs to ", log_file)
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

    # Wait for the simulator to do irs thing and update the files
    println("Waiting for simulator...")
    wait_for_service("act")

    # Create the service client object
    service_client = ServiceProxy{Act}("act")

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

    # A few logging variables
    action_history = []
    entropy_history = []
    state_history = []
    reward_history = []

    # Create a pomdp object. This contains the structure of the problem,
    # and has to be used in almost all calls to apomdp.jl in order to
    # inform the functions of the problem structure
    # All agents need to build aPOMDP structs that are the same, so that
    # they inform the aPOMDP primitives in the same way and ensure
    # consistency.
    println("Creating aPOMDP object... ")
    global pomdp
    pomdp = aPOMDP(n_states, n_actions, get_v_s, state_lut)

    # Initialize this agent's belief
    beliefs_vector[agent_id] = initialize_belief(pomdp)
    transitions_vector[agent_id] = initialize_transitions(pomdp)

    

    # Learn from a folder of log files
    pre_learn_file = nothing
    if learning_folder != nothing && learning_folder == "pre_learn"
        println("Reading from pre-learning file:")
        pre_learn_file = expanduser("~/catkin_ws/src/apomdp/config/pre_learning_$agent_id.yaml")
        println(pre_learn_file)
        pre_learn = YAML.load(open(pre_learn_file))
        for (key, val) in pre_learn
            new_key = map(x->parse(Int64,x),split(key, ", "))
            observed_transitions[new_key] = val
        end
        transitions_vector[agent_id] = observed_transitions
    elseif learning_folder != nothing
        learn_from_folder(learning_folder, agent_id, state_lut)
    end

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
    service_active = true
    policy = nothing
    while ! is_shutdown() && service_active
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
        println("Fusing transitions")
        fused_T = fuse_transitions(pomdp, transitions_vector)
        pomdp.transition_matrix = fused_T

        # After integrating the new stuff, get average entropy
        append!(entropy_history, calc_average_entropy(pomdp))

        # Solve
        if !rand_actions && learning_folder == nothing && pre_learn_file == nothing || policy == nothing && (learning_folder != nothing || pre_learn_file != nothing)
            println("Calculating new policy")
            tic()
            policy = get_policy(pomdp, fused_T, c_vector, get_v_s)
            elapsed = toq()
            println("Policy calculated in $elapsed seconds.")
        else
            calculate_reward_matrix(pomdp)
        end

        # println("POLICY")
        # println(policy)
        # println()
        # println("V_S")
        # println(pomdp.state_values)
        # println()
        # println("FUSED TRANSITIONS")
        # println(fused_T)
        # println()
        # println("TRANSITIONS")
        # println(pomdp.transition_matrix)
        # println()
        # println("OBSERVED TRANSITIONS")
        # println(observed_transitions)
        # println()
        # println("KNOWN KEYS IN MAIN:")
        # println(keys(pomdp.transition_matrix))
        # println()
        # println("REWARDS")
        # println(pomdp.reward_matrix)
        # println()

        # Call fuse belief function
        println("Fusing beliefs")
        fused_b = fuse_beliefs(pomdp, beliefs_vector)

        # Decision_making (action selection)
        if rand_actions
            action = get_action(pomdp, nothing, nothing)
        else
            action = get_action(pomdp, policy, fused_b)
        end
        push!(action_history, action)

        # Act and receive an observation 
        println("Applying action $action")
        observation = nothing
        try
            observation = act(action, agent_id, service_client)
        catch
            println("Acting not successful. The simulator is shut down or mission is completed.")
            service_active = false
            continue
        end
        index = state_to_idx(observation, state_lut)
        push!(state_history, index)
        println("Got an index: ", index)

        # Get a reward as well
        push!(reward_history, get_reward(pomdp, index, action))

        # Learn
        # Update the transition observation dict
        prev_state = argmax_belief(pomdp, beliefs_vector[agent_id])
        try
            push!(observed_transitions[[prev_state, action]], index)
        catch KeyError
            observed_transitions[[prev_state, action]] = []
            push!(observed_transitions[[prev_state, action]], index)
        end
        #transitions_vector[agent_id] = learn(pomdp, beliefs_vector[agent_id], action, previous_b, transitions_vector[agent_id])
        transitions_vector[agent_id] = observed_transitions

        # Update belief
        println("Updating local belief") 
        #TODO: rethink if we should use the local transition or the fused one
        previous_b = beliefs_vector[agent_id] # save the previous belief 
        beliefs_vector[agent_id] = update_belief(pomdp, index, action, beliefs_vector[agent_id], transitions_vector[agent_id])

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

    # Write apomdp logs
    exec_time = now() - start_time
    out_file = open(log_file, "w")
    log_execution(
        out_file, 
        iter, 
        exec_time,
        pomdp.state_values,
        action_history,
        entropy_history,
        state_history,
        reward_history,
        pomdp
        )
    close(out_file)
end


# If this module is called as main
if size(ARGS)[1] == 0
    println("For now, I need to receive an int on the command line to use as agent ID.")
    exit(-1)
else
    # Parse ID
    id = parse(Int64,ARGS[1])

    # Determine if we'll do random actions
    rand_actions = false
    learning_folder = nothing
    try
        if ARGS[2] == "true"
            rand_actions = true
        else
            learning_folder = ARGS[2]
        end
    catch BoundsError
        println("WARNING: you can pass a second argument \"true\" to have the agents perform random actions.")
    end

    if id < 1
        println("Agent indices start at 1.")
    else
        main(id, rand_actions, learning_folder)
    end
end