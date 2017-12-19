# julia version used 

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

#copyright 

# License 

# Imports, includes et al
include("./apomdp.jl")
using RobotOS

# Create our ROS types
@rosimport apomdp.srv: Act 
@rosimport apomdp.msg: shared_data
rostypegen()
using apomdp.srv
using apomdp.msg


# Global vars


# Agent-specific actions
#this function will return the v(s)
function get_v_s(state)
    #TODO: v(s) = -2n_victim * -n_depries * n_fire
end 

# this function will retun cost vector for all actions for this agent
#
function calc_cost_vector(belief, ability_vector) 

end 
######


# APOMDP-dependent functions (maths)
# fuse belief function
function fuse_beliefs(beliefs_vector)

end

# fuse transistions function 
function fuse_transitions(transitions_vector)

end

# function used to solve pomdp and return policy 
function get_policy(fused_T, c_vector)
    #get the v_s
    #v = get_v_s(state)
    #TODO: this function will return the value of state v(s) 
end

# this function is called for decision_making and it will return an action 
function get_action(policy,fused_b)
    # Get action
    #a = action(policy, apomdpDistribution(pomdp, state))
end 

# will update the belief and it will return beliefs_vector
function update_belief(observation,action,belief, transistion)

end 

# this function will return the transitions_vector
function learn(belief, action, previous_b) 
    # integrate_transition(pomdp, prev_state, state, prev_action)
end 
######


# ROS-related functions
# call simulation (function ?? or just from the main)
function act(action)

end

# publish to broadcast topic (function ?? or just from the main) 
function share_data(beliefs_vector, transitions_vector)

end

# subscribe to shared_data topic (function ?? or just from the main)
#call back function for shared data (belief, transistion..etc)
function share_data_cb(msg)

end


# And a main function
function main()
    #TODO: get the agent_id from somewhere
    agent_id = 0
    agent_index = agent_id + 1

    # Initialize ROS node
    println("Initializing bPOMDP")
    init_node("agent_$(agent_id)")
    start_time = now()

    # Create the service client object
    service_client = ServiceProxy{Act}("act")

    # Subscribe to the agent's shared_data topic
    # Create publisher, and a subscriber 
    pub = Publisher{shared_data}("broadcast", queue_size=10)
    sub = Subscriber{shared_data}("shared_data/$(agent_id)", share_data_cb, queue_size=10)

    # Initialize variables 
    # TODO: do it properly 
    fused_T = nothing
    beliefs_vector = [nothing, nothing, nothing, nothing] #TODO: should be global because it will be updated by the callback
    transitions_vector = [nothing, nothing, nothing, nothing] #TODO: should be global because it will be updated by the callback
    ability_vector = [nothing, nothing, nothing, nothing] #TODO: read this from configuration (agent.yaml) 

    # "spin" while waiting for requests
    println("Going into spin!")
    while ! is_shutdown()
        # Get the cost 
        c_vector = calc_cost_vector(beliefs_vector[agent_index], ability_vector)

        # Solve
        policy = get_policy(fused_T, c_vector)

        # Call fuse belief function
        fused_b = fuse_beliefs(beliefs_vector)
        #TODO: make beliefs_vector global variable 

        # Decision_making (action selection)
        action = get_action(policy,fused_b)

        # Act and receive an observation 
        observation = act(action)

        # Update belief - on the local 
        previous_b = beliefs_vector[agent_index] # save the previous belief 
        beliefs_vector[agent_index]= update_belief(observation,action,beliefs_vector[agent_index], transitions_vector[agent_index])
        #TODO: rethink of we should use the local transition or the fused one

        # Fuse beliefs
        fused_b = fuse_beliefs(beliefs_vector)

        # Learn - based on the local 
        transitions_vector[agent_index] = learn(beliefs_vector[agent_index], action, previous_b)
        # TODO: decide whether to use the fused_b or local belief 

        # Fuse new transition with everyone else's
        fused_T = fuse_transitions(transitions_vector)

        # Publish something to "broadcast" topic
        share_data(beliefs_vector, transitions_vector)
        # TODO: decide when to do this  
        # TODO: find a way to avoid fusing repeated information - depends on the time and the type of info to fuse (local or fused)
        # TODO: useing of time stamp of when every agent updated 

        # Take a short break from all this
        rossleep(Duration(0.1))
    end

    println("Agent exiting!")

    # TODO: Write logs
end


# Run stuff
main()