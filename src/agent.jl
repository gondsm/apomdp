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
#TODO: do we need more services? topics to be created? 
@rosimport apomdp.srv: Act 
@rosimport apomdp.msg: shared_data
rostypegen()
using apomdp.srv
using apomdp.msg


# Global vars: given that we're using a ROS service, there's some need
# for global state. This could be solved with a singleton in an OOP,  but
# alas, this is not the case.
# In essence, we maintain the pomdp object, the current policy and a flag
# to indicate it's solving time, allowing the service callback to add data 
# to the pomdp object and not have to solve it immediatelly
#TODO: we need to change in the aPOMDP the reward function as the isvr is not really the rewarding we are going to use 
pomdp = aPOMDP("isvr", 1, [3, 3, 3], 5)  # The aPOMDP object that we'll be using throughout execution
policy = nothing        # The most up-to-date policy
solve_flag = true       # A global flag that indicates whether we want to solve the POMDP
prev_state = nothing    # The previous state the user was in
prev_action = nothing   # The previous action taken by the system


# Global vars for log keeping: These are only written to, and are used
# to generate the logs necessary for plotting.
#TODO: do we need more vars for the logs? 
num_iter = 0
v_s = Dict()
reward_history = []
state_history = []
action_history = []
entropy_history = []

# Temporary function to attribute value to states
function set_valuable_states(pomdp::aPOMDP, n_v_s=1)
    # Generates random state values for simulation purposes and returns the value function used
    v_s = Dict()
    for state in pomdp.states
        v = 10*state[1]
        set_state_value(pomdp, state, v, 1)
        v_s[state] = v
    end
    calculate_reward_matrix(pomdp)
    return v_s
end

# Callback for serving the service. It gets the action from the policy, given
# the current observation, and returns it.
# function srv_cb(req::GetActionRequest)
#     # Pack state into Julia array (could probably be better optimized)
#     state = [convert(Int64, s) for s in req.observation]

#     # Check that the input conforms to our current structure
#     if size(state)[1] != size(pomdp.state_structure)[1]
#         println("Got a wrong-size state! State: $state")
#         println("Current state structure is $(pomdp.state_structure)")
#         return false
#     else
#         invalid = [state[i] > pomdp.state_structure[i] || state[i] < 1 for i in 1:size(state)[1]]
#         for v in invalid
#             if v == true
#                 println("Got an out of bounds value! State: $state")
#                 println("Current state structure is $(pomdp.state_structure)")
#                 return false
#             end
#         end
#     end

#     # Get action
#     a = action(policy, apomdpDistribution(pomdp, state))

#     # Update logs
#     global num_iter += 1
#     append!(state_history, state)
#     append!(action_history, a)
#     append!(reward_history, reward(pomdp, state, a))
#     append!(entropy_history, calc_average_entropy(pomdp))

#     # Pack into response
#     resp = GetActionResponse()
#     resp.action = a

#     # Integrate transition
#     if prev_state != nothing && prev_action != nothing
#         println("Integrating transition!")
#         integrate_transition(pomdp, prev_state, state, prev_action)
#     end

#     # Update globals
#     global prev_state = state
#     global prev_action = a
#     global solve_flag = true

#     # Return the response
#     return resp
# end


# fuse belief function
function fuse_beliefs(beliefs_vector)

end

# fuse transistions function 
function fuse_transitions(transistions_vector)

end

# call simulation (function ?? or just from the main)
function act(action)

end

# publish to broadcast topic (function ?? or just from the main) 
function share_data(beliefs_vector, transistions_vector)

end

# subscribe to shared_data topic (function ?? or just from the main)
#call back function for shared data (belief, transistion..etc)
function share_data_cb(msg)

end

# function used to solve pomdp and return policy 
function get_policy(fused_T)

end

# this function is called for decision_making and it will return an action 
function get_action(policy,fused_b)

end 

# will update the belief and it will return beliefs_vector
function update_belief(observation,action,belief, transistion)

end 

# this function will return the transistions_vector
function learn(belief, action, previous_b) 

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
    #create publisher, and a subscriber 
    pub = Publisher{shared_data}("broadcast", queue_size=10)
    sub = Subscriber{shared_data}("shared_data/$(agent_id)", share_data_cb, queue_size=10)

    # Initialize valuable states
    #TODO

    #Initialize variables 
    #TODO: do it proberly 
    fused_T = nothing
    beliefs_vector = [nothing, nothing, nothing, nothing] #TODO: should be global because it will be updated by the callback
    transistions_vector = [nothing, nothing, nothing, nothing] #TODO: should be global because it will be updated by the callback

    # "spin" while waiting for requests
    println("Going into spin!")
    while ! is_shutdown()

        policy = get_policy(fused_T)# solve

        # Call fuse belief function
        fused_b = fuse_beliefs(beliefs_vector)
        #TODO: make beliefs_vector global variable 

        # decision_making
        action = get_action(policy,fused_b)

        #observation 
        observation = act(action)

        #to save the prvious belief 
        previous_b = beliefs_vector[agent_index]

        # update belief - on the local 
        beliefs_vector[agent_index]= update_belief(observation,action,beliefs_vector[agent_index], transistions_vector[agent_index])
        #TODO: rethink of we should use the local transition or the fused one

        # update_belief
        fused_b = fuse_beliefs(beliefs_vector)

        # learn - based on the local 
        transistions_vector[agent_index] = learn(beliefs_vector[agent_index], action, previous_b)
        #TODO: decide whether to use the fused_b or local belief 

        fused_T = fuse_transitions(transistions_vector)

        # publish something to "broadcast" topic
        share_data(beliefs_vector, transistions_vector)
        # TODO: decide when to do this  
        # TODO: find a way to avoid fusing repeated information - depends on the time and the type of info to fuse (local or fused)
        # TODO: useing of time stamp of when every agent updated 

        # Take a short break from all this
        rossleep(Duration(0.1))
    end

    println("Agent exiting!")


    


    # Write logs to file
    # out_file = open("sr_results.yaml", "a")
    # log_execution(out_file,
    #               num_iter,
    #               reward_change_interval,
    #               re_calc_interval,
    #               exec_time,
    #               v_s,
    #               action_history,
    #               entropy_history,
    #               state_history,
    #               reward_history,
    #               scenario,
    #               pomdp)
    # close(out_file)
end


# Run stuff
main()