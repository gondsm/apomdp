#!/usr/bin/env julia06

# The main goal of this script is to serve as an interface for a ROS system using aPOMDP as its decision-making
# technique.
# It should interface with ROS by receiving the observations from the underlying system and relaying them to 
# aPOMDP, control the re-calculation of the policy, and by packing and sending the policy in an appropriate
# message.

# Copyright (C) 2017 University of Coimbra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Original author and maintainer: Gonçalo S. Martins (gondsm@gmail.com)


# Imports, includes et al
include("./apomdp.jl")
using RobotOS


# Create our ROS types
@rosimport apomdp.srv: GetAction
rostypegen()
using apomdp.srv


# Global vars: given that we're using a ROS service, there's some need
# for global state. This could be solved with a singleton in an OOP,  but
# alas, this is not the case.
# In essence, we maintain the pomdp object, the current policy and a flag
# to indicate it's solving time, allowing the service callback to add data 
# to the pomdp object and not have to solve it immediatelly
pomdp = aPOMDP("isvr", 1, [3, 3, 3], 5)  # The aPOMDP object that we'll be using throughout execution
policy = nothing        # The most up-to-date policy
solve_flag = true       # A global flag that indicates whether we want to solve the POMDP
prev_state = nothing    # The previous state the user was in
prev_action = nothing   # The previous action taken by the system


# Global vars for logkeeping: These are only written to, and are used
# to generate the logs necessary for plotting.
# TODO: Fuse some of these with the global state above
num_iter = 0
v_s = Dict()
reward_history = []
state_history = []
action_history = []
entropy_history = []

# Temporary function to attribute value to states
# TODO: Replace and remove
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
function srv_cb(req::GetActionRequest)
    # Pack state into Julia array (could probably be better optimized)
    state = [convert(Int64, s) for s in req.observation]

    # Check that the input conforms to our current structure
    if size(state)[1] != size(pomdp.state_structure)[1]
        println("Got a wrong-size state! State: $state")
        println("Current state structure is $(pomdp.state_structure)")
        return false
    else
        invalid = [state[i] > pomdp.state_structure[i] || state[i] < 1 for i in 1:size(state)[1]]
        for v in invalid
            if v == true
                println("Got an out of bounds value! State: $state")
                println("Current state structure is $(pomdp.state_structure)")
                return false
            end
        end
    end

    # Get action
    a = action(policy, apomdpDistribution(pomdp, state))

    # Update logs
    global num_iter += 1
    append!(state_history, state)
    append!(action_history, a)
    append!(reward_history, reward(pomdp, state, a))
    append!(entropy_history, calc_average_entropy(pomdp))

    # Pack into response
    resp = GetActionResponse()
    resp.action = a

    # Integrate transition
    if prev_state != nothing && prev_action != nothing
        println("Integrating transition!")
        integrate_transition(pomdp, prev_state, state, prev_action)
    end

    # Update globals
    global prev_state = state
    global prev_action = a
    global solve_flag = true

    # Return the response
    return resp
end


# Simple function for updating the system's policy
function update_policy()
    print("Solving aPOMDP... ")
    calculate_reward_matrix(pomdp)
    global policy = solve(pomdp, "qmdp")
    global solve_flag = false
    println("done!")
end


# And a main function
function main()
    # Initialize ROS node
    println("Initializing aPOMDP")
    init_node("policy_calculator")
    start_time = now()

    # Create the service server object
    const srv_action = Service("apomdp/get_action", GetAction, srv_cb)

    # Initialize valuable states
    global v_s = set_valuable_states(pomdp)

    # "spin" while waiting for requests
    println("Going into spin!")
    while ! is_shutdown()
        # Should we re-calculate the policy?
        if solve_flag update_policy() end

        # Take a short break from all this
        rossleep(Duration(0.1))
    end

    println("Policy calculator exiting!")

    # Do final calculations and create final vars
    exec_time = now() - start_time
    reward_change_interval = 0
    re_calc_interval = 1
    scenario = "human_interaction"

    # Write logs to file
    out_file = open("hri_results.yaml", "a")
    log_execution(out_file,
                  num_iter,
                  reward_change_interval,
                  re_calc_interval,
                  exec_time,
                  v_s,
                  action_history,
                  entropy_history,
                  state_history,
                  reward_history,
                  scenario,
                  pomdp)
    close(out_file)
end


# Run stuff
main()