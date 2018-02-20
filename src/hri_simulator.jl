# This script contains the code necessary to run our simulated experiments.
# It is based on a simple representation of the user wherein they react to
# an action by changing their state. This change is sent to the system

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

include("./apomdp.jl")

# Auxiliary Functions
function random_user_profile(pomdp::aPOMDP)
    # Generates a random user profile for simulation purposes
    user_profile = Dict()
    for state in pomdp.states, k = 1:pomdp.n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        key = vcat(state,[k])
        user_profile[key] = rand(pomdp.states)
    end
    return user_profile
end

function random_valuable_states(pomdp::aPOMDP, n_v_s=1)
    # Generates random state values for simulation purposes and returns the value function used
    v_s = Dict()
    for k = 1:n_v_s
        for state in pomdp.states
            v = rand(1:100)
            set_state_value(pomdp, state, v, k)
            if haskey(v_s, state)
                v_s[state] += v
            else
                v_s[state] = v
            end
        end
        calculate_reward_matrix(pomdp)
    end
    return v_s
end

function health_valuable_states(pomdp::aPOMDP)
    # Generates random state values for simulation purposes and returns the value function used
    # This function cannot be used with MSVR
    v_s = Dict()
    for state in pomdp.states
        v = 10*state[1] + 5*state[2]
        set_state_value(pomdp, state, v, 1)
        v_s[state] = v
    end
    calculate_reward_matrix(pomdp)
    return v_s
end

function toy_example_user_profile(pomdp::aPOMDP)
    # Generates a user profile according to the toy example and returns the value function used
    # TODO: limited to two state vars
    # TODO: toy example is currently completely broken (since the new state structure was introduced)
    user_profile = Dict()
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states, k = 1:pomdp.n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        key = [i,j,k]
        if k == 1
            user_profile[key] = [i > 1 ? i-1 : i, j < 3 ? j+1 : j]
        end
        if k == 2
            user_profile[key] = [i,j]
        end
        if k == 3
            user_profile[key] = [i < 3 ? i+1 : i, j > 1 ? j-1 : j]
        end
    end
    return user_profile
end

function toy_example_state_values(pomdp::aPOMDP)
    # Generates a reward function that is consonant with the toy example
    # TODO: limited to two state vars
    # TODO: toy example is currently completely broken (since the new state structure was introduced)
    v_s = Dict()
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states
        set_state_value(pomdp, [i,j], 10*i)
        v_s[[i,j]] = 10*i
    end
    calculate_reward_matrix(pomdp)
    return v_s
end

# Test cases
function basic_test(;re_calc_interval=0, num_iter=1000, out_file=-1, reward_change_interval=0, toy_example=false, solver_name="qmdp", reward_name="svr", n_rewards=1, state_structure=[3,3], n_actions=3)
    # This function instantiates a random user profile and state value function in order to test the basic cases where
    # the system is allowed or not to re-calculate its policy during execution.
    # If out_file is an IOStream, this function will write its own logs to that file.
    start_time = now()
    # Initialize POMDP
    if reward_name == "msvr"
        pomdp = aPOMDP(reward_name, n_rewards, state_structure, n_actions)
    else
        pomdp = aPOMDP(reward_name, 1, state_structure, n_actions)
    end

    # Define the user's profile
    if toy_example
        user_profile = toy_example_user_profile(pomdp)
    else
        user_profile = random_user_profile(pomdp)
    end

    # Define the valuable states
    if toy_example
        v_s = toy_example_state_values(pomdp)
    else
        v_s = random_valuable_states(pomdp, n_rewards)
    end
    # Get valuable states from the example (workshop paper)
    #v_s = health_valuable_states(pomdp)

    # Determine if we'll use dynamic recalculation
    dynamic_re_calc = false
    if re_calc_interval == -1
        dynamic_re_calc = true
    end

    # Decide initial state
    state = rand(pomdp.states)

    # Get an initial policy
    policy = solve(pomdp, solver_name)

    # Simulation loop:
    cumulative_reward = 0.0
    reward_history = []
    state_history = []
    action_history = []
    entropy_history = []
    for i = 1:num_iter
        print("o")
        if dynamic_re_calc == true
            re_calc_interval = ceil(i/10)
            #println(re_calc_interval)
        end
        
        # Get action
        a = action(policy, apomdpDistribution(pomdp, state))
        append!(action_history, a)

        # Gief reward
        cumulative_reward = reward(pomdp, state, a)
        append!(reward_history, reward(pomdp, state, a))

        # Transition state
        prev_state = copy(state)
        state = user_profile[[state..., a]]
        append!(state_history, prev_state)
        #state_history = [state_history; state']

        # Get current average entropy
        append!(entropy_history, calc_average_entropy(pomdp))

        # Increment knowledge
        integrate_transition(pomdp, prev_state, state, a)
        calculate_reward_matrix(pomdp)

        # Re-calculate policy, if we want to
        if re_calc_interval != 0 && i % re_calc_interval == 0
            policy = solve(pomdp, solver_name)
        end

        # Re-initialize reward function, if we want to
        if reward_change_interval != 0 && i % reward_change_interval == 0
            random_valuable_states(pomdp)
        end
    end

    # Append results to a yaml log file
    if typeof(out_file) == IOStream
        exec_time = now() - start_time
        scenario = toy_example ? "toy_example" : "random"
        log_execution(out_file, num_iter, reward_change_interval, 
                       re_calc_interval, 
                       exec_time,
                       v_s,
                       action_history,
                       entropy_history,
                       state_history,
                       reward_history,
                       scenario,
                       pomdp)
    end

    # Return the reward history, for compatibility with previous testing code
    return reward_history
end

# Run a quick test
f1 = open("results/random_qmdp_isvr_100_-1_0_100_new_struct_space_cenas.yaml", "a")
for i = 1:3
    println(i)
    tic()
    basic_test(re_calc_interval=200, num_iter=1000, out_file=f1, solver_name="qmdp", reward_name="isvr", state_structure=[3,3,3,3,3], n_actions=7)
    print("\n")
    toc()
end
println()

# New naming scheme for test results:
# condition_solver_reward_<n_iterations>_<T_c>_<T_V(S)>_<n_trials>.yaml
# f1 = open("results/random_sarsop_svr_100_1_0_1000.yaml", "a")
# println("random_sarsop_svr_100_1_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="sarsop", reward_name="svr")
# end
# close(f1)
# println()

# Below this line, there are the test cases used for the paper:

# f1 = open("results/random_sarsop_isvr_100_1_0_1000.yaml", "a")
# println("random_sarsop_isvr_100_1_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="sarsop", reward_name="isvr")
# end
# close(f1)
# println()

# f1 = open("results/random_qmdp_svr_100_1_0_1000.yaml", "a")
# println("random_qmdp_svr_100_1_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="qmdp", reward_name="svr")
# end
# close(f1)
# println()

# f1 = open("results/random_qmdp_isvr_100_1_0_1000.yaml", "a")
# println("random_qmdp_isvr_100_1_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="qmdp", reward_name="isvr")
# end
# close(f1)
# println()


# f1 = open("results/random_sarsop_svr_100_20_0_1000.yaml", "a")
# println("random_sarsop_svr_100_20_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=20, num_iter=100, out_file=f1, solver_name="sarsop", reward_name="svr")
# end
# close(f1)
# println()

# f1 = open("results/random_sarsop_isvr_100_20_0_1000.yaml", "a")
# println("random_sarsop_isvr_100_20_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=20, num_iter=100, out_file=f1, solver_name="sarsop", reward_name="isvr")
# end
# close(f1)
# println()

# f1 = open("results/random_qmdp_svr_100_20_0_1000.yaml", "a")
# println("random_qmdp_svr_100_20_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=20, num_iter=100, out_file=f1, solver_name="qmdp", reward_name="svr")
# end
# close(f1)
# println()

# f1 = open("results/random_qmdp_isvr_100_20_0_1000.yaml", "a")
# println("random_qmdp_isvr_100_20_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=20, num_iter=100, out_file=f1, solver_name="qmdp", reward_name="isvr")
# end
# close(f1)
# println()


# f1 = open("results/random_sarsop_msvr_100_20_0_1000.yaml", "a")
# println("random_sarsop_msvr_100_20_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=20, num_iter=100, out_file=f1, solver_name="sarsop", reward_name="msvr", n_rewards=3)
# end
# close(f1)
# println()

# f1 = open("results/random_sarsop_msvr_100_1_0_1000.yaml", "a")
# println("random_sarsop_msvr_100_1_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="sarsop", reward_name="msvr", n_rewards=3)
# end
# close(f1)
# println()

# f1 = open("results/random_qmdp_msvr_100_20_0_1000.yaml", "a")
# println("random_qmdp_msvr_100_20_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=20, num_iter=100, out_file=f1, solver_name="qmdp", reward_name="msvr", n_rewards=3)
# end
# close(f1)
# println()

# f1 = open("results/random_qmdp_msvr_100_1_0_1000.yaml", "a")
# println("random_qmdp_msvr_100_1_0_1000.yaml")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="qmdp", reward_name="msvr", n_rewards=3)
# end
# close(f1)
# println()