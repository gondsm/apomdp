# This script contains the code necessary to run our simulated experiments.
# It is based on a simple representation of the user wherein they react to
# an action by changing their state. This change is sent to the system
include("./apomdp.jl")

# Auxiliary Functions
function random_user_profile(pomdp::aPOMDP)
    # Generates a random user profile for simulation purposes
    # TODO: limited to two state vars
    user_profile = Dict()
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states, k = 1:pomdp.n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        key = [i,j,k]
        user_profile[key] = [rand(1:pomdp.n_var_states), rand(1:pomdp.n_var_states)]
    end
    return user_profile
end

function random_valuable_states(pomdp::aPOMDP)
    # Generates random state values for simulation purposes
    # TODO: limited to two state vars
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states
        set_state_value(pomdp, [i,j], rand(1:100))
    end
    calculate_reward_matrix(pomdp)
end

function toy_example_user_profile(pomdp::aPOMDP)
    # Generates a user profile according to the toy example
    # TODO: limited to two state vars
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
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states
        set_state_value(pomdp, [i,j], 10*i)
    end
    calculate_reward_matrix(pomdp)
end

# Test cases
function basic_test(;re_calc_interval=0, num_iter=1000, out_file=-1, reward_change_interval=0, toy_example=false, solver_name="qmdp")
    # This function instantiates a random user profile and state value function in order to test the basic cases where
    # the system is allowed or not to re-calculate its policy during execution.
    # If out_file is an IOStream, this function will write its own logs to that file.
    start_time = now()
    # Initialize POMDP
    pomdp = aPOMDP()

    # Define the user's profile
    if toy_example
        user_profile = toy_example_user_profile(pomdp)
    else
        user_profile = random_user_profile(pomdp)
    end

    # Define the valuable states
    if toy_example
        toy_example_state_values(pomdp)
    else
        random_valuable_states(pomdp)
    end

    # Decide initial state
    state = [rand(1:pomdp.n_var_states), rand(1:pomdp.n_var_states)]

    # Get an initial policy
    policy = solve(pomdp, solver_name)

    # Simulation loop:
    cumulative_reward = 0.0
    reward_history = []
    state_history = [] #Matrix(0, 2)
    action_history = []
    for i = 1:num_iter
        # Get action
        a = action(policy, apomdpDistribution(pomdp, state))
        append!(action_history, a)

        # Gief reward
        cumulative_reward = reward(pomdp, state, a)
        append!(reward_history, reward(pomdp, state, a))

        # Transition state
        prev_state = copy(state)
        state = user_profile[[state[1], state[2], a]]
        append!(state_history, prev_state)
        #state_history = [state_history; state']

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
        write(out_file, "- iterations: $num_iter\n")
        write(out_file, "  reward_change_interval: $reward_change_interval\n")
        scenario = toy_example ? "toy_example" : "random"
        write(out_file, "  scenario: $scenario\n")
        exec_time = now() - start_time
        exec_time = exec_time.value
        write(out_file, "  re_calc_interval: $re_calc_interval\n")
        write(out_file, "  execution_time_ms: $exec_time\n")
        write(out_file, "  actions:\n")
        for a in action_history
            write(out_file, "  - $a\n")
        end
        write(out_file, "  states:\n")
        for i = 1:2:size(state_history)[1]
            s1 = state_history[i]
            s2 = state_history[i+1]
            write(out_file, "  - - $s1\n")
            write(out_file, "    - $s2\n")
        end
        write(out_file, "  rewards:\n")
        for r in reward_history
            write(out_file, "  - $r\n")
        end
    end

    # Return the reward history, for compatibility with previous testing code
    return reward_history
end

f1 = open("sarsop.yaml", "a")
basic_test(re_calc_interval=1, num_iter=100, out_file=f1, solver_name="qmdp")
close(f1)

# Third Batch of Test Cases
# Tests with random scenario
# f1 = open("random_scenario_short_0.yaml", "a")
# println("Starting short random scenario with re_calc = 0")
# for i = 1:1000
#     print(".")
#     basic_test(0, 100, f1, 0, false)
# end
# close(f1)
# f1 = open("random_scenario_short_1.yaml", "a")
# println()
# println("Starting short random scenario with re_calc = 1")
# for i = 1:1000
#     print(".")
#     basic_test(1, 100, f1, 0, false)
# end
# close(f1)
# f1 = open("random_scenario_short_5.yaml", "a")
# println()
# println("Starting short random scenario with re_calc = 5")
# for i = 1:1000
#     print(".")
#     basic_test(5, 100, f1, 0, false)
# end
# close(f1)
# f1 = open("random_scenario_short_20.yaml", "a")
# println()
# println("Starting short random scenario with re_calc = 20")
# for i = 1:1000
#     print(".")
#     basic_test(20, 100, f1, 0, false)
# end
# println()
# close(f1)


# Second Batch of Test Cases
# basic_test(re_calc_interval=0, num_iter=1000, out_file=-1, reward_change_interval=0, toy_example=false)

# # Tests with random scenario
# f1 = open("random_scenario.yaml", "a")
# println("Starting random scenario with re_calc = 0")
# for i = 1:1000
#     print(".")
#     basic_test(0, 1000, f1, 0, false)
# end
# println()
# println("Starting random scenario with re_calc = 1")
# for i = 1:1000
#     print(".")
#     basic_test(1, 1000, f1, 0, false)
# end
# println()
# println("Starting random scenario with re_calc = 5")
# for i = 1:1000
#     print(".")
#     basic_test(5, 1000, f1, 0, false)
# end
# println()
# println("Starting random scenario with re_calc = 20")
# for i = 1:1000
#     print(".")
#     basic_test(20, 1000, f1, 0, false)
# end
# println()
# close(f1)


# # Tests with changing values
# f2 = open("random_scenario_changing.yaml", "a")
# println("Starting random scenario with changing value with re_calc = 0")
# for i = 1:1000
#     print(".")
#     basic_test(0, 1000, f2, 200, false)
# end
# println()
# println("Starting random scenario with changing value with re_calc = 1")
# for i = 1:1000
#     print(".")
#     basic_test(1, 1000, f2, 200, false)
# end
# println()
# println("Starting random scenario with changing value with re_calc = 5")
# for i = 1:1000
#     print(".")
#     basic_test(5, 1000, f2, 200, false)
# end
# println()
# println("Starting random scenario with changing value with re_calc = 20")
# for i = 1:1000
#     print(".")
#     basic_test(20, 1000, f2, 200, false)
# end
# println()
# close(f2)

# # Tests with toy example
# f3 = open("toy_example.yaml", "a")
# println("Starting toy example with re_calc = 0")
# for i = 1:1000
#     print(".")
#     basic_test(0, 1000, f3, 0, true)
# end
# println()
# println("Starting toy example with re_calc = 1")
# for i = 1:1000
#     print(".")
#     basic_test(1, 1000, f3, 0, true)
# end
# println()
# println("Starting toy example with re_calc = 5")
# for i = 1:1000
#     print(".")
#     basic_test(5, 1000, f3, 0, true)
# end
# println()
# println("Starting toy example with re_calc = 20")
# for i = 1:1000
#     print(".")
#     basic_test(20, 1000, f3, 0, true)
# end
# println()
# close(f3)


# # First batch of tests
# f = open("timeseries_data.txt", "a")

# println("Starting period 0")
# write(f, "# Period 0, 1000 iterations\n")
# for i = 1:1000
#     data = basic_test(0, 1000)
#     writedlm(f, data', " ")
# end

# println("Starting period 1")
# write(f, "# Period 1, 1000 iterations\n")
# for i = 1:1000
#     data = basic_test(1, 1000)
#     writedlm(f, data', " ")
# end

# println("Starting period 5")
# write(f, "# Period 5, 1000 iterations\n")
# for i = 1:1000
#     data = basic_test(5, 1000)
#     writedlm(f, data', " ")
# end

# println("Starting period 20")
# write(f, "# Period 20, 1000 iterations\n")
# for i = 1:1000
#     data = basic_test(20, 1000)
#     writedlm(f, data', " ")
# end

# close(f)



# println("Starting tests with re-calc every 10 iterations")
# with_recalc_10 = []
# for i = 1:1000
#     append!(with_recalc_10, basic_test(10))
# end
# println()
# println("Starting tests with re-calc every 100 iterations")
# with_recalc_100 = []
# for i = 1:1000
#     append!(with_recalc_100, basic_test(100))
# end
# println()
# println("Starting tests with re-calc every 5 iterations")
# with_recalc_5 = []
# for i = 1:1000
#     append!(with_recalc_5, basic_test(5))
# end
# println()
# println("Starting tests with no re-calc")
# no_recalc = []
# for i = 1:1000
#     append!(no_recalc, basic_test())
# end
# println()

# println("Mean results WITH re-calc every 100 iters: ", mean(with_recalc_100))
# println("Mean results WITH re-calc every 10 iters: ", mean(with_recalc_10))
# println("Mean results WITH re-calc every 5 iters: ", mean(with_recalc_5))
# println("Mean results WITHOUT re-calc: ", mean(no_recalc))