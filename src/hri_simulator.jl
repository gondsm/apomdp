# This script contains the code necessary to run our simulated experiments.
# It is based on a simple representation of the user wherein they react to
# an action by changing their state. This change is sent to the system
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

function toy_example_user_profile(pomdp::aPOMDP)
    # Generates a user profile according to the toy example and returns the value function used
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
    v_s = Dict()
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states
        set_state_value(pomdp, [i,j], 10*i)
        v_s[[i,j]] = 10*i
    end
    calculate_reward_matrix(pomdp)
    return v_s
end

# Test cases
function basic_test(;re_calc_interval=0, num_iter=1000, out_file=-1, reward_change_interval=0, toy_example=false, solver_name="qmdp", reward_name="svr", n_rewards=1)
    # This function instantiates a random user profile and state value function in order to test the basic cases where
    # the system is allowed or not to re-calculate its policy during execution.
    # If out_file is an IOStream, this function will write its own logs to that file.
    start_time = now()
    # Initialize POMDP
    if reward_name == "msvr"
        pomdp = aPOMDP(reward_name, n_rewards)
    else
        pomdp = aPOMDP(reward_name)
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

    # Decide initial state
    state = rand(pomdp.states)

    # Get an initial policy
    policy = solve(pomdp, solver_name)

    # Simulation loop:
    cumulative_reward = 0.0
    reward_history = []
    state_history = [] #Matrix(0, 2)
    action_history = []
    entropy_history = []
    for i = 1:num_iter
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
        # Number of iterations used
        write(out_file, "- iterations: $num_iter\n")
        # Interval of reward change
        write(out_file, "  reward_change_interval: $reward_change_interval\n")
        # Whether the toy example was run
        scenario = toy_example ? "toy_example" : "random"
        write(out_file, "  scenario: $scenario\n")
        # Policy re-calculation inteval
        write(out_file, "  re_calc_interval: $re_calc_interval\n")
        # Time it took to execute the whole scenario
        exec_time = now() - start_time
        exec_time = exec_time.value
        write(out_file, "  execution_time_ms: $exec_time\n")
        # The V(S) function used for this scenario
        write(out_file, "  v_s:\n")
        for state in keys(v_s)
            write(out_file, "    ? !!python/tuple\n")
            for i in state
                write(out_file, "    - $i\n")
            end
            write(out_file, "    : $(v_s[state])\n")
        end
        # The timeseries of action the system took
        write(out_file, "  actions:\n")
        for a in action_history
            write(out_file, "  - $a\n")
        end
        # The timeseries of average entropies
        write(out_file, "  entropies:\n")
        for h in entropy_history
            write(out_file, "  - $h\n")
        end
        # The timeseries of states the system was in
        write(out_file, "  states:\n")
        for i = 1:pomdp.n_state_vars:size(state_history)[1]
            s1 = state_history[i]
            write(out_file, "  - - $s1\n")
            for j in 1:pomdp.n_state_vars-1
                sj = state_history[i+j]
                write(out_file, "    - $sj\n")
            end
        end
        # The timeseries of the rewards obtained by the system
        write(out_file, "  rewards:\n")
        for r in reward_history
            write(out_file, "  - $r\n")
        end
    end

    # Return the reward history, for compatibility with previous testing code
    return reward_history
end

# Run a quick test
f1 = open("results/cenas.yaml", "a")
for i = 1:1
    print(".")
    basic_test(re_calc_interval=1, num_iter=10, out_file=f1, solver_name="qmdp", reward_name="msvr", n_rewards=3)
end

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


# f1 = open("cenas.yaml", "a")
# for i = 1:2
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=10, out_file=f1, solver_name="qmdp", reward_name="isvr")
# end
# println()
# close(f1)

# f1 = open("sarsop_random_1.yaml", "a")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=1000, out_file=f1, solver_name="sarsop")
# end
# close(f1)
# f1 = open("qmdp_random_1.yaml", "a")
# for i = 1:1000
#     print(".")
#     basic_test(re_calc_interval=1, num_iter=1000, out_file=f1, solver_name="qmdp")
# end
# close(f1)

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