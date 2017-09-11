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
    # TODO: limiter to two state vars
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
        #user_profile[key] = [rand(1:pomdp.n_var_states), rand(1:pomdp.n_var_states)]
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
function basic_test(re_calc_interval=0, num_iter=1000, out_file=-1, reward_change_interval=0, toy_example=false)
    # This function instantiates a random user profile and state value function in order to test the basic cases where
    # the system is allowed or not to re-calculate its policy during execution.
    # If out_file is an IOStream, this function will write its own logs to that file.
    print(".")
    # Initialize POMDP
    pomdp = aPOMDP()

    # Initialize solver
    solver = QMDPSolver()

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
    policy = solve(solver, pomdp)

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
            policy = solve(solver, pomdp)
        end

        # Re-initialize reward function, if we want to
        if reward_change_interval != 0 && i % reward_change_interval == 0
            random_valuable_states(pomdp)
        end
    end

    # Append results to a yaml log file
    if typeof(out_file) == IOStream
        write(f, "- iterations: $num_iter\n")
        write(f, "  reward_change_interval: $reward_change_interval\n")
        scenario = toy_example ? "toy_example" : "random"
        write(f, "  scenario: $scenario\n")
        write(f, "  re_calc_interval: $re_calc_interval\n")
        write(f, "  actions:\n")
        for a in action_history
            write(f, "  - $a\n")
        end
        write(f, "  states:\n")
        for i = 1:2:size(state_history)[1]
            s1 = state_history[i]
            s2 = state_history[i+1]
            write(f, "  - - $s1\n")
            write(f, "    - $s2\n")
        end
        write(f, "  rewards:\n")
        for r in reward_history
            write(f, "  - $r\n")
        end
    end

    # Return the reward history, for compatibility with previous testing code
    return reward_history
end

f = open("test.txt", "a")
basic_test(1, 30, f, 10, true)
close(f)
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