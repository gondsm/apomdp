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

# Test cases
function basic_test(re_calc_interval=0, num_iter=1000)
    # This function instantiates a random user profile and state value function in order to test the basic cases where
    # the system is allowed or not to re-calculate its policy during execution.
    print(".")
    # Initialize POMDP
    pomdp = aPOMDP()

    # Initialize solver
    solver = QMDPSolver()

    # Define the user's profile
    user_profile = random_user_profile(pomdp)

    # Define the valuable states
    random_valuable_states(pomdp)

    # Decide initial state
    state = [rand(1:pomdp.n_var_states), rand(1:pomdp.n_var_states)]

    # Get an initial policy
    policy = solve(solver, pomdp)

    # Simulation loop:
    cumulative_reward = 0.0
    reward_history = []
    for i = 1:num_iter
        # Get action
        a = action(policy, apomdpDistribution(pomdp, state))

        # Gief reward
        cumulative_reward = reward(pomdp, state, a)
        append!(reward_history, reward(pomdp, state, a))

        # Transition state
        prev_state = copy(state)
        state = user_profile[[state[1], state[2], a]]

        # Increment knowledge
        integrate_transition(pomdp, prev_state, state, a)
        calculate_reward_matrix(pomdp)

        # Re-calculate policy, if we want to
        if re_calc_interval != 0 && i % re_calc_interval == 0
            policy = solve(solver, pomdp)
        end
    end

    return reward_history
end

f = open("timeseries_data.txt", "a")

println("Starting period 0")
write(f, "# Period 0, 1000 iterations\n")
for i = 1:1000
    data = basic_test(0, 1000)
    writedlm(f, data', " ")
end

println("Starting period 1")
write(f, "# Period 1, 1000 iterations\n")
for i = 1:1000
    data = basic_test(1, 1000)
    writedlm(f, data', " ")
end

println("Starting period 5")
write(f, "# Period 5, 1000 iterations\n")
for i = 1:1000
    data = basic_test(5, 1000)
    writedlm(f, data', " ")
end

println("Starting period 20")
write(f, "# Period 20, 1000 iterations\n")
for i = 1:1000
    data = basic_test(20, 1000)
    writedlm(f, data', " ")
end

close(f)



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