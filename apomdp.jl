# Type and method definitions for the $\alpha$POMDP.
# This script defines the aPOMDP types and associated functions that allow
# for the definition and solving of this POMDP.
# It has been tested using Julia0.6 and the QMDP solver
using POMDPs, POMDPModels, POMDPToolbox, QMDP, SARSOP

# Define main type
type aPOMDP <: POMDP{Array{Int64, 1}, Int64, Array} # POMDP{State, Action, Observation}
    # Number of state variables
    # TODO: This is still fixed at 2
    n_state_vars::Int64
    # Number of variable states
    # TODO: for now, all variables have the same number of states
    n_var_states::Int64 
    # Number of possible actions
    # Actions will be 1 through n_action
    n_actions::Int64 
    # Maintains the value of each state according to the goal. A dict of the form [S] (vector) -> V (float)
    state_values 
    # Maintains the transition history in a dict of the form [S,A] (vector) -> P(S') (n-d matrix).
    # It is not normalized and acts as a history of occurrences. Normalization into a proper distribution
    # happens when it is queried via the transition() function
    transition_matrix::Dict 
    # Maintains the rewards associated with states in a dict of the form [S,A] (vector) -> R (float)
    reward_matrix::Dict 
    # The good old discount factor
    discount_factor::Float64
    # Maintains the state indices as a dict of the form [S] (vector) -> Int
    state_indices::Dict 
end

# Define probability distribution type
type apomdpDistribution
    # A list which each possible state
    state_space::Array
    # A distribution over the next state
    # This distribution is bi-dimensional, for both state dimensions
    # TODO: still limited to 2 state variables
    dist::Array{Float64, 2}
end

# Define a deterministic distribution from a simple state
function apomdpDistribution(pomdp::aPOMDP, state::Array)
    # TODO: constans on matrix definition
    dist = ones(Float64, 3, 3)/1000
    dist[state[1], state[2]] = 1000
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define a uniform distribution
function apomdpDistribution(pomdp::aPOMDP)
    # TODO: constans on matrix definition
    println("Called uniform dist!")
    dist = ones(Float64, 3, 3)/1000
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define iterator over distribution, returns the list of possible states
POMDPs.iterator(d::apomdpDistribution) = d.state_space

# Default constructor, initializes everything as uniform
function aPOMDP()
    # TODO: Only works for two state variables for now
    # TODO: constants on matrix definitions
    # (fors are repeated along n_var_states twice only, will have to
    # be expanded to work on n variables with iterators or something)
    # Initialize problem dimensions
    n_state_vars = 2
    n_var_states = 3
    n_actions = 3

    # Initialize V-function attributing values to states
    state_values_dict = Dict()
    for i = 1:n_var_states
        for j = 1:n_var_states
            key = [i,j]
            state_values_dict[key] = 0
        end
    end

    # Initialize state-index matrix
    curr_index = 1
    state_indices = Dict()
    for i = 1:n_var_states, j = 1:n_var_states
        state_indices[[i,j]] = curr_index
        curr_index += 1
    end

    # Initialize uniform transition matrix
    transition_dict = Dict()
    for i = 1:n_var_states, j = 1:n_var_states, k = 1:n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        key = [i,j,k]
        transition_dict[key] = ones(Float64, 3, 3)/1000
    end 

    # Initialize uniform reward matrix
    reward_dict = Dict()
    for i = 1:n_var_states, j = 1:n_var_states, k = 1:n_actions
        key = [i,j,k]
        reward_dict[key] = 0.0
    end

    # Create and return object
    return aPOMDP(n_state_vars, 
                  n_var_states,
                  n_actions,
                  state_values_dict,
                  transition_dict, 
                  reward_dict,
                  0.95,
                  state_indices)
end

# Define reward calculation function
function calculate_reward_matrix(pomdp::aPOMDP)
    # TODO: Only works for two state variables
    # TODO: Integrate information metric
    # Re-calculate the whole reward matrix according to the current transition matrix and state values
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states, k = 1:pomdp.n_actions
        key = [i,j,k]
        sum_var = 0
        # Get P(S'|S,A)
        dist = transition(pomdp, [i,j], k)
        for state = dist.state_space
            sum_var += pdf(dist, state)*(pomdp.state_values[state]-pomdp.state_values[[i,j]])
        end
        pomdp.reward_matrix[key] = sum_var
    end
end

# Define knowledge integration function
function integrate_transition(pomdp::aPOMDP, prev_state::Array, final_state::Array, action::Int64)
    # Update the transition function/matrix with new knowledge.
    # Since the matrix is not normalized, we can treat it as a simple occurrence counter and get away
    # with simply summing 1 to the counter.
    key = prev_state[:]
    append!(key, action)
    pomdp.transition_matrix[key][final_state[1], final_state[2]] += 1
end

# Set a state's value
function set_state_value(pomdp::aPOMDP, state::Array, value::Int64)
    # Set the value of a certain state to the given value
    pomdp.state_values[state[:]] = value
end

# Define state space
function POMDPs.states(pomdp::aPOMDP)
    # Simple iteration over all possible state combinations
    # TODO: still limiter to 2 state variables
    state_space = []
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states
        state_space = append!(state_space, [[i,j]])
    end
    return state_space
end

# Define action space
POMDPs.actions(pomdp::aPOMDP) = collect(1:pomdp.n_actions);
# Convenience: actions are the same regardless of state
POMDPs.actions(pomdp::aPOMDP, state::Array) = POMDPs.actions(pomdp);

# Define observation space
POMDPs.observations(pomdp::aPOMDP, s::Array{Int64, 1}) = POMDPs.observations(pomdp); #QMDP
POMDPs.observations(pomdp::aPOMDP) = POMDPs.states(pomdp); #SARSOP

# Define terminality (no terminal states exist)
POMDPs.isterminal(::aPOMDP, ::Array{Int64, 1}) = false;
POMDPs.isterminal_obs(::aPOMDP, ::Array{Int64, 1}) = false;

# Define discount factor
POMDPs.discount(pomdp::aPOMDP) = pomdp.discount_factor;

# Define number of states
POMDPs.n_states(pomdp::aPOMDP) = size(POMDPs.states(pomdp))[1];

# Define number of actions
POMDPs.n_actions(pomdp::aPOMDP) = size(POMDPs.actions(pomdp))[1]

# Define number of observations
POMDPs.n_observations(pomdp::aPOMDP) = size(POMDPs.observations(pomdp))[1]

# Define transition model
function POMDPs.transition(pomdp::aPOMDP, state::Array{Int64, 1}, action::Int64)
    # Returns the distribution over states
    # The distribution is first normalized, and then returned
    # TODO: probably also limited to 2 state variables (indirectly)
    key = state[:]
    append!(key, action)
    dist = copy(pomdp.transition_matrix[key])
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define reward model
function POMDPs.reward(pomdp::aPOMDP, state::Array{Int64, 1}, action::Int64)
    # Get the corresponding reward from the reward matrix
    # TODO: also likely limited to 2 state vars
    key = state[:]
    append!(key, action)
    return pomdp.reward_matrix[key]
end

# Define observation model. Fully observed for now.
function POMDPs.observation(pomdp::aPOMDP, state::Array{Int64, 1})
    # Return a distribution over possible states given the observation
    # TODO: make partially observed
    # TODO: also likely limited to 2 state vars
    # TODO: constants on distribution definition
    dist = zeros(Float64, 3, 3)
    dist[state[1], state[2]] = 100
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define uniform initial state distribution (SARSOP)
POMDPs.initial_state_distribution(pomdp::aPOMDP) = apomdpDistribution(pomdp);

# Define state indices
POMDPs.state_index(pomdp::aPOMDP, state::Array{Int64, 1}) = pomdp.state_indices[state];

# Define action indices
POMDPs.action_index(::aPOMDP, action::Int64) = action;

# Define observation indices (SARSOP)
POMDPs.obs_index(pomdp::aPOMDP, state::Array{Int64,1}) = POMDPs.state_index(pomdp, state);

# Define distribution calculation
POMDPs.pdf(dist::apomdpDistribution, state::Array) = dist.dist[state[1], state[2]]

# Define sampling function to sample a state from the transition probability
function POMDPs.rand(rng::AbstractRNG, dist::apomdpDistribution)
    # Sample from the distribution
    # This is done by flattening the distribution into a row-major vector, which we use as 
    # a histogram defining a distribution. This means that the flattened distribution
    # corresponds 1:1 with the states in the state_space vector. (perhaps I could even change
    # this system-wide...)
    # BE CAREFUL: when manipulating the transition matrix, this will have to be kept consistent!
    # Get a random number
    r = rand(rng)

    # Determine in which "box" it fits
    idx = 0
    flat = dist.dist'[:]
    for i = 1:size(flat)[1]
        if r < sum(flat[1:i])          
            idx = i
            break
        end
    end
    
    # Return the corresponding state
    return dist.state_space[idx]
end

#pomdp = aPOMDP()

# Test solver
#solver = QMDPSolver()
#solver = SARSOPSolver()
#policy = solve(solver, pomdp)

# Test integrating transitions, rewards, etc
#integrate_transition(pomdp::aPOMDP, prev_state::Array, final_state::Array, action::Int64)
# integrate_transition(pomdp, [1,1], [1,2], 1)
# integrate_transition(pomdp, [1,1], [1,3], 2)
# println(transition(pomdp, [1,1], 1))
# set_state_value(pomdp, [1,2], 10)
# set_state_value(pomdp, [1,3], 20)
# calculate_reward_matrix(pomdp)


# # Test whether distributions behave the way we want them to
# dist = ones(Float64, 3, 3)
# println(dist)
# #dist[state[1], state[2]] = 100
# dist[:] = normalize(dist[:], 1)
# println(dist)

# # Test the rand function
# pomdp = aPOMDP()
# dist = [1. 1. 1.; 1. 1. 1.; 1. 1. 1.]
# dist[:] = normalize(dist[:], 1)
# object = apomdpDistribution(POMDPs.states(pomdp), dist)
# rng = RandomDevice() 
# for i = 1:10
#     rand(rng, object)
# end

# # Print stuff for checking:
# println("State space:")
# println(POMDPs.states(pomdp))
# println("Action space:")
# println(POMDPs.actions(pomdp))
# println("Observation space:")
# println(POMDPs.observations(pomdp))
# println("Number of states:")
# println(POMDPs.n_states(pomdp))
# println("Number of actions:")
# println(POMDPs.n_actions(pomdp))
# println("Number of observations:")
# println(POMDPs.n_observations(pomdp))