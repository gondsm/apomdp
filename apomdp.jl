# Type and method definitions for the $\alpha$POMDP.
using POMDPs, POMDPModels, POMDPToolbox, QMDP, SARSOP

type aPOMDP <: POMDP{Array{Int64, 1}, Int64, Array} # POMDP{State, Action, Observation}
    n_state_vars::Int64 # Number of state variables
    n_var_states::Int64 # Number of variable states
    n_actions::Int64 # Number of possible actions
    state_values # Maintains the value of each state according to the goal. A dict of the form [S] (vector) -> V (float)
    transition_matrix::Dict # Maintains the transition history in a dict of the form [S,A] (vector) -> P(S') (n-d matrix). It is not normalized and acts as a history of occurrences
    reward_matrix::Dict # Maintains the rewards associated with states in a dict of the form [S,A] (vector) -> R (float)
    discount_factor::Float64
    state_indices::Dict # Maintains the state indices as a dict of the form [S] (vector) -> Int
end

# Define probability distribution type
type apomdpDistribution
    state_space::Array
    dist::Array{Float64, 2}
end

# Define iterator over distribution
POMDPs.iterator(d::apomdpDistribution) = d.state_space

# Default constructor, initializes everything as uniform
function aPOMDP()
    # TODO: Only works for two state variables for now
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
                  0.9,
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
    # TODO: decide the learning factor (if summing 1 is enough or not, essentially)
    # Update the transition function/matrix with new knowledge.
    # Since the matrix is not normalized, we can treat it as a simple occurrence counter.
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
    state_space = []
    for i = 1:pomdp.n_var_states, j = 1:pomdp.n_var_states
        state_space = append!(state_space, [[i,j]])
    end
    return state_space
end

# Define action space
POMDPs.actions(pomdp::aPOMDP) = collect(1:pomdp.n_actions);
POMDPs.actions(pomdp::aPOMDP, state::Array) = POMDPs.actions(pomdp);

# Define observation space
POMDPs.observations(pomdp::aPOMDP, s::Array{Int64, 1}) = POMDPs.states(pomdp);

# Define terminality
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
    # Returning the distribution over states, as mandated
    # The distribution is first normalized, and then returned
    key = state[:]
    append!(key, action)
    #println("Called the transition function ", state, " ", action, " -> ", pomdp.transition_matrix[key])
    dist = copy(pomdp.transition_matrix[key])
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define reward model
function POMDPs.reward(pomdp::aPOMDP, state::Array{Int64, 1}, action::Int64)
    key = state[:]
    append!(key, action)
    #println("Reward called ", state, " ", action, " -> ", pomdp.reward_matrix[key])
    return pomdp.reward_matrix[key]
end

# Define observation model. Fully observed for now.
function POMDPs.observation(pomdp::aPOMDP, state::Array{Int64, 1})
    # Return certainty as to the observed state, for now
    # println("Observation function called with state ", state)
    dist = zeros(Float64, 3, 3)
    dist[state[1], state[2]] = 100
    dist[:] = normalize(dist[:], 1)
    # println(dist)
    #println("Called observation function ", dist)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define uniform initial state distribution
POMDPs.initial_state_distribution(pomdp::aPOMDP) = apomdpDistribution(POMDPs.states(pomdp), pomdp.transition_matrix[[1,1,1]]);

# Define state indices
POMDPs.state_index(pomdp::aPOMDP, state::Array{Int64, 1}) = pomdp.state_indices[state];

# Define action indices
POMDPs.action_index(::aPOMDP, action::Int64) = action

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
    #println("Number: ", r)

    # Determine in which "box" it fits
    idx = 0
    flat = dist.dist'[:]
    for i = 1:size(flat)[1]
        if r < sum(flat[1:i])          
            idx = i
            break
        end
    end
    #println("Found index: ", idx)
    #println("Called rand ", dist.dist'[:], " -> ", dist.state_space[idx])
    
    # Return the corresponding state
    return dist.state_space[idx]
end


# Test integrating transitions, rewards, etc
# pomdp = aPOMDP()
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