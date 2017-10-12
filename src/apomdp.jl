# Type and method definitions for the $\alpha$POMDP.
# This script defines the aPOMDP types and associated functions that allow
# for the definition and solving of this POMDP.
# It has been tested using Julia0.6 and the QMDP solver
# POMDPs
using POMDPs, POMDPModels, POMDPToolbox

# Solvers
using QMDP, SARSOP#, DESPOT, MCVI # (MCVI is failing on load)

# Other stuff
using IterTools


# Define main type
type aPOMDP <: POMDP{Array{Int64, 1}, Int64, Array} # POMDP{State, Action, Observation}
    # Number of state variables
    n_state_vars::Int64
    # Number of variable states
    n_var_states::Int64 
    # Number of possible actions
    # Actions will be 1 through n_action
    n_actions::Int64 
    # Maintains the value of each state according to the goal. A dict of the form [S] (vector) -> V (float)
    state_values::Dict
    # Maintains the number of V(S) (state value) functions we will have in this problem
    n_v_s::Int64
    # Maintains the weights attributed to each V(S) function
    weights::Array{Float64,1}
    # Maintains the transition history in a dict of the form [S,A] (vector) -> P(S') (n-d matrix).
    # It is not normalized and acts as a history of occurrences. Normalization into a proper distribution
    # happens when it is queried via the transition() function
    transition_matrix::Dict 
    # Maintains the rewards associated with states in a dict of the form [S,A] (vector) -> R (float)
    reward_matrix::Dict 
    # The good old discount factor
    discount_factor::Float64
    # An array of all possible states
    states::Array
    # Maintains the state indices as a dict of the form [S] (vector) -> Int
    state_indices::Dict 
    # An array with the current state structure
    state_structure::Array
    # The kind of reward to be used. Can be one of svr, isvr or msvr
    reward_type::String
end

# Define probability distribution type
type apomdpDistribution
    # A list which each possible state
    state_space::Array
    # A distribution over the next state
    # This distribution is bi-dimensional, for both state dimensions
    dist::Array
end

# Define a deterministic distribution from a simple state
function apomdpDistribution(pomdp::aPOMDP, state::Array)
    dist = ones(Float64, pomdp.state_structure...)/1000
    dist[state...] = 1000
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define a uniform distribution
function apomdpDistribution(pomdp::aPOMDP)
    dist = ones(Float64, pomdp.state_structure...)/1000
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define iterator over distribution, returns the list of possible states
POMDPs.iterator(d::apomdpDistribution) = d.state_space

# Default constructor, initializes everything as uniform
function aPOMDP(reward_type::String="svr", n_v_s::Int64=1, state_structure::Array{Int64,1}=[3,3], n_actions=3, weights::Array{Float64,1}=normalize(rand(n_v_s), 1))
    # Initialize problem dimensions
    n_state_vars = size(state_structure)[1]
    n_var_states = 3

    # Generate an array with all possible states:
    vecs = [collect(1:n) for n in state_structure]
    states = collect(IterTools.product(vecs...))
    states = [[i for i in s] for s in states]

    # Initialize V-function attributing values to states
    # The inner cycle initializes V(S) as 0 for all V(S)
    # functions we want to have
    state_values_dict = Dict()

    for n = 1:n_v_s
        state_values_dict[n] = Dict()
        for state in states
            state_values_dict[n][state] = 0
        end
    end

    # Initialize state-index matrix
    curr_index = 1
    state_indices = Dict()
    for state in states
        state_indices[state] = curr_index
        curr_index += 1
    end

    # Initialize uniform transition matrix
    transition_dict = Dict()
    for state in states, k = 1:n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        key = vcat(state,[k])
        transition_dict[key] = ones(Float64, state_structure...)/1000
    end

    # Initialize uniform reward matrix
    reward_dict = Dict()
    for state in states, k = 1:n_actions
        key = vcat(state,[k])
        reward_dict[key] = 0.0
    end

    # Create and return object
    return aPOMDP(n_state_vars, 
                  n_var_states,
                  n_actions,
                  state_values_dict,
                  n_v_s,
                  weights,
                  transition_dict, 
                  reward_dict,
                  0.95,
                  states,
                  state_indices,
                  state_structure,
                  reward_type)
end

# Define reward calculation function
function calculate_reward_matrix(pomdp::aPOMDP)
    # Re-calculate the whole reward matrix according to the current transition matrix and state values
    for s in pomdp.states, k = 1:pomdp.n_actions
        key = vcat(s,[k])
        sum_var = 0
        # Get P(S'|S,A)
        dist = transition(pomdp, s, k)
        if pomdp.reward_type == "msvr"
            for f = 1:pomdp.n_v_s
                #println("Calculating MSVR for f = ", f)
                inner_sum = 0
                for state = dist.state_space
                    inner_sum += pdf(dist, state)*(pomdp.state_values[f][state]-pomdp.state_values[f][s])
                end
                sum_var += pomdp.weights[f]*inner_sum
            end
        else
            for state = dist.state_space
                sum_var += pdf(dist, state)*(pomdp.state_values[1][state]-pomdp.state_values[1][s])
            end
        end
        if pomdp.reward_type == "isvr" || pomdp.reward_type == "msvr"
            sum_var += calc_entropy(dist.dist)
        end
        pomdp.reward_matrix[key] = sum_var
    end
end

# A function for calculating the entropy of a discrete distribution
function calc_entropy(dist)
    h = 0
    for val in dist
        h += val*log(2,val)
    end
    return -h
end

# Define knowledge integration function
function integrate_transition(pomdp::aPOMDP, prev_state::Array, final_state::Array, action::Int64)
    # Update the transition function/matrix with new knowledge.
    # Since the matrix is not normalized, we can treat it as a simple occurrence counter and get away
    # with simply summing 1 to the counter.
    key = prev_state[:]
    append!(key, action)
    pomdp.transition_matrix[key][final_state...] += 1
end

# Set a state's value
function set_state_value(pomdp::aPOMDP, state::Array, value::Int64, index::Int64=1)
    # Set the value of a certain state to the given value
    # Index selects which V(S) function we're using when using MSVR
    pomdp.state_values[index][state[:]] = value
end

# Define state space
function POMDPs.states(pomdp::aPOMDP)
    # Simple iteration over all possible state combinations
    return pomdp.states
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
# All possible rewarding schemes are defined as fuctions here.
# Selection of rewarding scheme is done when constructing the aPOMDP
# struct (see aPOMDP()).
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
    dist = zeros(Float64, pomdp.state_structure...)
    dist[state...] = 100
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
POMDPs.pdf(dist::apomdpDistribution, state::Array) = dist.dist[state...]

# Discrete Belief constructor from apomdpDistribution (SARSOP)
function POMDPToolbox.DiscreteBelief(dist::apomdpDistribution)
    # Copy the values in order to a new distribution
    new_dist = []
    for state in dist.state_space
        append!(new_dist, POMDPs.pdf(dist, state))
    end

    # Construct and return new DiscreteBelief object
    discrete_b = POMDPToolbox.DiscreteBelief(new_dist)
    return discrete_b
end

# Discrete belief converter from apomdpDistribution (SARSOP), adding it to the SARSOP module so it's found
function SARSOP.convert(::Type{POMDPToolbox.DiscreteBelief}, dist::apomdpDistribution)
    return POMDPToolbox.DiscreteBelief(dist)
end

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

# Function for calculating the average entropy of the transition matrix
function calc_average_entropy(pomdp)
    # Initialize an empty vector
    entropy_vec = []
    # Iterate over state-action combinations and calculate entropies
    for state in states(pomdp)
        for action = 1:n_actions(pomdp)
            append!(entropy_vec, calc_entropy(transition(pomdp, state, action).dist))
        end
    end
    # And get the mean
    return mean(entropy_vec)
end

immutable apomdpBounds end
immutable apomdpState end
immutable apomdpObservation end    

function apomdpBounds() # DESPOT
    1,0
end

function apomdpState() # DESPOT
    [0,0]
end

function apomdpObservation() # DESPOT
    [0,0]
end


function solve(pomdp::aPOMDP, solver_name::String="")
    # Solve the POMDP according to the solver requested
    # So, apparently Julia doesn't have switch statements. Nice.
    if solver_name == "qmdp"
        solver = QMDPSolver()
        policy = POMDPs.solve(solver, pomdp)
    elseif solver_name == "sarsop"
        solver = SARSOPSolver()
        policy = POMDPs.solve(solver, pomdp, silent=true)
    elseif solver_name == "despot"
        # TODO: fix this stuff
        #solver = DESPOTSolver{apomdpState, Int64, apomdpObservation, apomdpBounds, RandomStreams}()
        #policy = POMDPs.solve(solver, pomdp)
    elseif solver_name == "mcvi"
        # TODO: build a decent constructor call
        #solver = MCVISolver(sim, nothing, 1, 10, 8, 50, 100, 500, 10, 1, 10)
        #solver = MCVISolver()
        #policy = POMDPs.solve(solver, pomdp)
    else
        println("aPOMDPs solve function received a request for an unknown solver: $solver_name")
        throw(ArgumentError)
    end

    # Get policy and return it
    
    return policy
end

#pomdp = aPOMDP("msvr", 2)
#pomdp = aPOMDP("msvr", 3)
#pomdp = aPOMDP("isvr")

# Test solvers
#policy = solve(pomdp, "despot")
#policy = solve(pomdp, "qmdp")

# Test integrating transitions, rewards, etc
# println(calc_average_entropy(pomdp))
# integrate_transition(pomdp, pomdp.states[1], pomdp.states[2], 1)
# integrate_transition(pomdp, [1,1], [1,3], 2)
# println(calc_average_entropy(pomdp))
# println(transition(pomdp, [1,1], 1))
# set_state_value(pomdp, [1,2], 10)
# set_state_value(pomdp, [1,3], 20)
# set_state_value(pomdp, [1,3], 10, 2)
# calculate_reward_matrix(pomdp)
# println(pomdp.state_values[1])
# println(pomdp.state_values[2])
# println(reward(pomdp, [1,3], 1))
# set_state_value(pomdp, [1,3], 20, 2)
# calculate_reward_matrix(pomdp)
# println(reward(pomdp, [1,3], 1))

# Test entropy function
# dist = apomdpDistribution(pomdp)
# println("Distribution: ", dist.dist)
# println("Entropy of uniform: ", calc_entropy(dist.dist))
# dist2 = apomdpDistribution(pomdp, pomdp.states[1])
# println("Distribution: ", dist2.dist)
# println("Entropy of non-uniform: ", calc_entropy(dist2.dist))


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