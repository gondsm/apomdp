using POMDPs, POMDPModels, POMDPToolbox, QMDP, SARSOP, RobotOS

type aPOMDP <: POMDP{Array{Int64, 1}, Int64, Array} # POMDP{State, Action, Observation}
    n_state_vars::Int64 # Number of state variables
    n_var_states::Int64 # Number of variable states
    n_actions::Int64 # Number of possible actions
    state_values # Maintains the value of each state according to the goal. A dict of the form [S] (vector) -> V (float)
    transition_matrix::Dict # Maintains the transition probabilities in a dict of the form [S,A] (vector) -> P(S') (n-d matrix)
    reward_matrix::Dict # Maintains the rewards associated with states in a dict of the form [S,A] (vector) -> R (float)
    discount_factor::Float64
end

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

    # Initialize uniform transition matrix
    transition_dict = Dict()
    for i = 1:n_var_states, j = 1:n_var_states, k = 1:n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        key = [i,j,k]
        transition_dict[key] = ones(Float64, 3, 3)
        transition_dict[key][:] = normalize(transition_dict[key][:], 1)
    end 

    # Initialize uniform reward matrix
    reward_dict = Dict()
    for i = 1:n_var_states, j = 1:n_var_states, k = 1:n_actions
        key = [i,j,k]
        reward_dict[key] = 0
    end

    # Create and return object
    return aPOMDP(n_state_vars, 
                  n_var_states,
                  n_actions,
                  state_values_dict,
                  transition_dict, 
                  reward_dict,
                  0.9)
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
POMDPs.actions(pomdp::aPOMDP) = collect(1:pomdp.n_actions)
POMDPs.actions(pomdp::aPOMDP, state::Array) = POMDPs.actions(pomdp)

# Define observation space
POMDPs.observations(::aPOMDP) = POMDPs.states(pomdp);
POMDPs.observations(pomdp::aPOMDP, s::Array{Int64, 1}) = observations(pomdp);

# Define terminality
POMDPs.isterminal(::aPOMDP, ::Array{Int64, 1}) = false;
POMDPs.isterminal_obs(::aPOMDP, ::Array{Int64, 1}) = false;

# Define discount factor
POMDPs.discount(pomdp::aPOMDP) = pomdp.discount_factor;

# Define number of states
POMDPs.n_states(pomdp::aPOMDP) = size(POMDPs.states(pomdp))[1];

# Define number of actions
POMDPs.n_actions(pomdp::aPOMDP) = size(POMDPs.actions(pomdp))[1];

# Define number of observations
POMDPs.n_observations(pomdp::aPOMDP) = size(POMDPs.observations(pomdp))[1];

# Define transition model
function POMDPs.transition(pomdp::aPOMDP, state::Array{Int64, 1}, action::Int64)
    key = state[:]
    append!(key, action)
    return pomdp.transition_matrix[key]
end

# Define reward model
function POMDPs.reward(pomdp::aPOMDP, state::Array{Int64, 1}, action::Int64)
    key = state[:]
    append!(key, action)
    return pomdp.reward_matrix[key]
end

# Define observation model
POMDPs.observation(pomdp::aPOMDP, state::Array{Int64, 1}) = true;

# Define initial state distribution
POMDPs.initial_state_distribution(pomdp::aPOMDP) = pomdp.transition_matrix[[1,1,1]][:];


# Define state and action indices
function POMDPs.state_index(::aPOMDP, ::Array{Int64, 1})
    # TODO
    return 1
end

function POMDPs.action_index(::aPOMDP, action::Int64)
    return action
end


# Define evaluation of distributions
function pdf(dist::Array, state::Array)
    # TODO
end

pomdp = aPOMDP()

# Initialize solver
solver = QMDPSolver()

# Get a policy
policy = solve(solver, pomdp)

# Print stuff for checking:
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
