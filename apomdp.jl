using POMDPs, POMDPModels, POMDPToolbox, QMDP, SARSOP, RobotOS

type aPOMDP <: POMDP{Array, Int32, Array} # POMDP{State, Action, Observation}
    n_state_vars::Int32 # Number of state variables
    n_var_states::Int32 # Number of variable states
    n_actions::Int32 # Number of possible actions
    state_values # Maintains the value of each state according to the goal. A dict of the form [S] (vector) -> V (float)
    transition_matrix::Dict # Maintains the transition probabilities in a dict of the form [S,A] (vector) -> P(S') (n-d matrix)
    reward_matrix::Dict # Maintains the rewards associated with states in a dict of the form [S,A] (vector) -> R (float)
end

# Default constructor, initializes everything as uniform
function aPOMDP()
    # Only works for two state variables for now
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
            state_values_dict[i,j] = 0
        end
    end

    # Initialize uniform transition matrix
    transition_dict = Dict()
    for i = 1:n_var_states, j = 1:n_var_states, k = 1:n_actions
        # For every S, A combination, we have a probability distribution indexed by 
        transition_dict[i,j,k] = ones(Float64, 3, 3)
        transition_dict[i,j,k][:] = normalize(transition_dict[i,j,k][:], 1)
    end 

    # Initialize uniform reward matrix
    reward_dict = Dict()
    for i = 1:n_var_states, j = 1:n_var_states, k = 1:n_actions
        reward_dict[i,j,k] = 0
    end

    # Create and return object
    return aPOMDP(n_state_vars, 
                  n_var_states,
                  n_actions,
                  state_values_dict,
                  transition_dict, 
                  reward_dict)
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
POMDPs.observations(pomdp::aPOMDP, s::Bool) = observations(pomdp);

pomdp = aPOMDP()

# Print stuff for checking:
println("State space:")
println(POMDPs.states(pomdp))
println("Action space:")
println(POMDPs.actions(pomdp))
println("Observation space:")
println(POMDPs.observations(pomdp))

#println(pomdp.transition_matrix)
#println(pomdp.transition_matrix[1,1,1][1,:])
#println(pomdp.transition_matrix[1,1,1])
#println(pomdp.reward_matrix[[1,1,1]])
