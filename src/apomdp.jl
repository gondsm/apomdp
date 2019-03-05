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

# Type and method definitions for the $\alpha$POMDP.
# This script defines the aPOMDP types and associated functions that allow
# for the definition and solving of this POMDP.
# It has been tested using Julia0.6 and the QMDP solver
# POMDPs
using POMDPs, POMDPModels, POMDPToolbox

# Solvers
using QMDP, SARSOP

# Other stuff
using IterTools


# Define main type
type aPOMDP <: POMDP{Array{Int64, 1}, Int64, Array} # POMDP{State, Action, Observation}
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
    # An array of all possible states---- this is the bPOMDP_states 
    states::Array
    # Maintains the state indices as a dict of the form [S] (vector) -> Int
    #state_indices::Dict 
    # An array with the current state structure
    # The state structure is of the form [i, j, k, ...]
    # i.e. first var has i states, second has j states, and so on
    state_structure::Array
    # The kind of reward to be used. Can be one of svr, isvr or msvr
    reward_type::String
    # to define agents state we need 1) number of agents 
    # TODO: Determine if these are really necessary in the aPOMDP object
    # agents_size::Int64
    # # to define agents state we need 2) specification of agents which is for now their location node 
    # agents_structure::Int64
    # # for search and rescue scenario we have topological map and it has nodes
    # # this to defines how many nodes we have  
    # nodes_num::Int64
    # # for search and rescue scenario - in order to define the world what in it, we need to know how many specifications are there  
    # world_structure::Array
    # # A matrix of ones that represents the full state space
    state_dims::Array
    # # nodes locations
    # nodes_location::Dict
    # # connectivity of nodes
    # nodes_connectivity::Dict
    # # cost vector
    # c_vector::Array
    # Corresponds to the old aPOMDP state structure, for state conversion
    expanded_structure::Array
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
    for s in state
        dist[s] = 1000
    end
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define a uniform distribution
function apomdpDistribution(pomdp::aPOMDP)
    # TODO: change this to work with the new new state structure
    dist = ones(Float64, pomdp.state_structure...)/1000
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
end

# Define iterator over distribution, returns the list of possible states
POMDPs.iterator(d::apomdpDistribution) = d.state_space

# A convenient constructor for bPOMDP agents
function aPOMDP(n_states, n_actions, get_v_s, state_lut)
    # Create object
    pomdp = aPOMDP("isvr", 1, [n_states], n_actions)

    # Set all the state values
    for s in pomdp.states
        set_state_value(pomdp, s, get_v_s(s, state_lut))
    end

    # And return it
    return pomdp
end

# Default constructor, initializes everything as uniform
function aPOMDP(reward_type::String="svr", n_v_s::Int64=1, state_structure::Array{Int64,1}=[3,3], n_actions::Int64=8, weights::Array{Float64,1}=normalize(rand(n_v_s), 1))
    # Fill in state structure
    expanded_structure = [s+1 for s in state_structure]
    state_structure = [reduce(*, state_structure)]
    state_dims = state_structure[:]

    vecs = [collect(1:n) for n in state_structure]
    states = collect(IterTools.product(vecs...))
    states = [[a for a in s] for s in states]

    # Inform
    #println("Initialized new state structure:")
    #println(state_structure)

    # Lazy initialization of everything else
    #println("Generating everything else")
    # Initialize V-function attributing values to states
    # The inner cycle initializes V(S) as 0 for all V(S)
    # functions we want to have
    state_values_dict = Dict()

    # Lazy initialization: we create only the dicts, and then
    # initialize only when needed (i.e. when adding value to a state)
    for n = 1:n_v_s
        state_values_dict[n] = Dict()
        for state in states
            state_values_dict[n][state] = 0
        end
    end

    # Initialize uniform transition matrix
    transition_dict = Dict()
    # Lazy initialization again, code kept for future reference
    # for state in states, k = 1:n_actions
    #     # For every S, A combination, we have a probability distribution indexed by 
    #     key = vcat(state,[k])
    #     # TODO: this line is taking an inordinate amount of time and rendering
    #     # the whole thing unfeasible
    #     transition_dict[key] = ones(Float64, state_structure...)/1000
    # #    println(key," ", transition_dict[key])
    # end

    # Initialize uniform reward matrix
    reward_dict = Dict()
    # And more lazy initialization
    # for state in states, k = 1:n_actions
    #     key = vcat(state,[k])
    #     reward_dict[key] = 0.0
    # end

    nodes_location = Dict()

    nodes_connectivity = Dict()

    # Create an empty cost vector
    c_vector = []

    # Create and return object
    return aPOMDP(n_actions,
                  state_values_dict,
                  n_v_s,
                  weights,
                  transition_dict, 
                  reward_dict,
                  0.95,
                  states,
                  state_structure,
                  reward_type,
                  #agents_size,
                  #agents_structure,
                  #nodes_num,
                  #world_structure,
                  state_dims,
                  #nodes_location,
                  #nodes_connectivity,
                  #c_vector),
                  expanded_structure
                  )
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
    try
        pomdp.transition_matrix[key][final_state...] += 1
    catch
        pomdp.transition_matrix[key] = ones(Float64, pomdp.state_structure...)/1000
        pomdp.transition_matrix[key][final_state...] += 1
    end
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
    dist = nothing
    try
        dist = pomdp.transition_matrix[key]
    catch ex
        dist = apomdpDistribution(pomdp)
    end
    return dist
end

function POMDPs.transition(pomdp::aPOMDP, state::Int64, action::Int64)
    # Returns the distribution over states
    # The distribution is first normalized, and then returned
    # Note: This method serves the new int-based state space representation
    key = state[state, action]
    dist = nothing
    try
        dist = copy(pomdp.transition_matrix[key])
    catch
        dist = ones(Float64, pomdp.state_structure...)/1000
    end
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
     
    r = pomdp.reward_matrix[key]

    return r
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
function POMDPs.state_index(pomdp::aPOMDP, state::Array{Int64, 1})
    # Convert
    index = state[1]

    # And done!
    return index
end

function state_from_index(pomdp::aPOMDP, index)
    state_matrix = ones(pomdp.expanded_structure...)
    state = ind2sub(state_matrix, index)
    #state = []
    return state
end

# Define action indices
POMDPs.action_index(::aPOMDP, action::Int64) = action;

# Define observation indices (SARSOP)
POMDPs.obs_index(pomdp::aPOMDP, state::Array{Int64,1}) = POMDPs.state_index(pomdp, state);

# Define distribution calculation
POMDPs.pdf(dist::apomdpDistribution, state::Array) = dist.dist[state...]

POMDPs.pdf(dist::apomdpDistribution, state::Int64) = dist.dist[state]

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


function solve(pomdp::aPOMDP, solver_name::String="")
    # Solve the POMDP according to the solver requested
    # So, apparently Julia doesn't have switch statements. Nice.
    if solver_name == "qmdp"
        solver = QMDPSolver()
        policy = POMDPs.solve(solver, pomdp)
    elseif solver_name == "sarsop"
        solver = SARSOPSolver()
        policy = POMDPs.solve(solver, pomdp, silent=true)
    else
        println("aPOMDPs solve function received a request for an unknown solver: $solver_name")
        throw(ArgumentError)
    end

    # Get policy and return it
    return policy
end


# betapomdp
function fuse_beliefs(pomdp::aPOMDP, belief_vector)
    # Determine which beliefs are defined
    idxs = [index for (index, value) in enumerate(belief_vector) if belief_vector[index] != nothing]

    # Fuse them    
    if length(idxs) > 1
        # Multiply all of the beliefs together
        fused_belief = broadcast(*,  [belief_vector[i].dist for i in idxs]...)
        # Normalize
        fused_belief[:] = normalize(fused_belief[:], 1)
        # And pack into an apomdpDistribution
        new_distribution = apomdpDistribution(pomdp)
        new_distribution.dist = fused_belief
        print("Fused beliefs from agents ")
        println(idxs)
    else
        # Copy this agent's distribution
        new_distribution = belief_vector[idxs[1]]
    end

    # And return the fused belief
    return new_distribution
end


function fuse_transitions(pomdp::aPOMDP, transition_vector)
    # Combine the observations from all agents
    observed_transitions = Dict()
    i = 0
    for (agent, trans) in transition_vector
        if trans != nothing
            for (key, state) in trans
                try
                    append!(observed_transitions[key], state)
                    i+=length(state)
                catch KeyError
                    observed_transitions[key] = []
                    append!(observed_transitions[key], state)
                    i+=length(state)
                end
            end
        end
    end
    println("Fused a total of $i transitions")

    # Convert them to a transition matrix and return it
    return observations_to_transition_matrix(pomdp, observed_transitions)
end


function observations_to_transition_matrix(pomdp::aPOMDP, observed_transitions)
    # Builds a matrix of apomdpDistribution, as necessary for policy calculation,
    # from a dict of observed transitions
    transition_matrix = Dict()
    for (key, states) in observed_transitions
        transition_matrix[key] = apomdpDistribution(pomdp, states)
    end

    return transition_matrix
end


function get_action(pomdp::aPOMDP, policy, belief)
    # Get the best action according to policy for the received belief
    if policy == nothing
        println("WARNING: get_action is returning random actions. because a bogus belief was received.")
        a = rand(1:pomdp.n_actions)
    else
        println("Getting action from policy.")
        a = SARSOP.action(policy, belief)
    end

    # And return the action
    return a
end


function initialize_belief(pomdp::aPOMDP)
    # Initializes an empty belief.
    # Essentially abstracts away this initialization from the agent, so we can
    # deal with beliefs only in this module.
    return apomdpDistribution(pomdp)

    # TODO: Use the get() function to provide a default value when running PDF!
    # https://en.wikibooks.org/wiki/Introducing_Julia/Dictionaries_and_sets#Dictionaries
end


function initialize_transitions(pomdp::aPOMDP)
    # Initializes an empty transition matrix.
    # Essentially abstracts away this initialization from the agent, so we can
    # deal with transitions only in this module.
    return Dict()
end


function argmax_belief(pomdp::aPOMDP, belief)
    # Returns the most likely state according to the belief

    # If the belief is empty, i.e. all states are equally likely, we return a
    # random state.
    state_maxval, state_idx = findmax(belief.dist,1)

    # And we're done
    return state_idx[1]
end


function update_belief(pomdp::aPOMDP, observation::Int64, action::Int64, belief, transition=Dict())
    # Use the apomdp machinery and the classical formulation to determine the
    # belief over the next state.

    # TODO
    # Also rework argument types
    println("WARNING: belief update is returning simple distributions!")

    return apomdpDistribution(pomdp, [observation])
    
    # TODO: correct this according to the equation
    # https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
end


function get_policy(pomdp::aPOMDP, fused_T, c_vector, get_v_s::Function)
    #println("WARNING: get_policy is returning bogus policies!")
    #return nothing

    # Integrate fuset_T into pomdp
    println("WARNING: get_policy is not taking cost vectors into account!")
    #pomdp.transition_matrix = fused_T

    # Iterate over all possible states to -construct (set) the V(S) in apomdp
    #calculate_reward_matrix(pomdp, c_vector, get_v_s)
    calculate_reward_matrix(pomdp)

    # Initialize a solver and solve
    solver = QMDPSolver()
    policy = POMDPs.solve(solver, pomdp)

    # Return the policy
    return policy 
end


function learn(pomdp::aPOMDP, current_belief, action, previous_belief, local_transition_matrix) 
    # pomdp contains structure
    # belief contains the final state
    # action is the action
    # previous_b is the previous belief (previous state)

    # Get the key from the previous state
    prev_state_maxval, prev_state_indx = findmax(previous_belief.dist,1)
    key = [prev_state_indx[1], action]

    # Find out which is the state we are in
    current_belief_maxval, current_belief_indx = findmax(current_belief.dist,1)
    final_state = current_belief_indx[1]
    
    # Add this transition to the matrix
    try
        local_transition_matrix[key].dist[final_state] += 1
    catch
        local_transition_matrix[key] = apomdpDistribution(pomdp)
        local_transition_matrix[key].dist[final_state] += 1
        local_transition_matrix[key].dist[:] = normalize(local_transition_matrix[key].dist[:], 1)
    end

    # Return the matrix for assignment
    return local_transition_matrix
end 

function get_reward(pomdp, index, action)
    return reward(pomdp, [index], action)
end

# Logging
function log_execution(out_file, 
                       num_iter, 
                       exec_time,
                       v_s,
                       action_history,
                       entropy_history,
                       state_history,
                       reward_history,
                       pomdp,)
    # Number of iterations used
    write(out_file, "- iterations: $num_iter\n")
    # Time it took to execute the whole scenario
    exec_time = exec_time.value
    write(out_file, "  execution_time_ms: $exec_time\n")
    # The V(S) function used for this scenario
    write(out_file, "  v_s:\n")
    #println(keys(v_s))
    for (state, val) in v_s[1]
        #write(out_file, "    ? !!python/tuple\n")
        #for i in state
        write(out_file, "    $(state[1]): $(val)\n")
        #end
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
    for i = 1:size(pomdp.state_structure)[1]:size(state_history)[1]
        s1 = state_history[i]
        write(out_file, "  - - $s1\n")
        for j in 1:size(pomdp.state_structure)[1]-1
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