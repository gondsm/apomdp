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
using QMDP, SARSOP#, DESPOT, MCVI # (MCVI is failing on load)

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
    agents_size::Int64
    # to define agents state we need 2) specification of agents which is for now their location node 
    agents_structure::Int64
    # for search and rescue scenario we have topological map and it has nodes
    # this to defines how many nodes we have  
    nodes_num::Int64
    # for search and rescue scenario - in order to define the world what in it, we need to know how many specifications are there  
    world_structure::Array
    # A matrix of ones that represents the full state space
    state_dims::Array
    # nodes locations
    nodes_location::Dict
    # connectivity of nodes
    nodes_connectivity::Dict
    # cost vector
    c_vector::Array
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
    # TODO: change this to work with the new new state structure
    dist = ones(Float64, pomdp.state_structure...)/1000
    dist[state...] = 1000
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

# Default constructor, initializes everything as uniform
function aPOMDP(reward_type::String="svr", n_v_s::Int64=1, state_structure::Array{Int64,1}=[3,3], n_actions::Int64=8, weights::Array{Float64,1}=normalize(rand(n_v_s), 1), agents_size::Int64=0, agents_structure::Int64=0, nodes_num::Int64=0, world_structure::Array{Int64,1}=[],nodes_location=Dict(), nodes_connectivity=Dict())
    # Generate aPOMDP state structure from bPOMDP structure
    # TODO: Make this compatible with aPOMDP again.
    state_structure = convert_structure(agents_size, nodes_num, agents_structure, world_structure)
    #println("state_structure: ", state_structure)

    #println("Generating states")
    states = collect(1:reduce(*, state_structure))
    #println(size(states))
    state_dims = ones(state_structure...)

    state_structure = [states[end]]

    println("Initialized new state structure:")
    println(state_structure)

    #println("Generating everything else")
    # Initialize V-function attributing values to states
    # The inner cycle initializes V(S) as 0 for all V(S)
    # functions we want to have
    state_values_dict = Dict()

    # Lazy initialization: we create only the dicts, and then
    # initialize only when needed (i.e. when adding value to a state)
    for n = 1:n_v_s
        state_values_dict[n] = Dict()
    #     for state in states
    #         state_values_dict[n][state] = 0
    #     end
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
                  #state_indices,
                  state_structure,
                  reward_type,
                  agents_size,
                  agents_structure,
                  nodes_num,
                  world_structure,
                  state_dims,
                  nodes_location,
                  nodes_connectivity,
                  c_vector)
end

# Define reward calculation function
function calculate_reward_matrix(pomdp::aPOMDP)
    # Re-calculate the whole reward matrix according to the current transition matrix and state values
    for s in pomdp.states, k = 1:pomdp.n_actions
        tic()
        key = vcat(s,[k])
        println("Starting new key")
        print(key)
        sum_var = 0
        # Get P(S'|S,A)
        dist = transition(pomdp, s, k)
        if pomdp.reward_type == "msvr"
            for f = 1:pomdp.n_v_s
                inner_sum = 0
                for state = dist.state_space
                    v_s_1 = try
                        pomdp.state_values[f][state]
                    catch
                        0
                    end
                    v_s_2 = try 
                        pomdp.state_values[f][s]
                    catch
                        0
                    end
                    inner_sum += pdf(dist, state)*(v_s_1-v_s_2)
                end
                sum_var += pomdp.weights[f]*inner_sum
            end
        else
            for state = dist.state_space
                v_s_1 = try
                    pomdp.state_values[1][state]
                catch
                    0
                end
                v_s_2 = try 
                    pomdp.state_values[1][s]
                catch
                    0
                end
                sum_var += pdf(dist, state)*(v_s_1-v_s_2)
            end
        end
        if pomdp.reward_type == "isvr" || pomdp.reward_type == "msvr"
            sum_var += calc_entropy(dist.dist)
        end

        # Add the cost relative to the c_vector
        try
            sum_var += c_vector[k]
        catch
            println("Tried to add action cost to reward, but c_vector seems to be emtpy!")
        end

        if sum_var != 0
            pomdp.reward_matrix[key] = sum_var
        end
    end
    toc()
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
        dist = copy(pomdp.transition_matrix[key])
    catch
        dist = ones(Float64, pomdp.state_structure...)/1000
    end
    dist[:] = normalize(dist[:], 1)
    return apomdpDistribution(POMDPs.states(pomdp), dist)
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
    r = try
        pomdp.reward_matrix[key]
    catch
        0
    end
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
    # = pomdp.state_indices[state];
    #println("state_structure: ",pomdp.state_structure)
    state_dims = ones(pomdp.state_structure...)
    #println("state_dims: ",state_dims)
    index = sub2ind(size(state_dims), state...)
    return index
end

function state_from_index(pomdp::aPOMDP, index)
    state =  ind2sub(pomdp.state_dims, index)
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


# betapomdp
function convert_structure(agents_size::Int64, nodes_num::Int64, agents_structure::Int64, world_structure::Array{Int64,1})
    
    # Create array
    apomdp_structure=Array{Int64}((length(agents_structure)*agents_size) + (length(world_structure)*nodes_num))

    #counter is index for the apomdp_structure
    counter = 1 

    # Agents
    for i in 1:agents_size
        for j in 1:length(agents_structure)
            apomdp_structure[counter]=agents_structure[j]
            counter+=1
        end
    end

    # Nodes
    for i in 1:nodes_num
        for j in 1:length(world_structure)
            apomdp_structure[counter]=world_structure[j]
            counter+=1
        end
    end
    
    # And return it
    return apomdp_structure
end 


function fuse_beliefs(pomdp::aPOMDP, belief_vector)
    # Steps
    # First we check if there is a need to fuse beliefs 
    # Create the vector of fusion from the beliefs_vector which will only contains belief for fusion --> the clean vector
    # Pass a clean vector(a vector without nothing, a vector that can be used)
    # We should have the fused_b to be returned

    # Warning: the belief_vector may contain "nothings"

    # Fuse the vector of beliefs that is received as input
    # Each element of the vector is itself a vector in floats that encode
    # a probability distributions.
    # Fusion takes place via:
    # bf = b1*b2*...*bn
    # TODO: look up the theory on this

    #println(pomdp.states)#will get indcies of states 
    
    #create a fake vector of two beliefs , each vector with length of states size 
    #beliefs_vec = ones(2,length(pomdp.states))
    #fused_belief = zeros(length(pomdp.states))

    #if not fake should be like this
    #beliefs_vec = belief_vector
    #=fused_belief = zeros(size(belief_vector,2))
    println("size(belief_vector,1): ", size(belief_vector,2))

    for x=1:size(belief_vector,2)     
        println("belief_vector: ",belief_vector)
        println("fused_belief: ",fused_belief)
        fused_belief = fused_belief + belief_vector[x]
    end
    println("not-normalized fused_belief: ", fused_belief)
    #normalize 
    fused_belief[:] = normalize(fused_belief[:], 1)

    println("normalized fused_belief: ", fused_belief)=#

    b = Any[]
    for i=1:size(belief_vector,1)
        #println("belief_vector[i]:", belief_vector[i])
        if i==1
            b=belief_vector[i]
        else
            b+=belief_vector[i]
        end 
    #println("b", b)
    end 
    #normalize 
    #
    b[:] = normalize(b[:], 1)
    #println("normalized b: ", b)
   return b
    
end


function fuse_transitions(pomdp::aPOMDP, transition_vector)
    # Steps
    # First we check if there is a need to fuse transistions 
    # Create the vector of fusion from the transitions_vector which will only contains transitions for fusion --> the clean vector
    # Pass a clean vector(a vector without nothing, a vector that can be used)
    # We should have the fused_T to be returned

    # Warning: the vector may contain "nothings"

    # Fuse the vector of transition matrices
    # Each element of the vector is a full transition matrix
    # for the whole state space (both s and s') and action space
    # Fusion is done in the same principle as beliefs, but iterating
    # over all possible combinations of s', s and a.
    # Use state indices to iterate?

    #=
    transition_vector[agent][state, action] -> distributions
    distributions -> fuse_belief -> fused transition for current key (state, action)
    transition_vector
    [
        transitions(s,a) 
        [
            1D prob distributions
        ]
    ]
    T(s,a) -> dist
    =#
    
    fused_transitions = Dict()
    #for testing 
    #states_n = 2
    #actions_n = 2

    for s in pomdp.states
        # [1, 2, 3, 4, ..., n_actions]
        for a in 1:pomdp.n_actions
            key = [s, a]
            distributions = [transition[key] for transition in transition_vector]
            fused_transitions[key] = fuse_beliefs(pomdp,distributions)
        end
    end
    return fused_transitions
end


function set_transition_matrix(pomdp::aPOMDP, transition_matrix)
    # Set the transition matrix of the current system as the one received
end


function set_c_vector(pomdp::aPOMDP, c_vector)
    # Set the current c vector as the one received
    # TODO: 
    # extend the apomdp to have a c_vector (all ones by default)
    # extend the reward (MSVR by defaul, uniform thetas by default).
    # SVR corresponds to having theta 1 just for that term, and so on
end


function get_action(pomdp::aPOMDP, policy, belief)
    # Get the best action according to policy for the received belief
    # The belief is assumed as a vector of floars over the state space,
    # which will have to be converted into a apomdpDistribution to
    # plug into the policy using: 
    # a = action(policy, apomdpDistribution(pomdp, belief))

    # TODO: actually get from policy!
    # The return value should be integer
    return 1
end


function update_belief(pomdp::aPOMDP, observation, action, belief, transition)
    # Steps: 
    # Will pass observation, action, belief and transition to a function in apomdp.update_belief
    # It will return the updated belief 

    # Use the apomdp machinery and the classical formulation to determine the
    # belief over the next state.
    # Observation is a dict, action is an int, belief is a vector over state
    # indices as usual, transition is a transition matrix as defined before.

    # b(s)
    # [P(s=1), P(s=2), P(s=3), ..]
    #
    # b'(s')
    # [b'(s'=1), b'(s'=2), ...]

    # 3 cells
    # O-------O-------O
    # 1       2       3
    #
    # State 1:
    # agent_1 is in cell 1
    # fire is in cell 2
    #
    # State 2:
    # agent_1 is in cell 1
    # fire is in cell 3
    #
    # State 3:
    # agent_1 is in cell 2
    # fire is in cell 3
    #
    # State 2 is closer to State 1 than State 3!
    # Distance(1->2) < Distance(1->3)
    # Function D

    # o = 1
    # P(o=1|s', a=1)
    # s' = 1 -> P = 0.6
    # s' = 2 -> P = 0.3
    # s' = 3 -> P = 0.1
    #
    # Because:
    # D(1,2) > D(1,3) and o = 1 (i.e. observation says we are in state 1)

    # Concepts:
    # -> Some states are closer to one another than others
    # -> We can define a real-valued function for distance between states
    # -> O(o|s',a) could be defined as a function of distance between states
    #    (closer states should have higher probability)

    # Tentative formulation
    #
    # O(o|s', a) = normalize D(o,s')
    #
    # -> Y axis: D(o=1, s')
    # -> X axis: s'
    # This can be a distribution over s' if we normalize!

    # State s is an in (e.g 1)
    # But it translates into an instantiation of state space:
    #
    # [3,2,4,3,1]
    #
    # So, D could be just the norm of the difference vector!
    
    # Better example
    # 4 states: [1,1], [1,2], [2,1], [2,2]
    #           1      2      3      4
    #
    # We get o = 2
    # Distances (from 2 to x):
    #
    # [1.0, 0.0, 1.41, 1.0]
    #
    # Subtract by maximum and take absolute:
    #
    # [0.41, 1.41, 0.0, 0.41]
    #
    # Normalize:
    #
    # [0.18, 0.63, 0.0, 0.18] -> Possible definition of O(o=2|s', a)
    #
    # If we want O(o=2, s'=1, a) -> we get 0.18


    # Implementation
    # -> Convert from int state to vector
    # -> Calculate the difference vector: (C = A - B)
    # -> Calculate the norm: (norm(C))

    updated_b = Any[]
    #equation of the belief update_belief
    #=
        b(s)=norm(O(o|s,a)sum(T(s'|s,a))b(s)))
    =#
    sum_t =0.0
    for s in pomdp.states
           key = [s,action]
           println("key:",key)
           println("transition[key]:",transition[key])
           t=transition[key]
           println("b",b)
           println("a",a)
           println("b[a]:",t[action])
           sum_t += t[action]*belief[s]
           update_belief[s] = observation[s]*sum_t
    end

    #normalize
    update_belief[:] = normalize(update_belief[:], 1)

    # Return empty bogus array. Final type must match shared_data.msg
    return Float32[]
end


function get_policy(pomdp::aPOMDP, fused_T, c_vector)
    #get the v_s
    #v = get_v_s(state)
    #TODO: this function will return the value of state v(s) 
    println("calling get_policy")
    # Integrate fuset_T into pomdp
    pomdp.transition_matrix = fused_T
    # c_vector
    pomdp.c_vector = c_vector
    # Iterate over all possible states to -construct (set) the V(S) in apomdp
    calculate_reward_matrix(pomdp)

    # Initialize a solver and solve
    solver = QMDPSolver()
    policy = POMDPs.solve(solver, pomdp)

    # Return the policy
    return policy 
end


function learn(pomdp::aPOMDP, current_belief, action, previous_belief, local_transition_matrix) 
    # integrate_transition(pomdp, prev_state, state, prev_action) 
    # pomdp contains structure
    # belief contains the final state
    # action is the action
    # previous_b is the previous belief (previous state)

    # Get the key from the previous state
    # key = prev_state[:]
    # Key will be result of argmaxing the belif and appending action
    prev_state_maxval, prev_state_indx = findmax(previous_belief,2)
    key = [prev_state_indx[1], action]
    
    # Find out which is the state we are in
    current_belief_maxval, current_belief_indx = findmax(current_belief,2)
    final_state = current_belief_indx[1]
    
    # Add this transition to the matrix
    try
        local_transition_matrix[key][final_state] += 1
    catch
        local_transition_matrix[key] = ones(Float64, pomdp.state_structure...)/1000
        local_transition_matrix[key][final_state] += 1
    end

    # Return the matrix for assignment
    return local_transition_matrix

    # Return empty bogus array. Final type must match shared_data.msg
    #return Float32[]
end 

function state_b_to_a(pomdp::aPOMDP,bpomdp_states::Dict) #it should be return type ?
    # Converts a bPOMDP state to an aPOMDP state, allowing for plug-and-play
    # correspondence between the two

    #println("bpomdp_states: ",bpomdp_states)
    #println("bpomdp state agents: ",bpomdp_states["Agents"][1])
    #println("bpomdp state agents length: ",length(bpomdp_states["Agents"]))
    #println("bpomdp state world: ",bpomdp_states["World"][1])
    #println("bpomdp state world length: ",length(bpomdp_states["World"]))
    #println("size of array world: ", length(bpomdp_states["World"][1]))

    index =1
    alpha_states=Array{Int64}(pomdp.agents_size+(pomdp.nodes_num*length(bpomdp_states["World"][1])))
    #alpha_states= zeros(pomdp.agents_size+(pomdp.nodes_num*length(bpomdp_states["World"][1])))
    #println("alpha_states length: ",length(alpha_states))
    #println(alpha_states)

    #TODO: make it general to use keys of dic===> for k in keys(dict) println(k, " ==> ", dict[k])
    #for k in keys(bpomdp_states) 
     #   println(k, " ==> ", bpomdp_states[k])
    
    for x in 1: length(bpomdp_states["Agents"]) 
        #println("agent in node: ",bpomdp_states["Agents"][x])
        alpha_states[index] = bpomdp_states["Agents"][x]
        index=index+1
    end

    for y in 1: length(bpomdp_states["World"])
        #println("World in node: ",bpomdp_states["World"][y])
        for z in 1: length(bpomdp_states["World"][y])
            alpha_states[index] = bpomdp_states["World"][y][z]
            index = index+1
        end
        
    end
    #end
    #println(alpha_states)
    return alpha_states
    #index = (pomdp.agents_size)+1 #index where i to continou the loop of alpha states 
    #alpha_states=zeros(pomdp.nodes_num,pomdp.world_specfi)
    #q=0
    #construct world_states 
    #for x in 1:pomdp.world_specfi
    #   c = zeros(pomdp.nodes_num)
     #   for y in 1:pomdp.nodes_num
            #create vector 
      #      c[y]=bpomdp_states[index]
       #     index=index+1
        #end 
        #println(c)
        #append vectors 
        #alpha_states[x+q:x+x]=c
        #q=x
    #end
    #alpha_states#to print the result 
end

function state_a_to_b(pomdp::aPOMDP, apomdp_states::Array) ##it should be return type ?
    # Converts an aPOMDP state to a bPOMDP state, allowing for plug-and-play
    # correspondence between the two
    

    #i=1
    #construct agents_states
    #p=0
    #beta_states = zeros(pomdp.agents_size,pomdp.agents_specfi)
    #for x in 1:pomdp.agents_specfi
     #   b = zeros(pomdp.agents_size)
    #  for y in 1:pomdp.agents_size
            #create vector 
     #       b[y]=apomdp_states[i]
      #      i=i+1
       # end 
        #append vectors 
        #beta_states[x+p:x+x]=b'
        #p=x
    #end
     #  beta_states#to print the result 
end 

# Logging
function log_execution(out_file, 
                       num_iter, 
                       reward_change_interval, 
                       re_calc_interval, 
                       exec_time,
                       v_s,
                       action_history,
                       entropy_history,
                       state_history,
                       reward_history,
                       scenario,
                       pomdp,)
    # Number of iterations used
    write(out_file, "- iterations: $num_iter\n")
    # Interval of reward change
    write(out_file, "  reward_change_interval: $reward_change_interval\n")
    # Whether the toy example was run
    write(out_file, "  scenario: $scenario\n")
    # Policy re-calculation inteval
    write(out_file, "  re_calc_interval: $re_calc_interval\n")
    # Time it took to execute the whole scenario
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