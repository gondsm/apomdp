# Example initially adapted from the documentation at
# https://github.com/JuliaPOMDP/POMDPs.jl
# https://nbviewer.jupyter.org/github/sisl/POMDPs.jl/blob/master/examples/Tiger.ipynb
#  Imports
using POMDPs, POMDPModels, POMDPToolbox, QMDP, SARSOP

# Define a POMDP type
type MyPOMDP <: POMDP{Bool, Symbol, Bool} # POMDP{State, Action, Observation} all parametarized by Int64s
    r_listen::Float64 # reward for listening (default -1)
    r_findtiger::Float64 # reward for finding the tiger (default -100)
    r_escapetiger::Float64 # reward for escaping (default 10)
    p_listen_correctly::Float64 # prob of correctly listening (default 0.85)
    discount_factor::Float64 # discount
end

# Default constructor
function MyPOMDP()
    return MyPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)
end;

# Define state space
POMDPs.states(::MyPOMDP) = [true, false];

# Define action space
POMDPs.actions(::MyPOMDP) = [:openl, :openr, :listen]
POMDPs.actions(pomdp::MyPOMDP, state::Bool) = POMDPs.actions(pomdp)

# Define action -> index mapping
function POMDPs.action_index(::MyPOMDP, a::Symbol)
    if a==:openl
        return 1
    elseif a==:openr
        return 2
    elseif a==:listen
        return 3
    end
    error("invalid MyPOMDP action: $a")
end;

# Define observation space
POMDPs.observations(::MyPOMDP) = [true, false];
POMDPs.observations(pomdp::MyPOMDP, s::Bool) = observations(pomdp);

# Define distribution type for observations and transitions
type MyPOMDPDistribution
    p::Float64
    it::Vector{Bool}
end
MyPOMDPDistribution() = MyPOMDPDistribution(0.5, [true, false])
POMDPs.iterator(d::MyPOMDPDistribution) = d.it

# transition and observation pdf
function POMDPs.pdf(d::MyPOMDPDistribution, so::Bool)
    so ? (return d.p) : (return 1.0-d.p)
end;

# Sampling function for our distribution
POMDPs.rand(rng::AbstractRNG, d::MyPOMDPDistribution) = rand(rng) <= d.p;

# Transition model
# Resets the problem after opening door; does nothing after listening
function POMDPs.transition(pomdp::MyPOMDP, s::Bool, a::Symbol)
    d = MyPOMDPDistribution()
    if a == :openl || a == :openr
        d.p = 0.5
    elseif s
        d.p = 1.0
    else
        d.p = 0.0
    end
    d
end;

# Define reward model
function POMDPs.reward(pomdp::MyPOMDP, s::Bool, a::Symbol)
    r = 0.0
    a == :listen ? (r+=pomdp.r_listen) : (nothing)
    if a == :openl
        s ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    if a == :openr
        s ? (r += pomdp.r_escapetiger) : (r += pomdp.r_findtiger)
    end
    return r
end
POMDPs.reward(pomdp::MyPOMDP, s::Bool, a::Symbol, sp::Bool) = reward(pomdp, s, a);

# Define observation model
function POMDPs.observation(pomdp::MyPOMDP, a::Symbol, s::Bool)
    d = MyPOMDPDistribution()
    pc = pomdp.p_listen_correctly
    if a == :listen
        s ? (d.p = pc) : (d.p = 1.0-pc)
    else
        d.p = 0.5
    end
    d
end;

# Distount factor
POMDPs.discount(pomdp::MyPOMDP) = pomdp.discount_factor

# Number of states, actions and observations
POMDPs.n_states(::MyPOMDP) = 2
POMDPs.n_actions(::MyPOMDP) = 3
POMDPs.n_observations(::MyPOMDP) = 2;

# Initial uniform distribution
POMDPs.initial_state_distribution(pomdp::MyPOMDP) = MyPOMDPDistribution(0.5, [true, false]);

# Create new MyPOMDP instance
pomdp = MyPOMDP()

# Initialize solver
solver = QMDPSolver()
#solver = SARSOPSolver()

# Get a policy
policy = solve(solver, pomdp)

# Create a belief updater 
belief_updater = updater(policy)

# Run a simulation
println("Simulating MyPOMDP")
history = simulate(HistoryRecorder(max_steps=20), 
                   pomdp, 
                   policy, 
                   belief_updater)

# look at what happened
for (s, b, a, o) in eachstep(history, "sbao")
    println("State was $s,")
    println("belief was $b,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
println("Discounted reward was $(discounted_reward(history)).")


