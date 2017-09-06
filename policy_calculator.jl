# The main goal of this script is to serve as an interface for a ROS system using aPOMDP as its decision-making
# technique.
# It should interface with ROS by receiving the observations from the underlying system and relaying them to 
# aPOMDP, control the re-calculation of the policy, and by packing and sending the policy in an appropriate
# message.
include("./apomdp.jl")

using RobotOS

# Initialize POMDP
println("Initializing aPOMDP")
pomdp = aPOMDP()

# Initialize values and rewards
integrate_transition(pomdp, [1,1], [1,2], 1)
#integrate_transition(pomdp, [1,2], [1,2], 3)
integrate_transition(pomdp, [1,1], [1,3], 2)
#integrate_transition(pomdp, [1,3], [1,3], 2)
set_state_value(pomdp, [1,2], 10)
set_state_value(pomdp, [1,3], 20)
calculate_reward_matrix(pomdp)

# Initialize solver
println("Initializing solver")
solver = QMDPSolver()
#solver = SARSOPSolver() # Brings a whole new host of problems

# Get a policy
print("Solving... ")
policy = solve(solver, pomdp)
println("Done!")

# Create a belief updater 
println("Creating belief updater")
belief_updater = updater(policy)

# Run a simulation
println("Simulating POMDP")
history = simulate(HistoryRecorder(max_steps=20), pomdp, policy, belief_updater)

# look at what happened
for (s, b, a, o, r) in eachstep(history, "sbaor")
    println("State was $s,")
    println("belief was $b,")
    println("action $a was taken,")
    println("reward $r was received")
    println("and observation $o was received.\n")
end
println("Discounted reward was $(discounted_reward(history)).")