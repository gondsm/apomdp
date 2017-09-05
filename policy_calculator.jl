include("./apomdp.jl")

using RobotOS

# Initialize POMDP
println("Initializing aPOMDP")
pomdp = aPOMDP()

# Initialize solver
println("Initializing solver")
solver = QMDPSolver()
#solver = SARSOPSolver() # Brings a whole new host of problems

# Get a policy
print("Solving... ")
policy = solve(solver, pomdp)
println("Done!")

#println(policy)

# Create a belief updater 
println("Creating belief updater")
belief_updater = updater(policy)

# Run a simulation
println("Simulating POMDP")
history = simulate(HistoryRecorder(show_progress=true, max_steps=20), pomdp, policy, belief_updater)

# look at what happened
for (s, b, a, o) in eachstep(history, "sbao")
    println("State was $s,")
    println("belief was $b,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
println("Discounted reward was $(discounted_reward(history)).")