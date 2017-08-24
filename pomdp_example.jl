using POMDPs, POMDPModels, POMDPToolbox, QMDP
pomdp = TigerPOMDP()

# initialize a solver and compute a policy
solver = QMDPSolver() # from QMDP
policy = solve(solver, pomdp)
belief_updater = updater(policy)

# run a short simulation with the QMDP policy
println("Simulating TigerPOMDP")
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


