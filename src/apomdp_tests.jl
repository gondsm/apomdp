# A small script that works as a unit test of sorts for the aPOMDP module

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

include("./apomdp.jl")

#pomdp = aPOMDP("msvr", 2)
#pomdp = aPOMDP("msvr", 3)
pomdp = aPOMDP("isvr")

set_state_value(pomdp, [2,1], 10)

integrate_transition(pomdp, [1,1], [2,1], 1)

calculate_reward_matrix(pomdp)

println([1,1])
println([2,1])

#println(pomdp.reward_matrix)

# Test solvers
#policy = solve(pomdp, "despot")
#policy = solve(pomdp, "qmdp")

# Test integrating transitions, rewards, etc
# println(calc_average_entropy(pomdp))
# integrate_transition(pomdp, pomdp.states[1], pomdp.states[2], 1)
# integrate_transition(pomdp, [1,1], [1,3], 2)
print("Average entropy: ")
println(calc_average_entropy(pomdp))
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
dist = apomdpDistribution(pomdp)
println("Distribution: ", dist.dist)
# println("Entropy of uniform: ", calc_entropy(dist.dist))
dist2 = apomdpDistribution(pomdp, [1,1])
println("Distribution: ", dist2.dist)
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