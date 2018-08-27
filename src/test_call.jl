import YAML

#Pkg.add("PyPlot")
#using PyPlot
#using Plots

if !isdefined(:aPOMDP)
   include("./apomdp.jl")
end
include("./apomdp.jl")


#function plot_(x, y)
#	plot(x, y, color="red", linewidth=2.0, linestyle="--")
#end 

#=agents_states = rand(2,2)
world_states = rand(2,3)
	
a_states1 = agents_states[:]
w_states1 = world_states[:]
alpha_states = [a_states1; w_states1]

state_structure=[2,3]

weight = normalize(rand(1),1)=#
#reward_name = "svr"
#agents_size=2
#agents_specfi=2
#nodes_num=2
#world_specfi=3

###############################
#	convert_structure         #
# apomdp_structure = [agent1_stucture, agent2_strucure,..agentN_stucture,node1_stucture,node2_stucture...nodeN_structure]
###############################
# define variables 
#n_agents = 2
#nodes_num = 2 #world size 
#agents_structure = [2,2] #every agent will have 2 specifications with two values [node, equipment], nodes 1 or 2, equipment 0 or 1
#world_structure = [2,2,2] # every node of the world will have 3 speicifications with 2 values 0 or 1 [fire, debris, victim]

# Read configuration
config = YAML.load(open("/home/hend/catkin_ws/src/apomdp/config/common.yaml"))
n_actions = config["n_actions"]
n_agents = config["n_agents"]
nodes_num = config["nodes_num"]
agents_structure = config["agents_structure"]
world_structure = config["world_structure"]
nodes_location = config["nodes_location"]
nodes_connectivity = config["nodes_connectivity"]
agents_capabibilities = config["Agents_capabiblities"]
println("Read a configuration file:")
println("n_actions: ",n_actions)
println("n_agents: ",n_agents)
println("nodes_num: ",nodes_num)
println("agents_structure: ",agents_structure)
println("world_structure: ",world_structure)
println("nodes_location: ",nodes_location)
println("nodes_connectivity: ",nodes_connectivity)
println("Agents_capabiblities: ",agents_capabibilities)

#apomdp_structure = convert_structure(n_agents, nodes_num, agents_structure, world_structure)
#TODO: confirm if we need this now or not?
# state_structure = Array{Int64, 1}([])
#TODO: call the state_b_to_a then print to see 
# state_structure = convert_structure(n_agents, nodes_num, agents_structure, world_structure)

println("Creating aPOMDP object")
pomdp = aPOMDP("isvr", 1, [3,3], 5, normalize(rand(1), 1), n_agents, agents_structure, nodes_num, world_structure,nodes_location, nodes_connectivity)
println("Finish creating aPOMDP object")

#state_a_to_b(pomdp,alpha_states)

# call the function and save it in array 
#convert_structure(3, 3, [2,1], [3,3,3])

#call fuse_belief #########################################
#belief_vector = [[1.0 2.0], [3.0 4.0], [5.0 6.0]]
#fused_belief = fuse_beliefs(pomdp, belief_vector)
#println("fused_belief:",fused_belief)

#call fuse_transition #####################################
#println("transition_vector")
# Bogus transitions
# For 2 states and 2 actions
t1 = Dict([1,1] => [0.25, 0.75], [1,2] => [0.15, 0.85], [2,1] => [0.05, 0.95], [2,2] => [0.95, 0.05])
t2 = Dict([1,1] => [0.35, 0.65], [1,2] => [0.25, 0.75], [2,1] => [0.75, 0.25], [2,2] => [0.85, 0.15])
t3 = Dict([1,1] => [0.45, 0.55], [1,2] => [0.35, 0.65], [2,1] => [0.65, 0.35], [2,2] => [0.25, 0.75])
println("t1:", t1)
transition_vector = [t1, t2, t3]
fused_T = fuse_transitions(pomdp, transition_vector)
println("fused_T: ", fused_T)

# call learn function ####################################
#bogas info for 2 states and 2 actions 
#=current_belief = [0.6 0.4]
action = 1
previous_belief = [0.1 0.9]
local_transition_matrix = Dict([1,1] => [0.25, 0.75], [1,2] => [0.15, 0.85], [2,1] => [0.05, 0.95], [2,2] => [0.95, 0.05]) 
println("local_transition_matrix before learn: ", local_transition_matrix)
local_transition_matrix = learn(pomdp, current_belief, action, previous_belief, local_transition_matrix)
println("local_transition_matrix after learn: ", local_transition_matrix)
=#

#call get_policy #########################################
#for two actions and two states 
c_vector = [1, 0]
println("c_vector:", c_vector)
policy = get_policy(pomdp, fused_T, c_vector)
println("policy: ", policy)


#plot(fused_belief, color="red", linewidth=2.0, linestyle="--")
#=b = Any[]
println("beliefs: ",belief_vector)
println("size(belief_vector,1):", size(belief_vector,1))
println("size(belief_vector[1],1):", size(belief_vector[1],1))
println("size(belief_vector,2):", size(belief_vector,2))
println("length(belief_vector):", length(belief_vector))

for i=1:size(belief_vector,1)
	println("belief_vector[i]:", belief_vector[i])
	if i==1
		b=belief_vector[i]
	else
		b+=belief_vector[i]
	end 
	println("b", b)
end 
#normalize 
#
b[:] = normalize(b[:], 1)
println("normalized b: ", b)=#

#println("(belief_vector[1]:", belief_vector[1])
#result = sum(belief_vector,(1))
#fused_belief = float(belief_vector)
#println("beliefs after sum: ",fused_belief)
#normalize 
#
#fused_belief[:] = normalize(fused_belief[:], 1)
#println("normalized fused_belief: ", fused_belief)

#display(display(plot(result, fused_belief, color="red", linewidth=2.0, linestyle="--")))
#plotly() # Choose the Plotly.jl backend for web interactivity
#plot(rand(5,5),linewidth=2,title="My Plot")
#=fused_belief = zeros(size(belief_vector,2))
println("size(belief_vector,1): ", size(belief_vector,2))

for x=1:size(belief_vector,1)     
    println("belief_vector: ",belief_vector)
    println("fused_belief: ",fused_belief)
    fused_belief = fused_belief + belief_vector[x]
end
println("not-normalized fused_belief: ", fused_belief)
#normalize 
fused_belief[:] = normalize(fused_belief[:], 1)
println("normalized fused_belief: ", fused_belief)
=#
#belief_f = fuse_beliefs(pomdp,belief_vector)
#println("after fused: ", belief_f)
#plot_(result, fused_belief)

