import YAML

if !isdefined(:aPOMDP)
   include("./apomdp.jl")
end 
#include("./apomdp.jl")

agents_states = rand(2,2)
world_states = rand(2,3)
	
a_states1 = agents_states[:]
w_states1 = world_states[:]
alpha_states = [a_states1; w_states1]

state_structure=[2,3]
#reward_name = "svr"
weight = normalize(rand(1),1)
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

#apomdp_structure = convert_structure(n_agents, nodes_num, agents_structure, world_structure)
#TODO: confirm if we need this now or not?
state_structure = Array{Int64, 1}([])
#TODO: call the state_b_to_a then print to see 
state_structure = convert_structure(n_agents, nodes_num, agents_structure, world_structure)


println("Creating aPOMDP object")
#pomdp=aPOMDP()
#pomdp = aPOMDP("isvr", 1, [3,3], 5, normalize(rand(1), 1), n_agents, agents_structure, nodes_num, world_structure)
pomdp = aPOMDP("isvr", 1, [3,3], 5, normalize(rand(1), 1), n_agents, agents_structure, nodes_num, world_structure,nodes_location, nodes_connectivity)
println("Finish creating aPOMDP object")

#state_a_to_b(pomdp,alpha_states)

# call the function and save it in array 
#
#convert_structure(3, 3, [2,1], [3,3,3])

#call fuse_belief 
vector =[] 
belief_f = fuse_beliefs(pomdp,vector)
