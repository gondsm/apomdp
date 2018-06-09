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
agents_size = 2
nodes_num = 2 #world size 
agents_structure = [2,2] #every agent will have 2 specifications with two values [node, equipment], nodes 1 or 2, equipment 0 or 1
world_structure = [2,2,2] # every node of the world will have 3 speicifications with 2 values 0 or 1 [fire, debris, victim]

#apomdp_structure = convert_structure(agents_size, nodes_num, agents_structure, world_structure)

println("Creating aPOMDP object")
#pomdp=aPOMDP()
pomdp = aPOMDP("isvr", 1, [3,3], 5, normalize(rand(1), 1), agents_size, agents_structure, nodes_num, world_structure)
println("Finish creating aPOMDP object")

#state_a_to_b(pomdp,alpha_states)

# define function convert_structure
function convert_structure(agents_size::Int64, nodes_num::Int64, agents_structure::Array{Int64,1}, world_structure::Array{Int64,1})
	#counter is index for the apomdp_structure
	counter = 1 
	for i in 1:agents_size
		for j in 1:length(agents_structure)
			apomdp_structure[counter]=agents_structure[j]
			counter+=1
		end
	end

	for i in 1:nodes_num
		for j in 1:length(world_structure)
			apomdp_structure[counter]=world_structure[j]
			counter+=1
		end
	end
	
	println("apomdp_structure:")
	#apomdp_structure # didnt print the array 
	for i in 1: length(apomdp_structure)
        println(apomdp_structure[i])
    end 
end 


# call the function and save it in array 
#
convert_structure(3, 3, [2,1], [3,3,3])
