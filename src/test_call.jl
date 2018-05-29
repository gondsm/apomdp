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
reward_name = "svr"
weight = normalize(rand(1),1)
agents_size=2
agents_specfi=2
nodes_num=2
world_specfi=3

state_values_dict::Dict


#pomdp = aPOMDP(reward_name, 1, state_structure, 5,weight,agents_size,agents_specfi,nodes_num,world_specfi)

pomdp = aPOMDP(5,state_values_dict,1,normalize(rand(n_v_s)),transition_dict, reward_dict,0.95,states,state_indices,state_structure,"svr",agents_size,agents_specfi,nodes_num,world_specfi)

#state_a_to_b(pomdp,alpha_states)