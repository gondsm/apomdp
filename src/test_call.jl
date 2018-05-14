include("./apomdp.jl")

agents_states = rand(2,2)
world_states = rand(2,3)
	
a_states1 = agents_states[:]
w_states1 = world_states[:]
alpha_states = [a_states1; w_states1]
#state_b_to_a(pomdp::aPOMDP,)
state_a_to_b(pomdp::aPOMDP,alpha_states::Array)