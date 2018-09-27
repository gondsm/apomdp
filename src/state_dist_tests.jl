import YAML
include("./apomdp.jl")

# Calculate state distances
function calc_state_dist(pomdp, s1, s2)
	state_1 = collect(state_from_index(pomdp,s1))
	state_2 = collect(state_from_index(pomdp,s2))
	println("States:")
	println(state_1)
	println(state_2)
	dist = norm(state_1 - state_2) 
	return dist
end

# Check how many states fall within a certain distance
function sample_distances(pomdp, observation)
	# Sample states from a fixed observation and show the distances
	n_examples = 200

	obs = pomdp.states[rand(1:pomdp.state_structure[1])]

	dists = zeros(n_examples)

	for i in 1:n_examples
		new_state = pomdp.states[rand(1:pomdp.state_structure[1])]
		dists[i] = calc_state_dist(pomdp, obs, new_state)
	end

	println("Distances:")
	println(sort(dists))
end


function get_closest_states(pomdp, observation)

	obs = pomdp.states[rand(1:pomdp.state_structure[1])]

	n_examples = 100000

	threshold = 3.0

	neighborhood = []

	for i in 1:n_examples
		println(i)
		new_state = pomdp.states[rand(1:pomdp.state_structure[1])]
		if calc_state_dist(pomdp, new_state, obs) < threshold
			append!(neighborhood, i)
		end
	end

	#println(neighborhood)
	println("Number of examples: $n_examples")
	println("Size of neighborhood below $threshold:")
	println(size(neighborhood))
end


function test_modifying_state(pomdp)
	# Fixed observation
	obs = pomdp.states[rand(1:pomdp.state_structure[1])]
	println("Selected observation:")
	println(obs)

	s1 = collect(state_from_index(pomdp, obs))


	# Generate the neighborhood instead of looking for it
	neighborhood = []
	for i in 1:length(s1)
		# For each element in the observation, add one and subtract one
		# (taking care of the boudaries)
		# add to the neighborhood
	end

	# Obs
	# [1, 1, 1, 1]
	# neighborhood:
	# [2, 1, 1, 1]
	# [1, 2, 1, 1]
	# [...]
	# [0, 1, 1, 1]
	# [...]

end


# Read configuration
config = YAML.load(open("/home/vsantos/catkin_ws/src/apomdp/config/common.yaml"))
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

# Haha, create object
println("Creating aPOMDP object")
pomdp = aPOMDP("isvr", 1, [3,3], 5, normalize(rand(1), 1), n_agents, agents_structure, nodes_num, world_structure,nodes_location, nodes_connectivity)
println("Finish creating aPOMDP object")

println("Starting distance tests")
#sample_distances(pomdp, observation)

#get_closest_states(pomdp, observation)

test_modifying_state(pomdp)