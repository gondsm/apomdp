#!/usr/bin/env julia06

# The main goal of this script is to serve as an interface for a ROS system using aPOMDP as its decision-making
# technique.
# It should interface with ROS by receiving the observations from the underlying system and relaying them to 
# aPOMDP, control the re-calculation of the policy, and by packing and sending the policy in an appropriate
# message.

# Imports, includes et al
include("./apomdp.jl")
using RobotOS


# Create our ROS types
@rosimport apomdp.srv: GetAction
rostypegen()
using apomdp.srv


# Global vars: given that we're using a ROS service, there's some need
# for global state. This could be solved with a singleton in an OOP,  but
# alas, this is not the case.
# In essence, we maintain the pomdp object, the current policy and a flag
# to indicate it's solving time, allowing the service callback to add data 
# to the pomdp object and not have to solve it immediatelly
pomdp = aPOMDP("isvr")  # The aPOMDP object that we'll be using throughout execution
policy = nothing        # The most up-to-date policy
solve_flag = true       # A global flag that indicates whether we want to solve the POMDP


# Callback for serving the service. It gets the action from the policy, given
# the current observation, and returns it.
function srv_cb(req::GetActionRequest)
    # Get correct action for this observation
    # Pack state into Julia array (could probably be better optimized)
    state = []
    for s in req.observation
        append!(state, convert(Int64, s))
    end
    # Get action
    a = action(policy, apomdpDistribution(pomdp, state))

    # Pack into response
    resp = GetActionResponse()
    resp.action = a
    
    # Add data to the POMDP
    # TODO
    global solve_flag = true

    # Return the response
    return resp
end


# Simple function for updating the system's policy
function update_policy()
    print("Solving aPOMDP... ")
    global policy = solve(pomdp, "qmdp")
    global solve_flag = false
    println("done!")
end


# And a main function
function main()
    # Initialize ROS node
    println("Initializing aPOMDP")
    init_node("policy_calculator")

    # Create the service server object
    const srv_action = Service("apomdp/get_action", GetAction, srv_cb)

    # "spin" while waiting for requests
    println("Going into spin!")
    while ! is_shutdown()
        # Should we re-calculate the policy?
        # If so, we update the global policy and solve flag
        if solve_flag update_policy() end

        # Take a short break from all this
        rossleep(Duration(0.1))
    end
end

main()