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


# Callback for serving the service. It gets the action from the policy, given
# the current observation, and returns it.
function srv_cb(req::GetActionRequest)
    println("Service called!")
    resp = GetActionResponse()
    println("Got a state:")
    for s in req.observation
        println(s)
    end
    resp.action = 1
    return resp
end

function main()
    # Initialize ROS node
    init_node("policy_calculator")

    # Create the service server object
    const srv_action = Service("apomdp/get_action", GetAction, srv_cb)

    # "spin" while waiting for requests
    println("Going into spin!")
    while ! is_shutdown()
        rossleep(Duration(0.1))
    end
end

main()