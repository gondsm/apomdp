#!/usr/bin/python
# -*- coding: utf-8 -*-


# Futures
from __future__ import division
from __future__ import print_function

# STD
import random
import itertools

# ROS
import rospy
from apomdp.srv import GetAction

# Custom
from conductor import gmu_functions as robot


if __name__ == "__main__":
	rospy.init_node("apomdp_conductor")
	get_action = rospy.ServiceProxy('apomdp/get_action', GetAction)
	#print(a.action)
	robot.init_functions()
	#robot.speak("Cenas")
	#print(robot.listen(2))

	states = list(itertools.product([1,2,3],[1,2,3]))
	for i in range(100):
		#print(random.choice(states))
		a = get_action(random.choice(states))
	