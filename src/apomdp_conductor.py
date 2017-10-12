#!/usr/bin/python
# -*- coding: utf-8 -*-


# Futures
from __future__ import division
from __future__ import print_function

# ROS
import rospy
from apomdp.srv import GetAction

# Custom
#from conductor import gmu_functions as robot


if __name__ == "__main__":
	get_action = rospy.ServiceProxy('apomdp/get_action', GetAction)
	a = get_action([1,1])
	print(a.action)