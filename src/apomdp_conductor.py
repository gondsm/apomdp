#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2017 University of Coimbra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Original author and maintainer: Gonçalo S. Martins (gondsm@gmail.com)

# Futures
from __future__ import division
from __future__ import print_function

# STD
import random
import itertools
import re

# ROS
import rospy
from apomdp.srv import GetAction

# Custom
from conductor import gmu_functions as robot

# The list of questions to be used in this test
# (partly re-used from the bum_conductor)
txt_intro = "My name is ghida and we will now be having a short conversation. When I ask you, please let me know if you are satisfied with my closeness and speaking volume by saying yes or no. Is this okay?"
questions = ["What can you tell me about your day?",
             "What are your favourite hobbies?",
             "How is your research going? Are you getting good results?",
             "Did you study electrical engineering? What did you like the most about it?",
             "Do you like working with your colleagues? Do you like the environment at your work?",
             "Have you ever supervised any students? What was it like?",
             "I hate repeating myself. Have you ever had to repeat any experiments, or do your experiments go well on the first try?",
             "Are you a researcher? What is your field of research?",
             "Being a robot, I feel thunderstorms very personally. How do you feel about the weather we have been having lately?"]
questions_satisfaction = ["Are you satisfied with my speaking volume and distance?", "Do you think I am speaking at the correct volume and distance?", "Are you okay with my current volume and distance?"]

# Regexes for answer processing
# (also partly re-used)
re_yes = ".*(yes|of course).*"
re_no = ".*(no|not).*"

# Current robot state
# (starts in the middle for both state values, 1-indexed)
current_volume = 2
current_distance = 2
current_satisfaction = 1

# A dictionary that maps the possible actions into functions that the system
# can execute.
# It is only ever read, and kept global for readability.
actions = {
           1: lambda: robot.ask_question(questions, replace=False, speech_time=False), # Ask the user a question
           2: lambda: robot.step_forward(False),	# Take a step forward
           3: lambda: robot.step_forward(True),		# Take a step back
           4: lambda: increase_volume(False),		# Increase speaking volume
           5: lambda: increase_volume(True)			# Decrease speaking volume
          }


def increase_volume(reverse=False):
	""" Increases or decreases the current volume. No input checks for now. 
	Updates the global state variable. 
	"""
	# Declare globals
	global current_volume
	global current_volume

	# Volume percentages we go through
	volume_steps = [50, 70, 80]

	# Increase
	if not reverse:
		robot.change_volume(volume_steps[int(current_volume)])	
		current_volume += 1

	# Decrease
	else:
		current_volume -= 1
		robot.change_volume(volume_steps[int(current_volume)])


def estimate_state():
	""" Estimates the current satisfaction level of the user. """
	# Declare the globals we'll use
	global current_satisfaction

	# Ask the question
	words = robot.ask_question(questions_satisfaction, replace=True, speech_time=False)
	rospy.loginfo("Got response: {}.".format(words))

	# Check if it was a negative answer
	if re.search(re_yes, words):
		rospy.loginfo("User is more satisfied!")
		if current_satisfaction < 3:
			current_satisfaction += 1

	# Check if it was a posivite answer
	elif re.search(re_no, words):
		rospy.loginfo("User is not satisfied!")
		if current_satisfaction > 1:
			current_satisfaction -= 1

	# And what if it's not recognized?
	else:
		rospy.logwarn("I got an unrecognized result for the state estimation!")

	# And return the current state
	return [current_satisfaction, current_volume, current_distance]


def execute_action(action):
	""" Executes an action, updating the robot's current state """
	# TODO: Improve the decoupling. At this point, the action dictionary is completely unnecessary.
	# Declare our globals
	global current_distance
	global current_volume

	# If action is to move forward or back, we need to check with the current state
	if action == 2:		# Step forward
		if current_distance > 2:
			rospy.logwarn("Already at maximum distance!")
			rospy.loginfo("Taking a step forward.")
			return False
		else:
			current_distance += 1

	elif action == 3:	# Step back
		if current_distance < 2:
			rospy.logwarn("Already at minimum distance!")
			return False
		else:
			rospy.loginfo("Taking a step back.")
			current_distance -= 1

	# If action is to mess with the volume, the same
	elif action == 4:	# Increase volume
		if current_volume > 2:
			rospy.logwarn("Already at maximum volume!")
			return False
	elif action == 5:	# Lower volume
		if current_volume < 2:
			rospy.logwarn("Already at minimum volume!")
			return False

	# Execute action
	actions[action]()
	return True


def main():
	""" The main loop that executes the demo. """
	# Initialization
	rospy.init_node("apomdp_conductor")		
	get_action = rospy.ServiceProxy('apomdp/get_action', GetAction)
	robot.init_functions()

	# Introduction
	robot.ask_question("Hello, "+txt_intro, kw_repeat=["repeat", "no"], txt_repeat="I was saying that "+txt_intro)

	# Main loop
	state = [1, current_volume, current_distance]
	#while not rospy.is_shutdown():
	for i in range(10):
		# Get an action
		rospy.loginfo("Getting action from service.")
		a = get_action(state).action

		# Execute action
		rospy.loginfo("Executing action {}.".format(a))
		execute_action(a)

		# Update state
		state = estimate_state()
		rospy.loginfo("New estimated state: {}.".format(state))

		# Check whether we're shutting down
		if rospy.is_shutdown():
			break


if __name__ == "__main__":
	main()
