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

# The list of questions to be used in this test
# (partly re-used from the bum_conductor)
questions = ["What can you tell me about your day?",
             "What are your favourite hobbies?",
             "How is your research going? Are you getting good results?",
             "Did you study electrical engineering? What did you like the most about it?",
             "Do you like working with your colleagues? Do you like the environment at your work?",
             "Have you ever supervised any students? What was it like?",
             "I hate repeating myself. Have you ever had to repeat any experiments, or do your experiments go well on the first try?",
             "Are you a researcher? What is your field of research?",
             "Being a robot, I feel thunderstorms very personally. How do you feel about the weather we have been having lately?"]
questions_satisfaction = ["Are you enjoying this interaction?"]

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
actions = {
           1: lambda: robot.ask_question(questions, replace=False, speech_time=False, keyboard_mode=false), # Ask the user a question
           2: lambda: robot.step_forward(False),	# Take a step forward
           3: lambda: robot.step_forward(True),		# Take a step back
           4: lambda: increase_volume(False),		# Increase speaking volume
           5: lambda: increase_volume(True)			# Decrease speaking volume
          }


def increase_volume(reverse=False):
	""" Increases or decreases the current volume. No input checks for now. 
	Updates the global state variable. 
	"""
	# Volume percentages we go through
	volume_steps = [50, 70, 80]

	# Increase
	if not reverse:
		robot.change_volume(volume_steps[int(current_volume)])
		global current_volume
		current_volume += 1

	# Decrease
	else:
		global current_volume
		current_volume -= 1
		robot.change_volume(volume_steps[int(current_volume)])


def estimate_state():
	""" Estimates the current satisfaction level of the user. """
	# Ask the question
	words = robot.ask_question(questions_satisfaction, replace=True, speech_time=True, keyboard_mode=false)

	# Check if it was a negative answer
	if re.search(re_no, words):
		rospy.loginfo("User is more satisfied!")

	# Check if it was a posivite answer
	elif re.search(re_no, words):
		rospy.loginfo("User is not satisfied!")

	# And what if it's not recognized?
	else:
		rospy.logwarn("I got an unrecognized result for the state estimation!")

	# And return the current state
	return [current_satisfaction, current_volume, current_distance]


def execute_action(action):
	""" Executes an action, updating the robot's current state """
	# If action is to move forward or back, we need to check with the current state
	if action == 2:		# Step forward
		return
	elif action == 3:	# Step back
		return

	# If action is to mess with the volume, the same
	elif action == 4:	# Increase volume
		return
	elif action == 5:	# Lower volume
		return

	actions[action]()


def main():
	""" The main loop that executes the demo. """
	# Initialization
	rospy.init_node("apomdp_conductor")		
	get_action = rospy.ServiceProxy('apomdp/get_action', GetAction)
	robot.init_functions()

	# Main loop
	state = [1, current_volume, current_distance]
	while not rospy.is_shutdown():
		# Get an action
		a = get_action(state)

		# Execute action
		execute_action(a)

		# Update state
		state = estimate_state()


if __name__ == "__main__":
	main()
