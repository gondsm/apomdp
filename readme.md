# The aPOMDP ($\alpha$POMDP) framework for User-Adaptive Decision Making
This package contains the Julia code used in our simulations, as well as our Python plotter. In the future it will contain the ROS package that will tackle the user-adaptiveness problem on-line.

The main entry point is the `hri_simulator.jl` script, which is our main simulator. This makes use of the "class" defined in `apomdp.jl`.

`plotter.py` can be used to obtain the illustrations used throughout the papers.