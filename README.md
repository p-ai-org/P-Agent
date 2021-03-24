# P-Agent: Training an Autonomous agent to Recognize + Fetch a Package using Pedras and AirSim
Fall Team: Michael Lee, Adam Guo, Vani Sachdev, Christine Cannon
Spring Team: Michael Lee, Jared Mejia, Danica Du, Rachel Yang, Nessa Kiani

# Project Layout
data: Stores the results of Data Collection\
Data_Collection: Scripts that run data collection\
Network\
-CNN\
-Hardcoded (Controller)\
-*Reinforcement Learning Algorithm (Replaces Hardcoded)*\
Unreal_Envs: Holds project environments\

# How To:
1. Open a packaged unreal environment located in unreal_envs to start the simulation (F1 opens options)
2. Open and execute the python script you wish to run 
3. Exit the simulation by typing "~ Exit"

## Approach
We use a Convolutional Neural Network to learn images of packages and extract information of the agent's relative pitch, yaw roll.
 A hardcoded controller executes protocols (e.g. If it doesn't have a direct line of sight, it should rotate (rather than hit a wall)).

