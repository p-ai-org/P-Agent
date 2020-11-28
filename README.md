# P-Agent: Training an Autonomous agent to Recognize + Fetch a Package using Pedras and AirSim
Team: Michael Lee, Adam Guo, Vani Sachdev, Christine Cannon

# What is PEDRA?
PEDRA is a programmable engine for Drone Reinforcement Learning (RL) applications. The engine is developed in Python and is module-wise programmable. PEDRA is targeted mainly at goal-oriented RL problems for drones, but can also be extended to other problems such as SLAM etc. The engine interfaces with Unreal gaming engine using AirSim to create the complete platform. Figure below shows the complete block diagram of the engine. [Unreal engine](https://www.unrealengine.com/en-US/) is used to create 3D realistic environments for the drones to be trained in. Different level of details can be added to make the environment look as realistic or as required as possible. PEDRA comes equip with a list of 3D realistic environments that can be selected by user. Once the environment is selected, it is interfaced with PEDRA using using [AirSim](https://github.com/microsoft/AirSim). AirSim is an open source plugin developed by Microsoft that interfaces Unreal Engine with Python. It provides basic python functionalities controlling the sensory inputs and control signals of the drone. PEDRA is built onto the low level python modules provided by AirSim creating higher level python modules for the purpose of drone RL applications.

## Creator
* [Aqeel Anwar](https://www.prism.gatech.edu/~manwar8) - Georgia Institute of Technology

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## We use a Convolutional Neural Network to learn images of packages
From this we get pitch, yaw roll.
 If it doesn't have a direct line of sight, it should rotate (rather than hit a wall).
