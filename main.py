import airsim
import numpy as np

from Network.HardCode_Controller import HardCode_Controller

# Check if Connection with Airsim is Good
client = airsim.MultirotorClient()
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#Takeoff and Prepare for instructions
client.takeoffAsync().join()

done = False
mem = np.zeros([3,1])

while(done == False):
	# pts, confidence = CNN(client.simGetImages)   # Where pts = [bottomleft, topleft, topright, bottom right] of bounding box 
	pts = []
	# TODO: Confidence Threshold Check
	done, mem = HardCode_Controller(client, pts, mem)		# Probably have to call a method after def.
	done.policy()
print("I got to the Package!")