import airsim

# Check if Connection with Airsim is Good
client = airsim.multirotor()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#Takeoff and Prepare for instructions
client.takeoffAsync().join()

done = False

while(done == False):
	pts = CNN(client.simGetImages)   # Where pts = [bottomleft, topleft, topright, bottom right] of bounding box 
	done = HardCode_Controller(client, pts)

print("I got to the Package!")