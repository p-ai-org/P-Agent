import airsim

import numpy as np
import pprint
import os
import time

pp = pprint.PrettyPrinter(indent=4)

# Outline Circle
n = 20       # Points in circle
offset = 0
r = (180 * .001) - offset         # Radius of Circle (in Unreal) # TODO: Figure out the scaling...
velocity = 1

pts = np.zeros([n, 2])

pts[:,0] = [r * np.sin(x) for x in range(n)]
pts[:,1] = [r * np.cos(y) for y in range(n)]

# connect to the AirSim simulator
client = airsim.VehicleClient()

airsim.wait_key('Press any key to get camera parameters')

camera_info = client.simGetCameraInfo("front_left")
print("CameraInfo %d: %s" % (0, pp.pprint(camera_info)))

airsim.wait_key('Press any key to get images')
tmp_dir = os.path.join("..", "Data_Collection", "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
        os.makedirs(os.path.join(tmp_dir, str(1)))
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for x, y in pts:
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, 0), airsim.to_quaternion(0, 0, 1)), True)    # Z in unreal coordinates needs to be flipped
    time.sleep(0.1)

    responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene)])

    # TODO Save the files to Data folder

    pose = client.simGetVehiclePose()
    pp.pprint(pose)

