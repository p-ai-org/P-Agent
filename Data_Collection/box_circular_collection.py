import airsim

import numpy as np
import pprint
import os
import time

pp = pprint.PrettyPrinter(indent=4)

segmentation = True

# Outline Circle
n = 20       # Points in circle
offset_x = 1.2          # 120 units from native unreal -> Airsim (cm -> m); Position of Package
offset_y = 1.4
r = (170 * .01)         # Radius of Circle (in Unreal) # TODO: Figure out the scaling...

# Create Points and angles

pts = np.zeros([n, 2])


pts[:,0] = [r * np.sin(x) + offset_x for x in np.linspace(0, (2*np.pi), n)]
pts[:,1] = [r * np.cos(y) + offset_y for y in np.linspace(0, (2*np.pi), n)]
z = 0

pitch = np.arcsin(z / (np.sqrt( (pts[:,0] - offset_x)**2 + (pts[:,1]-offset_y)**2 )))
yaw = np.arcsin( (pts[:,1] - offset_y  ) / ( np.sqrt( (pts[:,0] - offset_x)**2 + (pts[:,1]-offset_y)**2 ) ) )

yaw[:n//2] = yaw[:n//2] - (np.pi)
yaw[n//2:] = -yaw[n//2:]


# connect to the AirSim simulator
client = airsim.VehicleClient()

airsim.wait_key('Press any key to get camera parameters')

camera_info = client.simGetCameraInfo("front_left")
print("CameraInfo %d: %s" % (0, pp.pprint(camera_info)))

airsim.wait_key('Press any key to get images')
tmp_dir = os.path.join("./data", "Circular")
# tmp_dir = os.path.join("./")

print ("Saving images to %s" % tmp_dir)
try:
        os.makedirs(os.path.join(tmp_dir, str(1)))
except OSError:
    if not os.path.isdir(tmp_dir):
        raise


for count, [x, y] in enumerate(pts):
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch[count], 0, yaw[count])), True)    # quaternion in pitch, roll, yaw (Possibly in radians?)
    time.sleep(1)

    # Save Data with desired channel
    if segmentation:
        responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene), 
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, True)])
    else:
        responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene)])

    for i, response in enumerate(responses):
        if response.pixels_as_float:
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, "Segmentation", str(x) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(os.path.join(tmp_dir, "Normal", str(x) + "_" + str(i) + '.png')), response.image_data_uint8)
            # airsim.write_file(os.path.normpath(os.path.join(tmp_dir,'trial.png')), response.image_data_uint8)

    # TODO Save the files to Data folder

    pose = client.simGetVehiclePose()
    pp.pprint(pose)

