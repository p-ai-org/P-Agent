import airsim
import cv2
import numpy as np
import pprint
import os
import time

pp = pprint.PrettyPrinter(indent=4)
demo = False

segmentation = True
file_name = "pos.txt"
object_name = "PACKAGE"
unreal_object = [1.2, 1.4, 0]           # Hardcoded Package Details
MAX_HEIGHT = 3.0
slack = 0.5         # Slack in the orientation +/- slack (degrees) so the package is not always completely centered.

# Outline Circle
n = 500       # Points 
offset_x = unreal_object[0]          # 120 units from native unreal -> Airsim (cm -> m); Position of Package relative to spawn
offset_y = unreal_object[1]
r = (120 * .01)         # Radius of Circle (in Unreal)
rev = 1     # Revolutions

# Create Points and angles
pts = np.zeros([n, 3])
pts[:,0] = [r * np.sin(x) + offset_x for x in np.linspace(0, rev * (2*np.pi), n)]
pts[:,1] = [r * np.cos(y) + offset_y for y in np.linspace(0, rev * (2*np.pi), n)]
pts[:,2] = [-np.abs(MAX_HEIGHT * np.sin(z)) for z in np.linspace(0, n, n)]
# pts[:,2] = [z for z in np.linspace(0, -MAX_HEIGHT, n)]

#Pitch: Positive is Up; Yaw: Positive is Right; Ro
# ll: Positive is roll clockwise

pitch = np.arctan(pts[:,2] / r)

yaw = - (np.arcsin( (abs(pts[:,0]) - pts[0,0]) / r ) + np.pi/2)     # Generating angle and flipping signs to turn left
yaw[n//4:3* n//4] = - yaw[n//4:3* n//4]     # Other side of the circle must be reflipped

# connect to the AirSim simulator
client = airsim.VehicleClient()

camera_info = client.simGetCameraInfo("front_left")
print("CameraInfo %d: %s" % (0, pp.pprint(camera_info)))

airsim.wait_key('Press any key to get images')
tmp_dir = os.path.join("./data", "Circular2")

if not demo:
    print ("Saving images to %s" % tmp_dir)
    try:
            os.makedirs(os.path.join(tmp_dir, str(1)))
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    try:
        with open(os.path.join(tmp_dir, file_name), 'w') as f:
            f.write("x   y   z   rel_x   rel_y   rel_z   pitch   yaw   roll\n")
    except OSError:
        raise

for count, [x, y, z] in enumerate(pts):
    #TODO: Add noise to vehicle pose for "Noisy" option.
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch[count] + float(np.random.uniform(-slack, slack, 1)), 0, yaw[count] + float(np.random.uniform(-slack, slack, 1)))), True)    # quaternion in pitch, roll, yaw (Possibly in radians?)
    time.sleep(0.2)     # So you don't throw up from motion sickness when you're watching data collection

    # Save Data with desired channel
    if segmentation:
        success = client.simSetSegmentationObjectID(object_name, 0)
        responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene),         # REGULAR PICTURE
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])

    else:
        responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene)])

    if not demo:
        for i, response in enumerate(responses):

            # Save images to file
            if response.pixels_as_float:
                print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, "Segmentation", str(round(x,2)) + "_" + str(round(y,2)) + "_" + str(round(z,2))+ "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
            elif response.compress:     #png format
                print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                airsim.write_file(os.path.normpath(os.path.join(tmp_dir, "Normal", str(round(x,2)) + "_" + str(round(y,2)) + "_" + str(round(z,2))+ "_" + str(i) + '.png')), response.image_data_uint8)
            else: #uncompressed array - numpy demo
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
                cv2.imwrite(os.path.join(tmp_dir, "Segmentation", str(round(x,2)) + "_" + str(round(y,2)) + "_" + str(round(z,2))+ "_" + str(i) + '.pfm'), img_rgb) # write to png

            if i == 1:
                pp.pprint(client.simGetVehiclePose().position.x_val)
                drone_state = client.simGetVehiclePose

                # Log relevent state information
                pitchRollYaw = airsim.utils.to_eularian_angles(drone_state().orientation)
                with open(os.path.join(tmp_dir, file_name), 'a') as f:
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(drone_state().position.x_val, 
                                                        drone_state().position.y_val,
                                                        drone_state().position.z_val,
                                                        drone_state().position.x_val - unreal_object[0], 
                                                        drone_state().position.y_val - unreal_object[1],
                                                        drone_state().position.z_val - unreal_object[2],
                                                        pitchRollYaw[0],
                                                        pitchRollYaw[1],
                                                        pitchRollYaw[2]))


    # Sanity Check
    pose = client.simGetVehiclePose()
    pp.pprint(pose)

