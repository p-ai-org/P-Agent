import airsim
import cv2
import numpy as np
import pprint
import os
import time
# NOTE: AirSim accepts meters but Unreal uses centimeters

pp = pprint.PrettyPrinter(indent=4)
demo = False

segmentation = True
object_name = "PACKAGE"
file_name = "pos.txt"
unreal_object = [1.2, 1.4, 0]           # Hardcoded Package Details

# corner points distance [meters] relative to spawn location 
TOP_LEFT = (-0.9, -0.9)
TOP_RIGHT = (2.7, -0.9)
BOT_LEFT = (-0.9, 2.5)
BOT_RIGHT = (2.7, 2.5)
MAX_HEIGHT = 2.5  # in z-axis ("Roof" is at 3.0)

# connect to the AirSim simulator
client = airsim.VehicleClient()

camera_info = client.simGetCameraInfo("front_left")
print("CameraInfo %d: %s" % (0, pp.pprint(camera_info)))

airsim.wait_key('Press any key to get images')

if not demo:
    tmp_dir = os.path.join("./data", "GridSearch")
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

# MOVEMENT PLAN:
# start top left corner (go down and up, shifting in +x direction)
# go in y direction, stopping in increments
    # in each stop, rotate incrementally in yaw-axis
# when we reach the lower boundary in y-direction, ...
    # shift x, jump back up to upper y position again to repeat

def load_points(x_increment, y_increment, z_increment):
    """
        params:
            y_increment = number of increments in y-direction movement
            x_increment = number of increments in x-direction movement
            z_increment = number of increments in z-direction movement
        returns nx3 numpy array of points evenly distributed and relative
        to spawn point (considered (0,0)), traversing across room, 
        starting in top left corner in bird's eye view
    """
    pts_L = []
    x_coords = np.linspace(TOP_LEFT[0], TOP_RIGHT[0], num=x_increment)
    y_coords = np.linspace(TOP_LEFT[1], BOT_LEFT[1], num=y_increment)
    z_coords = np.linspace(0, -MAX_HEIGHT, num=z_increment)     # Z axis flipped
    
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                pts_L.append([x, y, z])

    pts = np.array(pts_L)
    return pts

def collect_data(x_increment, y_increment, z_increment, rot_increment):
    """
        params:
            ...
            rot_increment = number of increments as camera rotates 
                                around yaw axis in a fixed spot
    """
    # get point data
    pts = load_points(x_increment, y_increment, z_increment)
    rot_coords = np.linspace(0, 2*np.pi, num=rot_increment)

    for [x, y, z] in pts:
        for yaw in rot_coords:
            # TODO: figure out what second param does
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, yaw)), True)  # PRY in radians
            time.sleep(0.2)

            # save data with desired channel
            if segmentation:
                success = client.simSetSegmentationObjectID(object_name, 20)
                responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),  # REGULAR PICTURE
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
            else:
                responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene)])
            
            if not demo:
                for i, response in enumerate(responses):
                    # save images to file
                    if response.pixels_as_float:
                        print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                        airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, "Segmentation", str(round(x,2)) + "_" + str(round(y,2)) + "_" + str(round(z,2))+ "_" + str(round(yaw,2)) + "_"+ str(i) + '.pfm')), airsim.get_pfm_array(response))
                    elif response.compress:  # png format
                        print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                        airsim.write_file(os.path.normpath(os.path.join(tmp_dir, "Normal",str(round(x,2)) + "_" + str(round(y,2)) + "_" + str(round(z,2))+ "_" + str(round(yaw,2)) + "_"+ str(i) + '.png')), response.image_data_uint8)
                    else:  # uncompressed array - numpy demo
                        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                        img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channel image array H X W X 3
                        cv2.imwrite(os.path.join(tmp_dir, "Segmentation", str(round(x,2)) + "_" + str(round(y,2)) + "_" + str(round(z,2))+ "_" + str(round(yaw,2)) + "_"+ str(i) + '.pfm'), img_rgb) # write to png

                    if i == 1:
                        pp.pprint(client.simGetVehiclePose().position.x_val)
                        drone_state = client.simGetVehiclePose

                        # log relevent state information
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
        # sanity check
        pose = client.simGetVehiclePose()
        pp.pprint(pose)



collect_data(5, 5, 5, 10)
