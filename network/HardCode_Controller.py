import numpy as np
import airsim
from shapely.geometry import Polygon
from pyquaternion import Quaternion

class HardCode_Controller:
    """
    Accept CNN results and excute navigation policies 
        If package is in frame and can be drawn, 4 element array
        If package cannot be drawn, CNN passes an empty array

    In:
    client: airsim controller object
    boundingBox_pts: List of Bounding Box Points (assumed to be [bottomleft, topleft, topright, bottomright])
    mem: Saves the past 3 actions to check if searching
    image_Coords: The coordinates of the edges of the picture (default = "Width": 256,"Height": 144) with 0,0 as bottom left corner

    Out:
    Takes action in Unreal Environment
    """

    def __init__(self, client, boundingBox_pts, mem):
        self.client = client
        self.pts = boundingBox_pts
        self.done = False
        self.mem = mem
        self.image_Coords = np.array([[0,144],[256,0]])

    def center(self, in_frame, z, dir = 1, duration = 0.2):
        """
        Specify protocols for getting package centered in camera view. With a Field of view of 90 deg.
        As it is now: drone does a 360 sweep and if no package is found, moves to a lower level and checks again.
        dir is the direction you will turn in fine tuning (1 = clockwise, -1 = counter clockwise)
        """
        
        # If the package isn't in frame, rotate and try to get it in frame
        if not in_frame:
            # Rotate and capture
            self.client.moveByVelocityZAsync(0, 0, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 90)).join()
            self.mem += 1

            # Lower the drone based on how many runs it's done        
            if self.mem % 16 == 0:       
                self.client.moveByVelocityZAsync(0, 0, z + 1, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 0)).join()       # Move down by 5 units
                self.mem = 0

        # If the package is in frame, do small turns until it's centered
        else:
            if dir == 0:
                self.client.moveByVelocityZAsync(0, 0, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0)).join()
            else:
                #TODO: If the distace of the max is less towards one side, move in that direction
                self.client.moveByVelocityZAsync(0, 0, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, dir * 20 )).join()

    # def move():
        #TODO: move function will just move the drone and correct for that movement in the orientation of the camera and then just use the camera orientation to direct itself.

    def area(self, coords):
        """
        Calculate the area of the bounding box given a (2,2) Numpy array
        """
        area = (coords[1,0] - coords[0,0]) * (coords[1,1] - coords[0,1])
        return area
    
    def overlap(self, rect, thresh_rect, ratio = .1):
        """
        rect: Bounding box points (upper left, lower right)
        thresh: Box to compare with (upper left, lower right) - Gets scaled down
        ratio: amount to take off
        """

        # Create an array with all points. 0: upper left corner, 1: lower left, 2: lower right, 3: upper right
        full_pts = np.array([ [rect[0,0], rect[0,1]], [rect[0,0], rect[1,1]], [rect[1,0], rect[1,1]], [rect[1,0], rect[0,1]] ])
        full_pts = list(map(tuple, full_pts))

        # Create an "inner box" as a threshold (move the box down?)
        thresh_x = np.add(thresh_rect[0,0], ratio * thresh_rect[1,0])
        thresh_x = np.append(thresh_x, np.subtract([thresh_rect[1,0]], ratio * thresh_rect[1,0]))
        thresh_x = np.reshape(thresh_x, (2,1))
        thresh_y = np.add([thresh_rect[0,1]], ratio * thresh_rect[0,1])
        thresh_y = np.append(thresh_y, np.subtract([thresh_rect[1,1]], ratio * thresh_rect[0,1]))
        thresh_y = np.reshape(thresh_y, (2,1))
        thresh = np.hstack((thresh_x, thresh_y))

        full_thresh = np.array([ [thresh[0,0], thresh[0,1]], [thresh[0,0], thresh[1,1]], [thresh[1,0], thresh[1,1]], [thresh[1,0], thresh[0,1]] ])
        full_thresh = list(map(tuple, full_thresh))

        # Create Polygons from areas and check overlap
        polygon = Polygon(full_pts)
        other_polygon = Polygon(full_thresh)
        return polygon.intersection(other_polygon).area/other_polygon.area

    def transformToEarthFrame(self, vector, q_):
        """
        onvert coordinates w.r.t drone -> global ("headless")
        """
        q = Quaternion(q_)
        return q.rotate(vector)

    def move(self, z, duration = 0.2, desired_velocity = 1):
        """
        Basic move command which just moves the drone forward
        """
        # q = self.client.getCameraInfo(0).pose.orientation #this is different from the getOrientation, taking the latter can create problem due to the fact that the drone move slower than the cam?
        # my_quaternion = Quaternion(w_val=q.w_val,x_val=q.x_val,y_val= q.y_val,z_val=q.z_val)
        # mvm = my_quaternion.rotate(action)
        # self.client.moveByVelocityZAsync(2, 0, z, duration, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=YawMode(False, 0))
        vx, vy, _ = self.transformToEarthFrame([desired_velocity, 0, 0], [self.client.simGetVehiclePose().orientation.w_val,\
            self.client.simGetVehiclePose().orientation.x_val,\
            self.client.simGetVehiclePose().orientation.y_val,self.client.simGetVehiclePose().orientation.z_val])
        self.client.moveByVelocityZAsync(vx=vx, vy=vy, z=z, yaw_mode=airsim.YawMode(True, 0), drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, duration=duration).join()

    def policy(self, center_thresh = .05, velocity = 3):

        # Grab State Information
        state = self.client.simGetVehiclePose()

        # Centering
        if self.pts == []:
            centered_box = False
            self.center(False, state.position.z_val)        # If you can't find the package at all
        else:
            # Check overlap with a "threshold" which is just an area in the lower center (x-edges brought in by 40%)
            thresh_coord = self.image_Coords
            thresh_coord[1,1] = thresh_coord[0,1]
            thresh_coord[0,1] = 0.5* thresh_coord[0,1]
            thresh_coord[0,0] = thresh_coord[0,0] + 0.40 * self.image_Coords[1,0]
            thresh_coord[1,0] = thresh_coord[1,0] - 0.40 * self.image_Coords[1,0]

            if self.overlap(self.pts, thresh_coord) > center_thresh:
                centered_box = True
            else:
                centered_box = False
                # Determine the direction to turn to fine tune.
                left_area = (self.image_Coords[1,0]/2 - self.pts[0,0])
                right_area = (self.pts[1,0] - self.image_Coords[1,0]/2)

                # Top Area comparison might be USELESS
                top_area = (self.image_Coords[0,1]/2 - self.pts[0,1])

                # If the difference in height is proportionally larger than the differences in left or right directions
                if top_area/(self.image_Coords[0,1]/2) > left_area/(self.image_Coords[1,0]/2) and \
                    top_area/(self.pts[0,1]/2) > right_area/(self.pts[1,0]/2):
                        self.center(True, state.position.z_val + 1, 0)

                if left_area > right_area:
                    self.center(True, state.position.z_val+0.1, -1)     # Rotate left
                else:
                    self.center(True, state.position.z_val+0.1, 1)

        # Move
        if centered_box:
            self.move(state.position.z_val)

        # Check if the drone collided or landed
        collided = self.client.simGetCollisionInfo()
        # landed = self.client.getMultirotorState().landed_state

        if collided.has_collided:
            self.done = True

        return self.done, self.mem, collided

print('hello')