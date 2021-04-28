import numpy as np
import airsim

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
        if in_frame == False:
            # Rotate and capture
            self.client.moveByVelocityZAsync(0, 0, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 90)).join()
            self.mem += 1

            # Lower the drone based on how many runs it's done        
            if self.mem % 18 == 0:       
                self.client.moveByVelocityZAsync(0, 0, z + 1, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 0)).join()       # Move down by 5 units
                self.mem = 0

        # If the package is in frame, do small turns until it's centered
        else:
            #TODO: If the distace of the max is less towards one side, move in that direction ISSUE WITH THIS!
            self.client.moveByVelocityZAsync(0, 0, 0, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, dir * 20 )).join()

    # def move():
        #TODO: move function will just move the drone and correct for that movement in the orientation of the camera and then just use the camera orientation to direct itself.

    def area(self, coords):
        """
        Calculate the area of the bounding box given a (2,2) Numpy array
        """
        area = (coords[1,0] - coords[0,0]) * (coords[1,1] - coords[0,1])
        return area
    
    def overlap(self, rect, ratio = .4):
        # Create an "inner box" as a threshold
        thresh = np.subtract([self.image_Coords[:,0]], ratio * self.image_Coords[1,0])
        thresh = np.hstack((thresh, np.subtract([self.image_Coords[:,1]], ratio * self.image_Coords[0,1])))
        thresh[0][0] = -thresh[0][0]
        thresh[0][3] = -thresh[0][3]
        thresh = np.reshape(thresh, (2,2))

        # Compare threshold with the bounding box
        intersect = np.sort(np.vstack((thresh, rect)), 0)
        intersect = intersect[1:3,:]        # Pick out the middle coordinates
        intersect[:,1] = np.flip(intersect[:,1])    # swap the y coordinates (the larger one should be 0th)
        int_area = self.area(intersect)
        total_area = self.area(self.image_Coords)

        return int_area/total_area

    def policy(self, center_thresh = .7, velocity = 3):

        # Grab State Information
        state = self.client.simGetVehiclePose()

        # Centering
        if self.pts == []:
            centered_box = False
            self.center(False, state.position.z_val)        # If you can't find the package at all
        else:
            # Check overlap with a "threshold" for the middle
            if self.overlap(self.pts) > center_thresh:
                centered_box = True
            else:
                centered_box = False
                # Determine the direction to turn to fine tune. 
                if (self.image_Coords[1,1]/2 - self.pts[0,0]) > (self.pts[1,0] - self.image_Coords[1,1]/2):
                    self.center(True, state.position.z_val, 1)
                else:
                    self.center(True, state.position.z_val, -1)

        # Move
        if centered_box:
            move(rel_position[x], rel_position[y], rel_position[z], velocity)

        # Check if the drone collided or landed
        collided = self.client.simGetCollisionInfo()
        # landed = self.client.getMultirotorState().landed_state

        if collided.has_collided:
            self.done = True

        return self.done, self.mem, collided

print('hello')