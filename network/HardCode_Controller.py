import numpy as np

# Possibility one: Given these outputs
# class HardCode_Controller(client, im_bool, rel_position, rel_pich, rel_yaw, rel_roll):
#     """
#     client: airsim controller object
#     im_bool: true if image is in frame
#     rel_position: relative position to box (vector)
#     relative orientation
#         rel_pitch
#         rel_yaw
#         rel_roll
#     """

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

    def __init__(self, client, boundingBox_pts):
        self.client = client
        self.pts = boundingBox_pts
        self.done = False
        self.mem = np.zeros([3,1])
        self.image_Coords = [[0,0], [0,144], [256,144], [256,0]]


    def move(self, pts):
        thresholdArea = (.75)(256)(144)
        if (self.area) < thresholdArea:
            self.moveByRollPitchYawThrottleAsync(self, 0, 0, 0, 0.5, 1, vehicle_name)
            self.done = False
        else:
            self.done = True

    def center(self, pitch_rate = np.pi/2, yaw_rate = np.pi/2, z = 0, duration = 1):
        """
        Specify protocols for getting package centered in camera view. With a Field of view of 90 deg.
        As it is now, drone looks down. Checks. Drone looks up. Checks. Drone straightens and pivots to right. Checks and Loops to beginning
        """
        # TODO: Replace 
        if self.mem[:-3]:
            self.client.moveByAngleRatesZAsync(0, pitch_rate, 0, 0, duration)
            self.mem = np.zeros([3,1])      # Reset to 0
        elif self.mem[:-2]:
            self.client.moveByAngleRatesZAsync(0, -2 * pitch_rate, 0, 0, duration)
        elif self.mem[:-1]:
            self.client.moveByAngleRatesZAsync(0, pitch_rate, yaw_rate, 0, duration)

    def area(self):
        """
        Calculate the area of the bounding box
        """
        area = (self.pts[0]-self.pts[1])*(self.pts[1]-self.pts[2])
        return area


    def policy(self, center_thresh = 50, velocity = 3):

        # Subtract the points and determine if the center is within
        bb_subtract = self.boundingBox_pts - self.image_Coords
        # Do this really janky check
        if 0 > bb_subtract[1][1] and 0 > bb_subtract[2][1] and 0 > bb_subtract[2][0] and 0 > bb_subtract[2][1] and bb_subtract[3][0] \
            and 0 < bb_subtract[0][0] and 0 < bb_subtract[0][1] and 0 < bb_subtract[1][0] and 0 < bb_subtract[3][1]:
            centered_box = True
        else:
            centered_box = False

        if not centered_box:       # This is real janky but... I hope it works.
            self.center()
            np.append(self.mem, 1)      # Yes, we did a search and we didn't find it
            self.mem = self.mem[1:]
        else:
            np.append(self.mem, 0)      # Yes, we did a search and we found it
            self.mem = self.mem[1:]
            move(rel_position[x], rel_position[y], rel_position[z], velocity)


        return self.done, self.mem


print('hello')