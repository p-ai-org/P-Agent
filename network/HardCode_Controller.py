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

class HardCode_Controller(client, boundingBox_pts):
    """
    Accept CNN results and excute navigation policies

    In:
    client: airsim controller object
    boundingBox_pts: List of Bounding Box Points (assumed to be [bottomleft, topleft, topright, bottomright])
    mem: Saves the past 3 actions to check if searching
    pictureCoords: The coordinates of the edges of the picture (default = "Width": 256,"Height": 144) with 0,0 as bottom left corner

    Out:
    Takes action in Unreal Environment
    """

    def __init__(self):
        self.client = client
        self.pts = boundingBox_pts
        self.mem = np.zeros([3,1])
        self.pictureCoords = [[0,0], [0,144], [256,144], [256,0]]

    def move(self, pts):
        thresholdArea = (.75)(256)(144)
        while ((pts[0]-pts[1])*(pts[1]-pts[2])) < thresholdArea:
            self.moveByRollPitchYawThrottleAsync(self, 0, 0, 0, 0.5, 1, vehicle_name)


    def center(self, pitch_rate = np.pi/2, yaw_rate = np.pi/2, z = 0, duration = 1):
        """
        Specify protocols for getting package centered. With a Field of view of 90 deg. adjustments can be made
        As it is now, drone looks down. Checks. Drone looks up. Checks. Drone straightens and pivots to right. Checks and Loops to beginning
        """
        elif self.mem[:-3]:
            self.client.moveByAngleRatesZAsync(0, pitch_rate, yaw_rate, 0, duration)
            self.mem = np.zeros([3,1])      # Reset to 0
        elif self.mem[:-2]:
            self.client.moveByAngleRatesZAsync(0, -2 * pitch_rate, 0, 0, duration)
        elif self.mem[:-1]:
            self.client.moveByAngleRatesZAsync(0, pitch_rate, 0, 0, duration)


    def area(self):
        """
        Calculate the area of the bounding box
        """
        area = self.pts[0] * self.pts[1] * self.pts[2] * self.pts[3]
        return area

    def finish(self, thresh):
        """
        In:
        thresh: Threshhold for area above which we can consider the package to be picked up / rel. distance = 0
        """
        if area() > thresh:
            return True
        else:
            return False

    def policy(self, center_thresh = 50, velocity = 3):

        client.takeoffAsync().join

        # TODO: Replace 

        if # TODO: Compare picture coordinates with bounding box coordinates on the picture. (If not centered) Idea: sum(self.pictureCoords-center_thresh > ):
            self.center()
            np.append(self.mem, 1)      # Yes, we did a search and we didn't find it
            self.mem = self.mem[1:]
        else:
            np.append(self.mem, 0)      # Yes, we did a search and we found it
            self.mem = self.mem[1:]
            move(rel_position[x], rel_position[y], rel_position[z], velocity)

        if finish():
            return True
        else:
            return False


### IN MAIN.PY ###

