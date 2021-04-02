"""
Accept CNN results and excute navigation policies
"""

class HardCode_Controller(client, im_bool, rel_position, rel_pich, rel_yaw, rel_roll):
    """
    client: airsim controller object
    im_bool: true if image is in frame
    rel_position: relative position to box (vector)
    relative orientation
        rel_pitch
        rel_yaw
        rel_roll
    """

def move(vec_x, vec_y, vec_z, velocity):
    client.movetoPositionAsync(vec_x, vec_y, vec_z, velocity).join()


def no_package():
    """
    Specify protocols for getting package in frame
    """

    client.moveByAngleRatesZAsync(0, pitch_rate, yaw_rate, z, duration, vehicle_name='')


def policy(velocity = 3):

    client.takeoffAsync().join()

    if not im_bool:
        no_package()

    move(rel_position[x], rel_position[y], rel_position[z], velocity)
    


### IN MAIN.PY ###

while():
    client = airsim.multirotor()
    x,y,z = CNN(client.simGetImages)

    if x == 0 & y == 0 & z == 0:
        break

    HardCode_Controller(client)

print("Job Done")