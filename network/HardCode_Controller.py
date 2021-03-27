"""
Accept CNN results and excute navigation policies
"""

class HardCode_Controller(im_bool, rel_position, rel_pich, rel_yaw, rel_roll):
    """
    im_bool: true/false if image is in frame
    rel_position: relative position to box
    relative orientation
        rel_pitch
        rel_yaw
        rel_roll
    """
    