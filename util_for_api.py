import numpy as np


def fix_minus(ang):
    if ang < 0:
        return ang + 360
    return ang
"""
:param x_offset   - Leftmost c of the ROI # (0 if the ROI includes the left edge of the image)
:param y_offset   - Topmost pixel of the ROI # (0 if the ROI includes the top edge of the image)
:param width  - Width of ROI
:param height  - Height of ROI
:param img_width - Pixels
:param img_height - Pixels
:param num_of_cams - int number of cameras
:param obj_in_cam_num - int camera number - which contains the center of the object.
:return: Az_deg, El_deg
"""
def bbox_to_angles(x_offset, y_offset, width, height, img_width, img_height, wide_len=True, num_of_cams=1, obj_in_cam_num=1):
    if img_width == 3840:
        if wide_len:
            MULTIPLICATIVE = 461
        else:
            MULTIPLICATIVE = 183
    elif img_width == 1920:
        if wide_len:
            MULTIPLICATIVE = 922
        else:
            MULTIPLICATIVE = 366
    elif img_width == 1280:
        MULTIPLICATIVE = 549
    else:
        raise Exception("resolution should be: 4k / FHD / HD")
    # step 1
    x_center = x_offset + width / 2
    y_center = y_offset + height / 2

    # step 2: get X,Y values relative to the center
    if num_of_cams == 1:
        X = x_center - img_width / 2
        Y = -y_center + img_height / 2
    elif num_of_cams == 4:
        if obj_in_cam_num==1: # cam 1 - top left
            X = x_center - img_width
            Y = -y_center + img_height
        elif obj_in_cam_num==2: # cam 2 - top right
            X = x_center
            Y = -y_center + img_height
        elif obj_in_cam_num==3: # cam 3 - bottom left
            X = x_center - img_width
            Y = -y_center
        elif obj_in_cam_num==4: # cam 4 - bottom right
            X = x_center
            Y = -y_center
        else:
            raise Exception("obj_in_cam_num should be between 1 to 4")
    else:
        raise Exception("num_of_cams can be 1 or 2")

    # step 3
    print('micro rad x: %s ,  micro rad  y:   %s' % (X,Y))
    Az_rad = pow(10,-6) * MULTIPLICATIVE * X #MicroRad
    El_rad = pow(10,-6) * MULTIPLICATIVE * Y #MicroRad
    # step 4
    Az_deg = fix_minus(np.rad2deg(Az_rad))
    El_deg = fix_minus(np.rad2deg(El_rad))
    return Az_deg, El_deg


if __name__ == "__main__":
    # object 100x100 pixels top left
    print('top left: ')
    print(bbox_to_angles(-50,-50, 100,100,3840,2160))

    # object 100x100 pixels top right
    print('top right: ')
    print(bbox_to_angles(3790,-50, 100,100,3840,2160))

    # object 100x100 pixels bottom left
    print('bottom left: ')
    print(bbox_to_angles(-50,2110, 100,100,3840,2160))

    # object 100x100 pixels bottom right
    print('bottom right: ')
    print(bbox_to_angles(3790,2110, 100,100,3840,2160))