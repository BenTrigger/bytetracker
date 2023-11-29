# Config for REST server AND pedestal client

class Server():
    HOST = '127.0.0.1'
    PORT = 8000

class Pedestal():
    axis_yaw = 0x01
    axis_pitch = 0x02
    axis_roll = 0x03
    default_speed = 10.0
    default_acceleration = 100.0
    # default_mode = "absolute"
    negative_limit = 0.0   # greater than
    positive_limit = 359.9  # less than
    activate_SWLS = 1    # 1 = activate || 0 = deactivate

class Opcode():
    mot_high = 0x01
    get_load_position = 0x09
    set_acceleration = 0x30
    set_speed = 0x31
    send_position = 0x32
    update = 0x34
    set_position_absolute = 0x39
    set_neg_SWLS = 0x36
    set_pos_SWLS = 0x37
    set_position_mode = 0x3B
    axis_on = 0x3C
    axis_off = 0x3D
    set_tum = 0x3F
    activate_SWLS = 0x46

# TODO: Convent to cinfig file, you can use pydantic package for conviciance.
