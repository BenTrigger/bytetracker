# Config for TCP MOCK server OR PEDESTAL 

class Packet_structure():
    start_byte_1 = 0x50
    start_byte_2 = 0x54
    length = 0x04
    group_id = 0x00

class Reply():
    Ack = 0x06
    Nack_Invalid_Command = 0xA6
    Nack_Execution_Error = 0xE6
    Nack_Wrong_Checksum = 0xF6

class Axis_id():
    YAW = 0x01
    PITCH = 0x02
    ROLL = 0x03

# MOCK SERVER OR PEDESTAL IP & PORT
class Server():
    ## PEDESTALS IP AND PORT:
    HOST = '192.168.10.120'
    PORT = 4949

    ## MOCK SERVER IP & Port
    # HOST = '127.0.0.1'
    # PORT = 9999
