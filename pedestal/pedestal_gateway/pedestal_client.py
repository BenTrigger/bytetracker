import asyncio
import socket
import struct
import logging
from pedestal_mock.config import *
from datetime import datetime
logging.basicConfig(filename='experiment_pc_logs_' + str(datetime.now()) + '.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s -%(message)s')
logger = logging.getLogger(__name__)


def calculate_checksum(message):
    try:
        checksum_bytes = message[2:]
        checksum_value = int(sum(checksum_bytes))
        # Calculate the checksum as the lowest byte of the sum
        checksum = checksum_value & 0xFF
        return checksum
    except Exception as e:
        print(f"Error calculating checksum: {e}")
        return None


def getFloatFromMessage(data):
        format_string = '>BBBBBBBfB'

        if len(data) == 12:
            unpacked_data =struct.unpack(format_string, data)
            float_bytes = unpacked_data[7:11]
            return float_bytes
        else:
            return None


async def send_tcp_request(axis, opcode_high, opcode_low, data = None):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((Server.HOST, Server.PORT))
        print(f"Client connection established with host: {Server.HOST} and port {Server.PORT}")
        logging.info(f"Client connection established with host: {Server.HOST} and port {Server.PORT}")

        if(data == None):
            message = construct_message(axis, opcode_high, opcode_low)
        elif(isinstance(data, float)):
                message = construct_message(axis, opcode_high, opcode_low, data, 'float')
        elif(isinstance(data, int)):
                message = construct_message(axis, opcode_high, opcode_low, data, 'uint8')
        client_socket.send(message)
        response = client_socket.recv(1024)
        client_socket.close()
        print(f"Request for axis: {axis}, with data: {data} sent successfully")

        if(len(response) != 1):
            float_bytes = getFloatFromMessage(response)
            if(float_bytes != None):
                print(f"LOAD POSITION VALUE: {float_bytes[0]}")
                logging.info(f"LOAD POSITION VALUE: {float_bytes[0]}")

            return float_bytes[0]
        return response

    except Exception as e:
        error_message = f"Error sending request for opcode: {hex(opcode_high)} {hex(opcode_low)} with data: {data} ERROR: {e}"
        return error_message


def construct_message(axis, opcode_high, opcode_low, data = None, data_type = None):
    try:   
        if(axis != 0x01 and axis != 0x02 and axis != 0x03):
            logging.error(f"Error sending request for axis {axis} : No such axis defined")
            return f"Error sending request for axis {axis} : No such axis defined"

        if(data == None):
            message_format = '>BBBBBBB'
            message = (Packet_structure.start_byte_1, Packet_structure.start_byte_2, Packet_structure.length, Packet_structure.group_id, 
                        axis, opcode_high, opcode_low)

        elif(data_type == 'float'):
            message_format = '>BBBBBBBf'
            response_length = Packet_structure.length + struct.calcsize('f')
            message = (Packet_structure.start_byte_1, Packet_structure.start_byte_2, response_length, Packet_structure.group_id, 
                        axis, opcode_high, opcode_low, data)

        elif(data_type == 'uint8'):
            message_format = '>BBBBBBBB'
            response_length = Packet_structure.length + struct.calcsize('B')
            message = (Packet_structure.start_byte_1, Packet_structure.start_byte_2, response_length, Packet_structure.group_id,
                        axis, opcode_high, opcode_low, data)

        else:
            print(f"Error: problem in construct_message with data: {data}")
            return

        check_sum = calculate_checksum(struct.pack(message_format, *message))
        message = (*message, check_sum)
        message_format = message_format + 'B'
        packed_message = struct.pack(message_format, *message)
        hexMessage = ' '.join(f'{n:02X}' for n in packed_message)
        print(f"hexMessage: {hexMessage}")
        logging.info(f"hexMessage: {hexMessage}")

        return packed_message
    
    except Exception as e:
        error_message = f"Error sending request for opcode: {hex(opcode_high)} {hex(opcode_low)} with data: {data} ERROR: {e}"
        return error_message


# TODO: Please add dockstrings at module and funciton level.
# TODO: Use type type hints.
# TODO: Literals must be named with a variable.
# TODO: Applicable recomendation from other files.
