import socket
import struct
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import *


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


def get_data_from_message(message, data_type):
    if(data_type == 'float'):
        format_string = '>BBBBBBBfB'
        format_length = 12
    elif(data_type == 'uint'):
        format_string = '>BBBBBBBBB'
        format_length = 9
    if len(message) == format_length:
        unpacked_data =struct.unpack(format_string, message)
        data_bytes = unpacked_data[7:(format_length - 1)]
        return data_bytes
    else:
        print(f"Error in get_data_from_message - unknown data_type {data_type}")
        return None


def action_get_float(opcode_high, opcode_low, axis_id, client_socket):
    try:
        load_position_degrees = 14.0   # MOCK DATA
        
        response_length = Packet_structure.length + struct.calcsize('f')
        check_sum_data = (Packet_structure.start_byte_1, Packet_structure.start_byte_2, response_length, Packet_structure.group_id, axis_id, opcode_high, opcode_low, load_position_degrees)
        check_sum_result = calculate_checksum(check_sum_data)
        response_data = (*check_sum_data, check_sum_result)
        print(f"response_data : {response_data}")

        packed_data = struct.pack('>BBBBBBBfB', *response_data)
        client_socket.send(packed_data)

    except Exception as e:
        print(f"Error in MOT_GetLoadPosition: {e}")
        client_socket.send(Reply.Nack_Execution_Error.to_bytes(1, byteorder='big'))


def action_data(data, data_type, client_socket):
    try:
        if(data_type == 'float'):
            data_bytes = get_data_from_message(data, data_type)
        elif(data_type == 'uint'):
            data_bytes = get_data_from_message(data, data_type)
        else:
            client_socket.send(Reply.Nack_Execution_Error.to_bytes(1, byteorder='big'))
            return
        if(data_bytes != None):
            print(f"Data Value: {data_bytes[0]}")
            # Do something with data...
            client_socket.send(Reply.Ack.to_bytes(1, byteorder='big'))
        else:
            client_socket.send(Reply.Nack_Execution_Error.to_bytes(1, byteorder='big'))

    except Exception as e:
        print(f"Error in action_data: {e}")
        client_socket.send(Reply.Nack_Execution_Error.to_bytes(1, byteorder='big'))


def action_no_data(axis_id, client_socket): 
    try:
        # Do some action...
        client_socket.send(Reply.Ack.to_bytes(1, byteorder='big'))
    except Exception as e:
        print(f"Error in MOT_Update: {e}")
        client_socket.send(Reply.Nack_Execution_Error.to_bytes(1, byteorder='big'))


def process_message(data, client_socket):
    try:    
        start_byte1, start_byte2, length, group_id, axis_id, opcode_high, opcode_low, *data_bytes, checksum = data

        expected_checksum = calculate_checksum(data[:-1])

        if checksum != expected_checksum:
            print("Checksum mismatch.")
            not_ack_message = bytes([Reply.Nack_Wrong_Checksum])
            client_socket.send(Reply.Nack_Wrong_Checksum.to_bytes(1, byteorder='big'))
            return

        hexMessage = ' '.join(f'{n:02X}' for n in data)
        print(f"hexMessage: {hexMessage}")

        if opcode_high == 0x01 and opcode_low == 0x09:
            print("Received MOT_GetLoadPosition command.")
            action_get_float(opcode_high, opcode_low, axis_id, client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x30:
            print("Received MOT_SetAcceleration command.")
            action_data(data, 'float', client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x31:
            print("Received MOT_SetSpeed command.")
            action_data(data, 'float', client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x32:
            print("Received MOT_SendPosition command.")
            action_data(data, 'float', client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x33:
            print("Received MOT_SetActualPosition command.")
            action_data(data, 'float', client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x34:
            print("Received MOT_Update command.")
            action_no_data(axis_id, client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x36:
            print("Received MOT_SetNegSWLS command.")
            action_data(data, 'float', client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x37:
            print("Received MOT_SetPoSWLS command.")
            action_data(data, 'float', client_socket)
        
        elif opcode_high == 0x01 and opcode_low == 0x39:
            print("Received MOT_SetPositionAbsolute command.")
            action_no_data(axis_id, client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x3B:
            print("Received MOT_SetPositionMode command.")
            action_no_data(data, client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x3C:
            print("Received MOT_AxisOn command.")
            action_no_data(data, client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x3D:
            print("Received MOT_AxisOff command.")
            action_no_data(data, client_socket)

        elif opcode_high == 0x01 and opcode_low == 0x46:
            print("Received MOT_ActivateSWLS command.")
            action_data(data, 'uint', client_socket)

        else:
            print(f"Error : Invalid Command")
            client_socket.send(Reply.Nack_Invalid_Command.to_bytes(1, byteorder='big'))

    except Exception as e:
        print(f"Error processing message: {e}")
        client_socket.send(Reply.Nack_Execution_Error.to_bytes(1, byteorder='big'))


async def handle_client(client_socket):
    try:
        data = await loop.run_in_executor(executor, client_socket.recv, 1024)
        if not data:
            return

        process_message(data, client_socket)
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        client_socket.close()

async def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((Server.HOST, Server.PORT))
    server_socket.listen(60)
    print(f"Server is listening on {Server.HOST}:{Server.PORT}")

    while True:
        client_socket, client_address = await loop.run_in_executor(executor, server_socket.accept)
        print(f"Accepted connection from {client_address}")
        asyncio.create_task(handle_client(client_socket))

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = ThreadPoolExecutor()
    loop.run_until_complete(main())
