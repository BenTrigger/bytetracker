from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field, confloat
from starlette.requests import Request
from pedestal_gateway.config import *
from pedestal_gateway.pedestal_client import send_tcp_request
import asyncio


app = FastAPI()

class Init_pedestal_data(BaseModel):
    speed: float = Field(default= Pedestal.default_speed)
    acceleration: float = Field(default= Pedestal.default_acceleration)

class Move_pedestal_data(BaseModel):
    axis_x: confloat(gt= Pedestal.negative_limit, lt= Pedestal.positive_limit) = Field(default=10.0)  # Pedestal YAW
    axis_y: confloat(gt= Pedestal.negative_limit, lt= Pedestal.positive_limit) = Field(default=10.0)  # Pedestal PITCH

class Get_pedestal_position_response(BaseModel):
    axis_x: float
    axis_y: float

class Move_pedestal_response(BaseModel):
    send_position_response_x: str
    send_position_response_y: str
    update_response_x: str
    update_response_y: str

class axis_on_response(BaseModel):
    axis_on_response_x: str
    axis_on_response_y: str

class Init_pedestal_response(BaseModel):
    set_position_mode_response_x: str
    set_position_mode_response_y: str
    set_position_absolute_response_x: str
    set_position_absolute_response_y: str
    set_acceleration_response_x: str
    set_acceleration_response_y: str
    set_speed_response_x: str
    set_speed_response_y: str
    set_neg_SWLS_response_x: str
    set_neg_SWLS_response_y: str
    set_pos_SWLS_response_x: str
    set_pos_SWLS_response_y: str
    activate_SWLS_response_x: str
    activate_SWLS_response_y: str


@app.get("/", response_model=dict)
async def pedestal_home():
    return {"message": "Welcome to the Magi Interface app. Add '/docs' in the URL to run commands"}


@app.post("/init_pedestal/")
async def init_pedestal(init_pedestal_data: Init_pedestal_data):
    set_position_mode = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.set_position_mode),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.set_position_mode)
    )
    print(f"results for set_position_mode request: {set_position_mode}")
    print("---------------------------------------")

    set_position_absolute = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.set_position_absolute),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.set_position_absolute)
    )
    print(f"results for set_position_absolute request: {set_position_absolute}")
    print("---------------------------------------")

    set_acceleration = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.set_acceleration, init_pedestal_data.acceleration),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.set_acceleration, init_pedestal_data.acceleration)
    )
    print(f"results for set_acceleration request: {set_acceleration}")
    print("---------------------------------------")

    set_speed = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.set_speed, init_pedestal_data.speed),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.set_speed, init_pedestal_data.speed)
    )
    print(f"results for set_speed request: {set_speed}")
    print("---------------------------------------")

    set_neg_SWLS = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.set_neg_SWLS, Pedestal.negative_limit),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.set_neg_SWLS, Pedestal.negative_limit)
    )
    print(f"results for set_neg_SWLS request: {set_neg_SWLS}")
    print("---------------------------------------")

    set_pos_SWLS = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.set_pos_SWLS, Pedestal.positive_limit),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.set_pos_SWLS, Pedestal.positive_limit)
    )
    print(f"results for set_pos_SWLS request: {set_pos_SWLS}")
    print("---------------------------------------")

    activate_SWLS = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.activate_SWLS, Pedestal.activate_SWLS),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.activate_SWLS, Pedestal.activate_SWLS)
    )
    print(f"results for MOT_ActivateSWLS request: {activate_SWLS}")
    print("***************************************")

    return Init_pedestal_response(
        set_position_mode_response_x = set_position_mode[0],
        set_position_mode_response_y = set_position_mode[1],
        set_position_absolute_response_x = set_position_absolute[0],
        set_position_absolute_response_y = set_position_absolute[1],
        set_acceleration_response_x = set_acceleration[0],
        set_acceleration_response_y = set_acceleration[1],
        set_speed_response_x = set_speed[0],
        set_speed_response_y = set_speed[1],
        set_neg_SWLS_response_x = set_neg_SWLS[0],
        set_neg_SWLS_response_y = set_neg_SWLS[1],
        set_pos_SWLS_response_x = set_pos_SWLS[0],
        set_pos_SWLS_response_y = set_pos_SWLS[1],
        activate_SWLS_response_x = activate_SWLS[0],
        activate_SWLS_response_y = activate_SWLS[1]
    )


@app.post("/move_pedestal/")
async def move_pedestal(Move_pedestal_data: Move_pedestal_data):
    send_position_result = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.send_position, Move_pedestal_data.axis_x),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.send_position, Move_pedestal_data.axis_y)
    )
    print(f"results for send_position request: {send_position_result}")
    print("---------------------------------------")
    
    update_result = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.update),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.update)
    )
    print(f"results for update request: {update_result}")
    print("***************************************")

    return Move_pedestal_response(
        send_position_response_x = send_position_result[0],
        send_position_response_y = send_position_result[1],
        update_response_x = send_position_result[0],
        update_response_y = send_position_result[1],
    )


@app.post("/axis_on_off/")
async def axis_on_off(isOn : bool):

    if(isOn):
        axis_on_off_response = await asyncio.gather(
            send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.axis_on),
            send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.axis_on)
        )
        print(f"results for axis_on_off_response request: {axis_on_off_response}")
        print("---------------------------------------")

    else:
        axis_on_off_response = await asyncio.gather(
            send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.axis_off),
            send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.axis_off)
        )
        print(f"results for axis_on_off_response request: {axis_on_off_response}")
        print("***************************************")

    return axis_on_response(
        axis_on_response_x=axis_on_off_response[0],
        axis_on_response_y=axis_on_off_response[1],
    )
    


@app.get("/get_pedestal_position/")
async def get_pedestal_position():
    get_pedestal_position = await asyncio.gather(
        send_tcp_request(Pedestal.axis_yaw, Opcode.mot_high, Opcode.get_load_position),
        send_tcp_request(Pedestal.axis_pitch, Opcode.mot_high, Opcode.get_load_position)
    )
    print(f"results for get_pedestal_position axis_x: {get_pedestal_position[0]}")
    print(f"results for get_pedestal_position axis_y: {get_pedestal_position[1]}")
    print("***************************************")

    return Get_pedestal_position_response(
        axis_x = get_pedestal_position[0],
        axis_y = get_pedestal_position[1]
    )
    

@app.route("/{path:path}", include_in_schema=False)
async def not_found_error(request: Request):
    error_message = f"Endpoint '{request.url.path}' not found"
    return JSONResponse(content={"detail": error_message}, status_code=404)


# TODO: Move literal text default param setting to config file.
# TODO: Use the Layes designe model to redesign the app, i.e:
# - Init module - loads parameters from file, initiates all below mentios objetcs (with params) and runs the IO to outside world
# - Interface layes- abstract base class describes hight level actions performed by the buisness logic layer.
# - Buisness logic layes (if applicable) handles the cases and conditions for running the the hight level actions (usually take the interface layer as parameter).
# - Implimentation layer, impliments the interface layer using avalible libraries.
# - Each layes must be in seporate mudule (file)
