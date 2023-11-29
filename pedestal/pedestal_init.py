import requests
from pedestal.pedestal_gateway.config import Pedestal


def start_pedestal():
    try:
        #TODO: Move to param file
        url = 'http://localhost:8000/axis_on_off/?isOn=true'
        headers = {
            'accept': 'application/json'
        }
        response = requests.post(url, headers=headers)
        print('response for axis_on_off: %s' % response)

        #TODO: Move to param file
        url = 'http://localhost:8000/init_pedestal/'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        data = {
            'speed': Pedestal.default_speed,
            'acceleration': Pedestal.default_acceleration
        }
        response = requests.post(url, headers=headers, json=data)
        print('response for init_pedestal: %s' % response)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    start_pedestal()

# TODO: Recomendations:
# - Use pydantic or similar package for loading parameters.
# - Use loggig package insted of print
# - use type hings where possible
# - Install and use black and black extension for vscode to inforece pepe8 convension