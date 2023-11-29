import requests

def move_pedestal():
    try:
        url = 'http://localhost:8000/move_pedestal/'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        data = {
            'axis_x': 10.0,
            'axis_y': 10.0
        }
        response = requests.post(url, headers=headers, json=data)
        print('RESPONSE FROM REST API: %s' % response)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    move_pedestal()