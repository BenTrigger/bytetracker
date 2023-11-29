import os
import sys
import uvicorn
from pedestal_gateway.config import *

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (pedestal) to the Python path
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)

# Now you can import and run your FastAPI app
from pedestal_gateway.pedestal_app import app

if __name__ == "__main__":
    uvicorn.run("pedestal_gateway.pedestal_app:app", host= Server.HOST, port=Server.PORT, reload=True)
