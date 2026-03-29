#UPDATE VERSION [33]

#==================================================
#Website Link: https://cs-470-ai-project-app-3c0cc8276da9.herokuapp.com/
#==================================================

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: General Purpose Image Classifier & Analyzer
#Assignment: Semester Project
#==================================================

#==================================================
#IMPORTANT NOTES
#==================================================
#Run Backend Server: uvicorn main:app --reload
#Server Runs At: http://127.0.0.1:8000
#Test EndPoint: http://127.0.0.1:8000/docs

#==================================================
#Start Program
#==================================================
print("[SERVER] Main.py Program Start!")

#==================================================
#Import Dependencies
#==================================================
print("[SERVER] Importing Dependencies...")

import io
import os
import gc
import json
import uvicorn
import tempfile
import subprocess
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from tensorflow.keras.applications.efficientnet import preprocess_input

print("[SERVER] Dependencies Imported!")

#==================================================
#Global Variables
#==================================================
#Important: Path To frondend Folder
FRONTEND_DIRECTORY = Path(__file__).parent.parent / "frontend"

#==================================================
#Application
#==================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #Allow All Origins (Good For localhost Testing)
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
#Important: Server frontend Folder as /static
app.mount("/static", StaticFiles(directory = FRONTEND_DIRECTORY), name = "static")

#==================================================
#Server APIs
#==================================================
@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIRECTORY / "index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("[SERVER] [main.py] Request Recieved From Client!")
    
    #Save Uploaded Image Temporarily
    print("[SERVER] [main.py] Save Uploaded Image Temporarily!")
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".jpg") as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    #Run A Separate Worker Process For Prediction (Execute predictor.py)
    print("[SERVER] [main.py] Run A Separate Worker Process For Prediction (Execute predictor.py)!")
    workerProcessResult = subprocess.run(
        ["python3", "backend/predictor.py", "--image", temp_path],
        capture_output = True,
        text = True,
        check = True,
    )

    #For Debugging The JSON
    print("[SERVER] [main.py] Raw stdout: ", repr(workerProcessResult.stdout))

    if workerProcessResult.returncode != 0:
        print("[SERVER] [main.py] stderr: ", workerProcessResult.stderr)
        raise RuntimeError("[SERVER] [main.py] predictor.py Subprocess Failed!")

    #Remove Temporary File
    print("[SERVER] [main.py] Remove Temporary File!")
    Path(temp_path).unlink()

    #Return Prediction JSON From Worker stdout
    JSON_RESPONSE_CONTENT_DATA = json.loads(workerProcessResult.stdout)
    print("[SERVER] [main.py] JSON_RESPONSE_CONTENT_DATA: ", JSON_RESPONSE_CONTENT_DATA)
    print("[SERVER] [main.py] Return Request To Client!")
    return JSONResponse(content = JSON_RESPONSE_CONTENT_DATA)

#==================================================
#Server Starter
#==================================================
#Important: Heroku Requires Dynamic Port, Use uvicorn To Run: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    serverPort = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host = "0.0.0.0", port = serverPort, reload = True)

#==================================================
#Terminate Program
#==================================================
print("[SERVER] Main.py Program Terminated...")