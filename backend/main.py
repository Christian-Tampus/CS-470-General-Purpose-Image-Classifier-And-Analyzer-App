#UPDATE VERSION [18]

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
#numpy==1.26.4
#pillow==12.1.1
#tensorflow-cpu==2.20.0

import io
import os
import uvicorn
#import numpy as np
#from PIL import Image
#import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
#from tensorflow.keras.applications.efficientnet import preprocess_input
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
    print("[SERVER] Request Recieved From Client!")
    print("[SERVER] Return Request To Client!")
    return {"Class": "Unknown", "Class Confidence": 0, "Attribute": "Unknown", "Attribute Confidence": 0}

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