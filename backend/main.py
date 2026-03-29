#UPDATE VERSION [26]

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
    print("[SERVER] Request Recieved From Client!")
    
    #Save Uploaded Image Temporarily
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".jpg") as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    #Run A Separate Worker Process For Prediction (Execute predictor.py)
    workerProcessResult = subprocess.run(
        ["python", "backend/predictor.py", "--image", temp_path],
        capture_output = True,
        text = True,
    )

    #Remove Temporary File
    Path(temp_path).unlink()

    #Return Prediction JSON From Worker stdout
    print("[SERVER] Return Request To Client!")
    print(workerProcessResult.stdout)
    return JSONResponse(content = workerProcessResult.stdout)

    #fileContents = await file.read()
    #image = Image.open(io.BytesIO(fileContents)).convert("RGB")
    #imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["MAIN_CLASSIFIER_MODEL"])
    #MAIN_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["MAIN_CLASSIFIER_MODEL"])
    #classPrediction = MAIN_CLASSIFIER_MODEL.predict(imageArray)[0]
    #classPredictionIndex = np.argmax(classPrediction)
    #predictedClass = CLASS_NAMES[classPredictionIndex]
    #predictedClassConfidence = float(classPrediction[classPredictionIndex])
    #match predictedClass:
        #case "Car":
            #imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            #CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            #attributePrediction = CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            #attributePredictionIndex = np.argmax(attributePrediction)
            #predictedAttribute = CAR_MODEL_ATTRIBUTES[attributePredictionIndex]
            #predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            #print("[SERVER] Return Request To Client!")
            #return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Car Model", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        #case "Cat":
            #imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            #CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            #attributePrediction = CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            #attributePredictionIndex = np.argmax(attributePrediction)
            #predictedAttribute = CAT_BREED_ATTRIBUTES[attributePredictionIndex]
            #predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            #print("[SERVER] Return Request To Client!")
            #return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Cat Breed", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        #case "Dog":
            #imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            #DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            #attributePrediction = DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            #attributePredictionIndex = np.argmax(attributePrediction)
            #predictedAttribute = DOG_BREED_ATTRIBUTES[attributePredictionIndex]
            #predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            #print("[SERVER] Return Request To Client!")
            #return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Dog Breed", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        #case "Human":
            #imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            #HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            #attributePrediction = HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            #attributePredictionIndex = np.argmax(attributePrediction)
            #predictedAttribute = HUMAN_RACE_ATTRIBUTES[attributePredictionIndex]
            #predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            #print("[SERVER] Return Request To Client!")
            #return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Human Race", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        #case "Character":
            #imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            #CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            #attributePrediction = CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            #attributePredictionIndex = np.argmax(attributePrediction)
            #predictedAttribute = CHARACTER_TYPE_ATTRIBUTES[attributePredictionIndex]
            #predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            #print("[SERVER] Return Request To Client!")
            #return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Character Type", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
    #return {"Class": "Unknown", "Class Confidence": 0, "Attribute": "Unknown", "Attribute Confidence": 0}

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