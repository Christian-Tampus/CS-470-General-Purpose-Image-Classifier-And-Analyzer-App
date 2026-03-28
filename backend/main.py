#UPDATE VERSION [9]

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
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

#==================================================
#Model Names
#==================================================
MODEL_NAMES = {
    "MAIN_CLASSIFIER_MODEL": "MAIN_CLASSIFIER_MODEL_VERSION_",
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
}

#==================================================
#Model Versions
#==================================================
MODEL_VERSIONS = {
    "MAIN_CLASSIFIER_MODEL": 5,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 2,
}

#==================================================
#AIModel Directory
#==================================================
MODEL_DIRECTORY = {
    "MAIN_CLASSIFIER_MODEL": Path("AIModels") / (MODEL_NAMES["MAIN_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["MAIN_CLASSIFIER_MODEL"]) + ".h5"),
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": Path("AIModels") / (MODEL_NAMES["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": Path("AIModels") / (MODEL_NAMES["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": Path("AIModels") / (MODEL_NAMES["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": Path("AIModels") / (MODEL_NAMES["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": Path("AIModels") / (MODEL_NAMES["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
}

#==================================================
#Model Image Size
#==================================================
MODEL_IMAGE_SIZE = {
    "MAIN_CLASSIFIER_MODEL": 224,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 128,
}

#==================================================
#Global Variables
#==================================================
MAIN_CLASSIFIER_MODEL = None
CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL = None
CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL = None
DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL = None
HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL = None
CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL = None
#IMPORTANT: CLASS_NAMES Must Be In The Same Order As It Is In The TraingSet Directory!
CLASS_NAMES = ["Car", "Cat", "Character", "Dog", "Human"]
CAR_MODEL_ATTRIBUTES = ["Acura", "AlfaRomeo", "AstonMartin", "Audi", "BMW", "Bentley", "Bugatti", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Citroen", "Daewoo", "Dodge", "Ferrari", "Fiat", "Ford", "GMC", "Genesis", "Honda", "Hudson", "Hyundai", "Infiniti", "Jaguar", "Jeep", "Kia", "LandRover", "Lexus", "Lincoln", "MG", "Maserati", "Mazda", "MercedesBenz", "Mini", "Mitsubishi", "Nissan", "Oldsmobile", "Peugeot", "Pontiac", "Porsche", "RamTrucks", "Renault", "Saab", "Studebaker", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo"]
CAT_BREED_ATTRIBUTES = ["Abyssinian", "AmericanShortHair", "Bengal", "BritishShortHair", "DevonRex", "ExoticShortHair", "MaineCoon", "NorwegianForestCat", "Persian", "Ragdoll", "RussianBlue", "ScottishFold", "Siamese", "Siberian", "Sphynx"]
DOG_BREED_ATTRIBUTES = ["Beagle", "BostonTerrier", "BullMastiff", "Bulldog", "Chihuahua", "Dalmation", "Doberman", "GermanSheperd", "GoldenRetriever", "GreatDane", "Labrador", "PitBull", "Poodle", "Rottweiler", "ShibaInu"]
HUMAN_RACE_ATTRIBUTES = ["Asian", "Black", "Indian", "White"]
CHARACTER_TYPE_ATTRIBUTES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

#==================================================
#Preprocess Image Function
#==================================================
def preprocessImage(image, imageSize):
    image = image.resize((imageSize, imageSize))
    image = np.array(image, dtype = np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis = 0)
    print("[SERVER] Image Processed!")
    return image

#==================================================
#Server Prediction API
#==================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("[SERVER] Request Recieved From Client!")
    global MAIN_CLASSIFIER_MODEL
    global CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL
    global CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL
    global DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL
    global HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL
    global CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL
    if MAIN_CLASSIFIER_MODEL is None:
        print("[SERVER] Loading MAIN_CLASSIFIER_MODEL...")
        MAIN_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["MAIN_CLASSIFIER_MODEL"])
        print("[SERVER] MAIN_CLASSIFIER_MODEL Loaded!")
    fileContents = await file.read()
    image = Image.open(io.BytesIO(fileContents)).convert("RGB")
    imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["MAIN_CLASSIFIER_MODEL"])
    classPrediction = MAIN_CLASSIFIER_MODEL.predict(imageArray)[0]
    classPredictionIndex = np.argmax(classPrediction)
    predictedClass = CLASS_NAMES[classPredictionIndex]
    predictedClassConfidence = float(classPrediction[classPredictionIndex])
    match predictedClass:
        case "Car":
            if CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL is None:
                print("[SERVER] Loading CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL...")
                CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
                print("[SERVER] CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL Loaded!")
            imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = np.argmax(attributePrediction)
            predictedAttribute = CAR_MODEL_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            print("[SERVER] Return Request To Client!")
            return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Car Model", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        case "Cat":
            if CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL is None:
                print("[SERVER] Loading CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL...")
                CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
                print("[SERVER] CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL Loaded!")
            imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = np.argmax(attributePrediction)
            predictedAttribute = CAT_BREED_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            print("[SERVER] Return Request To Client!")
            return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Cat Breed", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        case "Dog":
            if DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL is None:
                print("[SERVER] Loading DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL...")
                DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
                print("[SERVER] DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL Loaded!")
            imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = np.argmax(attributePrediction)
            predictedAttribute = DOG_BREED_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            print("[SERVER] Return Request To Client!")
            return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Dog Breed", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        case "Human":
            if HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL is None:
                print("[SERVER] Loading HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL...")
                HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
                print("[SERVER] HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL Loaded!")
            imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = np.argmax(attributePrediction)
            predictedAttribute = HUMAN_RACE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            print("[SERVER] Return Request To Client!")
            return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Human Race", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
        case "Character":
            if CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL is None:
                print("[SERVER] Loading CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL...")
                CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_DIRECTORY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
                print("[SERVER] CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL Loaded!")
            imageArray = preprocessImage(image, MODEL_IMAGE_SIZE["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = np.argmax(attributePrediction)
            predictedAttribute = CHARACTER_TYPE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
            print("[SERVER] Return Request To Client!")
            return {"Class": predictedClass, "Class Confidence": predictedClassConfidence, "Attribute Type": "Character Type", "Attribute Value": predictedAttribute, "Attribute Confidence": predictedAttributeConfidence}
    return {"Class": "Unknown", "Class Confidence": 0, "Attribute": "Unknown", "Attribute Confidence": 0}

#==================================================
#Terminate Program
#==================================================
print("[SERVER] Main.py Program Terminated...")