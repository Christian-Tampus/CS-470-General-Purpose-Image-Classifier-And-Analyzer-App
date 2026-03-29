#UPDATE VERSION [26]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: General Purpose Image Classifier & Analyzer
#Assignment: Semester Project
#==================================================

#==================================================
#Start Program
#==================================================
print("[SERVER] Predictor.py Program Start!")

#==================================================
#Import Dependencies
#==================================================
print("[SERVER] Predictor.py Importing Dependencies...")

import gc
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras import backend as tf_backend
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

print("[SERVER] Predictor.py Dependencies Imported!")

#==================================================
#Global Variables
#==================================================
#Important: Path To Base Directory
BASE_DIRECTORY = Path(__file__).parent
#IMPORTANT: CLASS_NAMES Must Be In The Same Order As It Is In The TrainingSet Directory!
CLASS_NAMES = ["Car", "Cat", "Character", "Dog", "Human"]
CAR_MODEL_ATTRIBUTES = ["Acura", "AlfaRomeo", "AstonMartin", "Audi", "BMW", "Bentley", "Bugatti", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Citroen", "Daewoo", "Dodge", "Ferrari", "Fiat", "Ford", "GMC", "Genesis", "Honda", "Hudson", "Hyundai", "Infiniti", "Jaguar", "Jeep", "Kia", "LandRover", "Lexus", "Lincoln", "MG", "Maserati", "Mazda", "MercedesBenz", "Mini", "Mitsubishi", "Nissan", "Oldsmobile", "Peugeot", "Pontiac", "Porsche", "RamTrucks", "Renault", "Saab", "Studebaker", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo"]
CAT_BREED_ATTRIBUTES = ["Abyssinian", "AmericanShortHair", "Bengal", "BritishShortHair", "DevonRex", "ExoticShortHair", "MaineCoon", "NorwegianForestCat", "Persian", "Ragdoll", "RussianBlue", "ScottishFold", "Siamese", "Siberian", "Sphynx"]
DOG_BREED_ATTRIBUTES = ["Beagle", "BostonTerrier", "BullMastiff", "Bulldog", "Chihuahua", "Dalmation", "Doberman", "GermanSheperd", "GoldenRetriever", "GreatDane", "Labrador", "PitBull", "Poodle", "Rottweiler", "ShibaInu"]
HUMAN_RACE_ATTRIBUTES = ["Asian", "Black", "Indian", "White"]
CHARACTER_TYPE_ATTRIBUTES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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
#AIModel Directory
#==================================================
#Original Directory Example: Path("AIModels") / (MODEL_NAMES["..."] + str(MODEL_VERSIONS["..."]) + ".h5"
MODEL_DIRECTORY = {
    "MAIN_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["MAIN_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["MAIN_CLASSIFIER_MODEL"]) + ".h5"),
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
}

#==================================================
#Preprocess Image Function
#==================================================
#def preprocessImage(image, imageSize):
    #image = image.resize((imageSize, imageSize))
    #image = np.array(image, dtype = np.float32)
    #image = preprocess_input(image)
    #image = np.expand_dims(image, axis = 0)
    #print("[SERVER] Image Processed!")
    #return image

def preprocessImage(imagePath, imageSize):
    image = Image.open(imagePath).convert("RGB").resize((imageSize, imageSize))
    imageArray = np.array(image)[np.newaxis, ...] #Add Batch Dimension
    return preprocess_input(imageArray)

#==================================================
#Main Function
#==================================================
def main(imagePath):
    print("[SERVER] [predictor.py] Subprocess main(imagePath) Start!")
    imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["MAIN_CLASSIFIER_MODEL"])
    MAIN_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["MAIN_CLASSIFIER_MODEL"])
    classPrediction = MAIN_CLASSIFIER_MODEL.predict(imageArray)[0]
    classPredictionIndex = int(np.argmax(classPrediction))
    predictedClass = CLASS_NAMES[classPredictionIndex]
    predictedClassConfidence = float(classPrediction[classPredictionIndex])
    attributeType = "Unknown"
    predictedAttribute = "Unknown"
    predictedAttributeConfidence = 0
    match predictedClass:
        case "Car":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Car Model"
            predictedAttribute = CAR_MODEL_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Cat":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Cat Breed"
            predictedAttribute = CAT_BREED_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Dog":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Dog Breed"
            predictedAttribute = DOG_BREED_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Human":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Human Race"
            predictedAttribute = HUMAN_RACE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Character":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Character Type"
            predictedAttribute = CHARACTER_TYPE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
    print("[SERVER] Result: ",json.dumps({
        "Class": predictedClass,
        "Class Confidence": predictedClassConfidence,
        "Attribute Type": attributeType,
        "Attribute Value": predictedAttribute,
        "Attribute Confidence": predictedAttributeConfidence
    }))
    print("[SERVER] [predictor.py] Subprocess main(imagePath) Start!")
    return {
        "Class": predictedClass,
        "Class Confidence": predictedClassConfidence,
        "Attribute Type": attributeType,
        "Attribute Value": predictedAttribute,
        "Attribute Confidence": predictedAttributeConfidence
    }

#==================================================
#Script Starter
#==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type = str, required = True)
    args = parser.parse_args()
    main(args.image)

    #Clear Memory Explicitly Before Process Exits
    tf_backend.clear_session()
    gc.collect()

#==================================================
#Terminate Program
#==================================================
print("[SERVER] Predictor.py Program Terminated...")