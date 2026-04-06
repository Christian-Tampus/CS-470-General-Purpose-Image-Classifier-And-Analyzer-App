#UPDATE VERSION [49]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: General Purpose Image Classifier & Analyzer
#Assignment: Semester Project
#==================================================

#==================================================
#Program Start
#==================================================

#==================================================
#Import Dependencies
#==================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #Prevent TensorFlow Spamming stdout
import gc
import json
import gdown #If Using Google Drive: pip install gdown
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras import backend as tf_backend
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

#==================================================
#Global Variables
#==================================================
#Important: Path To Base Directory
BASE_DIRECTORY = Path(__file__).parent
#IMPORTANT: CLASS_NAMES Must Be In The Same Order As It Is In The TrainingSet Directory!
CLASS_NAMES = ["Car", "Cat", "Character", "ComputerPart", "Dog", "Fish", "Food", "Human", "Plane", "Tool"]
CAR_MODEL_ATTRIBUTES = ["Acura", "AlfaRomeo", "AstonMartin", "Audi", "BMW", "Bentley", "Bugatti", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Citroen", "Daewoo", "Dodge", "Ferrari", "Fiat", "Ford", "GMC", "Genesis", "Honda", "Hudson", "Hyundai", "Infiniti", "Jaguar", "Jeep", "Kia", "LandRover", "Lexus", "Lincoln", "MG", "Maserati", "Mazda", "MercedesBenz", "Mini", "Mitsubishi", "Nissan", "Oldsmobile", "Peugeot", "Pontiac", "Porsche", "RamTrucks", "Renault", "Saab", "Studebaker", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo"]
CAT_BREED_ATTRIBUTES = ["Abyssinian", "AmericanShortHair", "Bengal", "BritishShortHair", "DevonRex", "ExoticShortHair", "MaineCoon", "NorwegianForestCat", "Persian", "Ragdoll", "RussianBlue", "ScottishFold", "Siamese", "Siberian", "Sphynx"]
DOG_BREED_ATTRIBUTES = ["Beagle", "BostonTerrier", "BullMastiff", "Bulldog", "Chihuahua", "Dalmation", "Doberman", "GermanSheperd", "GoldenRetriever", "GreatDane", "Labrador", "PitBull", "Poodle", "Rottweiler", "ShibaInu"]
HUMAN_RACE_ATTRIBUTES = ["Asian", "Black", "Indian", "White"]
CHARACTER_TYPE_ATTRIBUTES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
PLANE_MODEL_ATTRIBUTES = ["ATR", "Airbus", "Boeing", "C130", "F16"]
FISH_SPECIES_ATTRIBUTES = ["Bangus", "CatFish", "GoldFish", "GreenSpottedPuffer", "Tilapia"]
FOOD_DISH_ATTRIBUTES = ["Burger", "ChickenWings", "Donuts", "FrenchFries", "FriedRice", "HotDog", "IceCream", "Pizza", "Steak", "Taco"]
TOOL_TYPE_ATTRIBUTES = ["Hammer", "Pliers", "ScrewDriver", "Wrench"]
COMPUTERPART_PART_ATTRIBUTES = ["CPU", "Case", "GPU", "HDD", "Motherboard", "Ram"]

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
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
}

#==================================================
#Model Versions
#==================================================
MODEL_VERSIONS = {
    "MAIN_CLASSIFIER_MODEL": 6,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 2,
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": 1,
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
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": 224,
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
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": BASE_DIRECTORY / "AIModels" / (MODEL_NAMES["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5"),
}

#==================================================
#AIModel Google Drive URL
#==================================================
#Url Must Be: https://drive.google.com/uc?id=FILE_ID
#Click Share > Click Copy Link > Example Link: https://drive.google.com/file/d/1CZuOVz1IlRh0Fy9JJ3ZZMZ6RzN1izYzt/view?usp=drive_link
#FILE_ID From Example Link: 1QD-VPbrukO4C_fBnPxbN0lWdPdvnzwif
#Url Example: https://drive.google.com/uc?id=1QD-VPbrukO4C_fBnPxbN0lWdPdvnzwif
MODEL_GOOGLE_DRIVE_URL = {
    "MAIN_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1jrB1__nucIQg5tyIBihsm57w9N7pbpKK",
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1POPEabHDgkz4beEi7OhUb1WxczPVspjB",
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1qeelceGVNmw9IsUVGgsB1nRnzI5OiZLK",
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=15wRQ9645hDYUvyoTAbAuSEYcop_AUROP",
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1N1a0-IkKWGUrMZqBFb9S0RNPrevZLT1w",
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1-rD5ACmb3nFli5FfvXNZe2WiXq8daU0B",
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1oHx98LZgXZjNU8qa0Jht7l-xm9Z5qUDJ",
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1OKmTHvFyzXfQoWdr0TWXBPkgXmDN3NsB",
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1Or8jgAj8DwG7dNO3ZtbJm53Fans9-9Jg",
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1r5Znagedo2VBA0qwX99Qrl_fY9RX-JX5",
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": "https://drive.google.com/uc?id=1SV5S8dbxO7GuqQ8ttOQib5rC10JGveOm",
}

#==================================================
#Preprocess Image Function
#==================================================
def preprocessImage(imagePath, imageSize):
    image = Image.open(imagePath).convert("RGB").resize((imageSize, imageSize))
    imageArray = np.array(image)[np.newaxis, ...] #Add Batch Dimension
    return preprocess_input(imageArray)

#==================================================
#Download Model From Google Drive Function
#==================================================
def downloadModel(modelPath: Path, googleDriveUrl: str):
    if not modelPath.exists():
        modelPath.parent.mkdir(parents = True, exist_ok = True) #Create AIModels Folder If Missing
        gdown.download(googleDriveUrl, str(modelPath), quiet = False)

#==================================================
#Main Function
#==================================================
def main(imagePath):
    imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["MAIN_CLASSIFIER_MODEL"])
    modelPath = MODEL_DIRECTORY["MAIN_CLASSIFIER_MODEL"]
    googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["MAIN_CLASSIFIER_MODEL"]
    downloadModel(modelPath, googleDriveUrl)
    MAIN_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["MAIN_CLASSIFIER_MODEL"])
    classPrediction = MAIN_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
    classPredictionIndex = int(np.argmax(classPrediction))
    predictedClass = CLASS_NAMES[classPredictionIndex]
    predictedClassConfidence = float(classPrediction[classPredictionIndex])
    attributeType = "Unknown"
    predictedAttribute = "Unknown"
    predictedAttributeConfidence = 0
    match predictedClass:
        case "Car":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Car Model"
            predictedAttribute = CAR_MODEL_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Cat":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Cat Breed"
            predictedAttribute = CAT_BREED_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Dog":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Dog Breed"
            predictedAttribute = DOG_BREED_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Human":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Human Race"
            predictedAttribute = HUMAN_RACE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Character":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Character Type"
            predictedAttribute = CHARACTER_TYPE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Plane":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Plane Model"
            predictedAttribute = PLANE_MODEL_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Fish":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Fish Species"
            predictedAttribute = FISH_SPECIES_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Food":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Food Dish"
            predictedAttribute = FOOD_DISH_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "Tool":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Tool Type"
            predictedAttribute = TOOL_TYPE_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
        case "ComputerPart":
            imageArray = preprocessImage(imagePath, MODEL_IMAGE_SIZE["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"])
            modelPath = MODEL_DIRECTORY["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]
            googleDriveUrl = MODEL_GOOGLE_DRIVE_URL["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]
            downloadModel(modelPath, googleDriveUrl)
            COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL = load_model(MODEL_DIRECTORY["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"])
            attributePrediction = COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL.predict(imageArray, verbose = 0)[0]
            attributePredictionIndex = int(np.argmax(attributePrediction))
            attributeType = "Computer Part"
            predictedAttribute = COMPUTERPART_PART_ATTRIBUTES[attributePredictionIndex]
            predictedAttributeConfidence = float(attributePrediction[attributePredictionIndex])
    print(json.dumps({
        "Class": predictedClass,
        "Class Confidence": predictedClassConfidence,
        "Attribute Type": attributeType,
        "Attribute Value": predictedAttribute,
        "Attribute Confidence": predictedAttributeConfidence
    }), flush = True)

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
#Program End
#==================================================