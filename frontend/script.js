/*UPDATE VERSION [2]*/

/*
#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: General Purpose Image Classifier & Analyzer
#Assignment: Semester Project
#==================================================
*/

/*
#==================================================
#Start Program
#==================================================
*/
console.log("[CLIENT] script.js Program Start!");

/*
#==================================================
#Global Variables
#==================================================
*/
const imageInput = document.getElementById("imageInput");
const imageToPredict = document.getElementById("imageToPredict");
const result = document.getElementById("result");
result.innerText = "Class: Unknown\nClass Confidence: 0.00%\nAttribute Type: Unknown\nAttribute Value: Unknown\nAttribute Confidence: 0.00%"

/*
#==================================================
#Update Image Function
#==================================================
*/
imageInput.addEventListener("change", function() {
    const file = this.files[0];
    if (file) {
        const fileReader = new FileReader();
        fileReader.onload = function(event) {
            imageToPredict.src = event.target.result;
        };
        fileReader.readAsDataURL(file);
    };
    console.log("[CLIENT] Update Image To Predict!");
});

/*
#==================================================
#Client Prediction API
#==================================================
*/
async function predict()
{
    console.log("[CLIENT] Prediction Request Sent To Server!");
    result.innerText = "Analyzing Image...";
    let file = document.getElementById("imageInput").files[0];
    let formData = new FormData();
    formData.append("file", file)
    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
    });
    let data = await response.json();
    setTimeout(() => {
        let resultString = "Class: " + data.Class + "\nClass Confidence: " + (data["Class Confidence"] * 100).toFixed(2) + "%\n" + " Attribute Type: " + data["Attribute Type"] + "\nAttribute Value: " + data["Attribute Value"] + "\nAttribute Confidence: " + (data["Attribute Confidence"] * 100).toFixed(2) + "%";
        result.innerText = resultString;
        console.log("[CLIENT] Prediction Recieved From Server!");
    }, 3000);
};

/*
==================================================
#Terminate Program
==================================================
*/
console.log("[CLIENT] script.js Program Terminated...")