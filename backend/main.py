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
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
import uvicorn

#==================================================
#Global Variables
#==================================================
#Important: Path To frondend Folder
FRONTEND_DIRECTORY = Path(__file__).parent.parent / "frontend"

#==================================================
#Application
#==================================================
app = FastAPI()

# Serve frontend folder as /static
app.mount("/static", StaticFiles(directory = FRONTEND_DIR), name="static")

# Root route serves index.html
@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")

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