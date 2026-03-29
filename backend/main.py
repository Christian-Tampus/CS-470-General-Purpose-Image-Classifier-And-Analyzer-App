# main.py (BARE MINIMUM for deployment with static files)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
import uvicorn

app = FastAPI()

# Path to frontend folder
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Serve frontend folder as /static
app.mount("/static", StaticFiles(directory = FRONTEND_DIR), name="static")

# Root route serves index.html
@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")

# Heroku requires dynamic port, use uvicorn to run:
# uvicorn backend.main:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port = port, reload = True)