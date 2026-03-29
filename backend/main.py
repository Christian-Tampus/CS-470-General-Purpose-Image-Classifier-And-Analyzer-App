# main.py (BARE MINIMUM for deployment)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Serve frontend folder as static
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Root route serves index.html
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# Heroku requires dynamic port, use uvicorn to run:
# uvicorn main:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)