print("[SERVER] Starting minimal test app...")

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    print("[SERVER] Root endpoint hit!")
    return {"message": "Server is working!"}

@app.get("/test")
async def test():
    print("[SERVER] Test endpoint hit!")
    return {"status": "OK"}