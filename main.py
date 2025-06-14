from fastapi import FastAPI
from app.api.routes import api_router

app = FastAPI()
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "ML API is running..."}
