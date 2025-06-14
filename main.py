from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="ML Intelligence API")

class TextInput(BaseModel):
    text: str

@app.post("/predict-text/", operation_id="predict_text")
def predict_text(data: TextInput):
    print("this is data", data)
    print(
    {
        "input": data.text,
        "prediction": "dummy-sentiment-label"
    }
    )
    return {
        "input": data.text,
        "prediction": "dummy-sentiment-label"
    }

@app.post("/predict-image/", operation_id="predict_image")
async def predict_image(file: UploadFile = File(...)):
    print(
        {
            "filename": file.filename,
            "prediction": "dummy-image-class"
        }
    )

    return {
        "filename": file.filename,
        "prediction": "dummy-image-class"
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Intelligence API"}
