from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from Backend.MNISTMODEL import MNISTModel
from PIL import Image
from io import BytesIO
import numpy as np
import os

app = FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR,"../Frontend"))

API_KEY = os.getenv("API_KEY")

def ImageProcess(image):
    gray = Image.open(BytesIO(image)).convert("L")

    resized_gray = gray.resize((28,28),resample=Image.LANCZOS)

    return np.array(resized_gray)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.abspath(os.path.join(FRONTEND_DIR,"MNIST.html")))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    model = MNISTModel()
    image = ImageProcess(contents)
    Aj = model.predict_model(image)

    prediction = int(np.argmax(Aj))
    probability = float(np.max(Aj))

    result = {"0":Aj[0,0],
                  "1":Aj[1,0],
                  "2":Aj[2,0],
                  "3":Aj[3,0],
                  "4":Aj[4,0],
                  "5":Aj[5,0],
                  "6":Aj[6,0],
                  "7":Aj[7,0],
                  "8":Aj[8,0],
                  "9":Aj[9,0]}
    
    return {"Prediction":prediction,
            "Confidence":probability,
            "Probabilities":result}
