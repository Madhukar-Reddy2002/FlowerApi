import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np


model_path = './Model4.h5'
MODEL = tf.keras.models.load_model(model_path)

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

app = FastAPI()

def get_model():
    return MODEL

# Use the CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read():
    return "I am live"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        # Resize the image to match the expected input shape (240, 240)
        image = image.resize((240, 240))
        image = np.array(image)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., content_type="image/jpeg"),
    model: tf.keras.Model = Depends(get_model)
):
    try:
        image = read_file_as_image(await file.read())
        image_batch = np.expand_dims(image, 0)
        prediction = model.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))
        return {
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")