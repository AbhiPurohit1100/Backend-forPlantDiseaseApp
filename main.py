from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


app = FastAPI()

MODEL = tf.keras.models.load_model("models/tomato.h5")
CLASS_NAMES = ["Tomato__early_blight", "Tomato__late_blight", "Tomato__healthy"]

@app.get("/")
async def ping():
    return "message Hello World"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(
    file1: UploadFile = File(...)
):   
    image = read_file_as_image(await file1.read())
    #return {"filename": file.filename, "image_shape": image.shape}
    img_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    return {"class": predicted_class, "confidence": float(100*(predictions[0][np.argmax(predictions)]))}

    pass

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")


