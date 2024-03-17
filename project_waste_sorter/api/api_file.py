from fastapi import FastAPI, File, UploadFile
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from project_waste_sorter.ml_logic.registry import *
from google.cloud import storage
from io import BytesIO
from PIL import Image

app = FastAPI()

app.state.model = load_model()  # Load the model at startup

@app.get("/")
def index():
    return {"message": "API connected"}

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        uploaded_image = Image.open(BytesIO(contents))
        # Resize to 224 x 224
        uploaded_image = uploaded_image.resize((224, 224))

        # Convert the image pixels to a numpy array
        image = img_to_array(uploaded_image)

        # Reshape data for the model
        image = image.reshape((1, 224, 224, 3))

        # Prepare the image for the VGG model
        image = preprocess_input(image)

        # Run prediction
        model_prediction = app.state.model.predict(image)

        # Format the result
        INDEX_TO_CATEGORIES = {v: k for k, v in CATEGORIES_MAP.items()}
        predictions_with_categories = [(INDEX_TO_CATEGORIES[i], float(prob)) for i, prob in enumerate(model_prediction[0])]
        predictions_with_categories.sort(key=lambda x: x[1], reverse=True)
        best_prediction = predictions_with_categories[0]

        return dict(result = best_prediction)
    except Exception as e:
        return {"error": str(e)}
