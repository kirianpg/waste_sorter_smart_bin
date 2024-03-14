from fastapi import FastAPI, File, UploadFile
from project_waste_sorter.frontend.app.app import pred_streamlit
from project_waste_sorter.ml_logic.registry import *

api = FastAPI()

api.state.model = load_model()

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}

# prediction endpoint that calls streamlit function (depends on main file for model)
@api.post("/prediction")
async def pred(img: UploadFile = File(...)):
    img = img.file.read()
    predicted_class = pred_streamlit(img, api.state.model)

    return predicted_class
