import uvicorn
import numpy as np
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from CROP import CROP

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("./classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Load label encoder if available, else use fallback dict
try:
    with open("./label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    use_encoder = True
    print(f"Model loaded with label encoder. Classes: {list(le.classes_)}")
except FileNotFoundError:
    use_encoder = False
    print("Model loaded with fallback crop dict.")

CROP_DICT = {
    0: "Apple", 1: "Banana", 2: "Blackgram", 3: "Chickpea",
    4: "Coconut", 5: "Coffee", 6: "Cotton", 7: "Grapes",
    8: "Jute", 9: "Kidneybeans", 10: "Lentil", 11: "Maize",
    12: "Mango", 13: "Mothbeans", 14: "Mungbean", 15: "Muskmelon",
    16: "Orange", 17: "Papaya", 18: "Pigeonpeas", 19: "Pomegranate",
    20: "Rice", 21: "Watermelon",
}

@app.get("/")
def index():
    return {"message": "AI Farmer Crop Recommendation API"}

@app.post("/predict")
def predict_crop(data: CROP):
    d = data.dict()
    features = np.array([[d["N"], d["P"], d["K"], d["temperature"],
                          d["humidity"], d["ph"], d["rainfall"]]])
    prediction = classifier.predict(features)[0]

    if use_encoder:
        crop = le.inverse_transform([prediction])[0].capitalize()
    else:
        crop = CROP_DICT.get(int(prediction), "Unknown")

    return {"result": f"{crop} is the best crop to be cultivated right there"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
