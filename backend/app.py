from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_image_bytes


app = FastAPI(title="Mindsight ML Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "mindsight-ml-demo-api"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class for an uploaded image.

    Expects a file upload (image). Returns:
      - predicted class
      - probabilities per class
    """
    data = await file.read()
    pred_class, probs = predict_image_bytes(data)
    return {
        "pred_class": pred_class,
        "probs": probs,
    }

