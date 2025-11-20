# Mindsight ML Challenge – CIFAR-3 Vision Pipeline

Technical challenge solution for the **AI Engineer / Backend Developer (Barcelona)** role at **Mindsight Ventures**.

This repository contains a small but complete ML prototype:

- **Model training** on a subset of CIFAR-10 (3 classes: `airplane`, `automobile`, `ship`)
- **Evaluation** with accuracy, classification report, and confusion matrix
- **Inference API** using FastAPI (`/health`, `/predict`)
- **Frontend demo** using Streamlit (image upload + prediction + probability chart)
- Optional: **Dockerfiles** for backend & frontend + `docker-compose.yml` for local orchestration and easy deployment

The focus is on:

- Clear architecture
- Well-structured code
- End-to-end pipeline: *train → save → serve → demo UI*
- Easy local reproduction

---

## 1. Project structure

```
mindsight-ml-demo/
│
├── backend/
│   ├── app.py              # FastAPI app (health + /predict)
│   ├── config.py           # Shared config (paths, constants, seed)
│   ├── data.py             # CIFAR-10 loading, filtering to 3 classes, dataloaders
│   ├── train.py            # Training loop (ResNet18, transfer learning)
│   ├── eval.py             # Evaluation on test set + confusion matrix
│   ├── inference.py        # Model loading & inference helper
│   ├── requirements.txt    # Backend dependencies
│   ├── Dockerfile          # Backend Docker image (optional)
│   └── models/             # (ignored in repo) trained model file lives here
│
├── frontend/
│   ├── app.py              # Streamlit UI (image upload + results)
│   └── requirements.txt    # Frontend dependencies
│
├── docker-compose.yml      # Optional: run backend + frontend together
└── README.md
```
Note: data/ and models/ are intentionally excluded from version control.
The dataset is downloaded automatically and the model is generated via training (or provided as a Release asset).

## 2. Dataset

This project uses the CIFAR-10 dataset.

- Official source: https://www.cs.toronto.edu/~kriz/cifar.html
- Only 3 classes are used: airplane, automobile, ship.

You do not need to download the dataset manually.
The training script automatically downloads and caches it under backend/data/.

## 3. Local environment setup (no Docker)
### 3.1 Backend setup
```
cd backend
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Frontend setup
```
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Training the model

Run the training script:
```
cd backend
python3 train.py
```

This will:
1. Download CIFAR-10
2. Filter the dataset to the 3 selected classes
3. Fine-tune a pretrained ResNet18
4. Save the best checkpoint to:

backend/models/resnet18_cifar3_best.pt


Training logs include:

- Train accuracy / loss
- Validation accuracy / loss
- Best model saving events

## 5. Evaluating the model
```
cd backend
python3 eval.py
```

This produces:
- Test accuracy
- Classification report (precision, recall, F1 by class)
- Confusion matrix
- Saved figure under:

`/backend/confusion_matrix.png`

## 6. Running the backend API (FastAPI)

Start the API:
```
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Endpoints

`/GET /health`
Returns a basic status JSON.

`/POST /predict`
Send an image (multipart/form-data):
```
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.png"
```

Response example:
```
{
  "pred_class": "airplane",
  "probs": {
    "airplane": 0.87,
    "automobile": 0.05,
    "ship": 0.08
  }
}
```
## 7. Running the frontend demo (Streamlit)
```
cd frontend
streamlit run app.py
```

The UI will:
- Allow uploading an image
- Call the backend /predict endpoint
- Show the predicted class
- Visualize probabilities in a bar chart

Default backend URL:
`/http://localhost:8000/predict`

## 8. Docker support (optional)
### 8.1 Backend Dockerfile

The backend Dockerfile builds a CPU-only FastAPI service:
```
docker build -t mindsight-backend ./backend
docker run -p 8000:8000 mindsight-backend
```
### 8.2 Frontend Dockerfile
Build and run Streamlit UI:
```
docker build -t mindsight-frontend ./frontend
docker run -p 8501:8501 mindsight-frontend
```

### 8.3 docker-compose (orchestrates both)

To run backend + frontend together:
```
docker compose up --build
```

Then:
- UI → http://localhost:8501
- API → http://localhost:8000/health

## 9. Tiny cloud deployment (Render)

The backend is cloud-ready and deployable on Render, Railway, AWS, GCP, or any platform supporting Docker.

Typical Render deployment:

1. Push this repository to GitHub
2. Create a new Web Service
3. Set Root Directory to backend/
4. Use the provided Dockerfile
5. Expose port 8000
6. Once deployed, your API will be available at something like:

/`https://<service-name>.onrender.com`

You can then update the frontend UI to point API_URL to this cloud endpoint.

## 10. Possible extensions
- Add more CIFAR-10 classes
- Tune learning rate, schedulers, data augmentation
- Add feature importance / Grad-CAM
- Deploy both frontend + backend as separate cloud services
- Add authentication / API key protection
- Containerize and deploy both services in the cloud with HTTPS

## 11. Notes
This project is intentionally scoped to fit the expectations of a 4–8 hour technical challenge, while still demonstrating:
- ML training pipeline
- Model persistence and reuse
- Clean API design
- Frontend integration
- Optional Dockerization
- Cloud deployment capability
