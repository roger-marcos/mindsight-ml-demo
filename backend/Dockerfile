# 1) Base image
FROM python:3.10-slim

# 2) Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Create working directory
WORKDIR /app

# 4) Install system dependencies (required for pillow & torchvision)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 5) Copy dependency list
COPY requirements.txt .

# 6) Install PyTorch CPU + other dependencies
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 7) Copy rest of the app
COPY . .

# 8) Expose port
EXPOSE 10000

# 9) Start FastAPI server (ensure it's running on port 10000)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

