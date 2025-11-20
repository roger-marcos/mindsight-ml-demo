# 1) Imagen base
FROM python:3.10-slim

# 2) Configuración básica de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Directorio de trabajo
WORKDIR /app

# 4) Dependencias del sistema (Pillow / torchvision)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 5) Copiar requirements.txt
COPY requirements.txt .

# 6) Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 7) Copiar el resto del código
COPY . .

# 8) Exponer puerto del backend
EXPOSE 8000

# 9) Arrancar FastAPI con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

