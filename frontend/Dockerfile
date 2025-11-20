# 1) Imagen base
FROM python:3.10-slim

# 2) Configuración básica
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3) Instalar dependencias del sistema (por si Pillow las necesita)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 4) Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5) Copiar el código
COPY . .

# 6) Exponer puerto de Streamlit
EXPOSE 8501

# 7) Comando de arranque
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
