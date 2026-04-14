# 1. Base Image - Lightweight Python
FROM python:3.12-slim

# 2. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# 3. Working Directory
WORKDIR /app

# 4. System dependencies for LightGBM and PyTorch
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# We split COPY to leverage Docker cache
COPY requirements.txt .

# Install CPU-specific PyTorch to keep image size small
RUN pip install --no-cache-dir torch transformers --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code and model artifacts
# Ensure 'exports/' is present in your local folder!
COPY . .

# 7. Expose the port
EXPOSE 8000

# 8. Start the Application
# Using Gunicorn with Uvicorn workers for production stability
CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --timeout 120
