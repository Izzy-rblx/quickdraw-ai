# Use lightweight Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY . .

# Ensure model directory is included
COPY quickdraw_saved_model/ ./quickdraw_saved_model/

# Expose port for Render/Flask
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
