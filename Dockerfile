# Use an official lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for TensorFlow, h5py, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including model + categories)
COPY . .

# Expose the Hugging Face Space port
EXPOSE 7860

# Run with gunicorn (production server)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
