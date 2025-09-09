# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .
COPY categories.txt .
COPY quickdraw_saved_model ./quickdraw_saved_model

# Expose port
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
