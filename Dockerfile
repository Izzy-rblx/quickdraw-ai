# Use Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files, including model and categories
COPY . .

# Expose port for Render
ENV PORT=10000
EXPOSE 10000

# Run Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
