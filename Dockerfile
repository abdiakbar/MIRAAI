# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set work directory
WORKDIR /app

# Install system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs models

# Expose port (default 5000)
EXPOSE 5000

# Default PORT environment variable
ENV PORT=5000

# Command to run the application using Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 120 app:app"]
