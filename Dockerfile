# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory to backend
WORKDIR /app/backend

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend files for serving static content
COPY frontend/ ../frontend/

# Expose port 8000 (used by Uvicorn)
EXPOSE 8000

# Command to start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]