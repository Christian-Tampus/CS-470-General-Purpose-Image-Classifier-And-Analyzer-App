# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory to backend
WORKDIR /app/backend

# Copy the requirements.txt from root (not from backend/)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend code
COPY frontend/ ../frontend/

# Expose the port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]