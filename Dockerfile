# --- Build Stage (Frontend) ---
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Runtime Stage (Backend) ---
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and models
COPY server/ ./server/
COPY models/ ./models/
COPY config/ ./config/
# COPY app/ ./app/ # Optionally copy if needed

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (Render uses 10000 by default, HF uses 7860)
EXPOSE 7860
ENV PORT=7860

# Run the server
# We use --host 0.0.0.0 and dynamically bind to PORT
CMD uvicorn server.api_server:app --host 0.0.0.0 --port ${PORT:-7860}
