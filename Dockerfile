FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip && \
    pip install torch gradio matplotlib

# Copy project files
COPY . .

# Default command to run the model generation script
CMD ["python", "generate_model.py"]
