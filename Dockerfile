FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_autoci_v3.txt .
RUN pip install --no-cache-dir -r requirements_autoci_v3.txt

# Install additional AI Agent frameworks
RUN pip install --no-cache-dir \
    langgraph \
    crewai \
    langchain-core \
    langchain-community \
    langchain-openai \
    chromadb \
    weaviate-client \
    celery \
    redis \
    fastapi \
    uvicorn[standard]

# Copy application code
COPY . .

# Make Godot executable
RUN chmod +x godot_engine || true

# Create directories for projects
RUN mkdir -p /app/godot_projects /app/logs

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "autoci_main:app", "--host", "0.0.0.0", "--port", "8000"]