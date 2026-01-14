FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies (CPU-only PyTorch to avoid large CUDA downloads)
RUN pip install --no-cache-dir --timeout 120 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Install CityFlow from GitHub
RUN pip install git+https://github.com/cityflow-project/CityFlow.git

# Copy project files
COPY . .

# Create results directory
RUN mkdir -p results/logs results/models results/figures

# Default command: run training then evaluation
CMD ["sh", "-c", "python -m src.run_experiment --episodes 150 && python -m src.evaluate_comparison --model_path results/models/presslight_ep150.pth"]