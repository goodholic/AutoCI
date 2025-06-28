#!/bin/bash

# Advanced Model Installation Script for AutoCI
# Supports: Llama 3.1 70B/405B, Qwen2.5 72B, DeepSeek V2.5
# Optimized for 32GB RAM using quantization

set -e

echo "=== AutoCI Advanced Model Installation ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 30 ]; then
        log_warning "System has ${TOTAL_RAM}GB RAM. 32GB+ recommended for optimal performance."
    else
        log_info "RAM check passed: ${TOTAL_RAM}GB available"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 200 ]; then
        log_error "Insufficient disk space. Need at least 200GB free, found ${AVAILABLE_SPACE}GB"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python version: $PYTHON_VERSION"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # System packages
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        git-lfs \
        wget \
        curl \
        python3-pip \
        python3-venv \
        nvidia-cuda-toolkit \
        libblas-dev \
        liblapack-dev
    
    # Python packages
    pip install --upgrade pip
    pip install \
        torch \
        transformers \
        accelerate \
        bitsandbytes \
        sentencepiece \
        protobuf \
        safetensors \
        einops \
        flash-attn \
        optimum \
        auto-gptq \
        exllama \
        llama-cpp-python \
        vllm
}

# Install Ollama with advanced model support
install_ollama_advanced() {
    log_info "Installing Ollama with advanced model support..."
    
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Configure for large models
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo cat > /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_MODELS=/var/lib/ollama/models"
Environment="CUDA_VISIBLE_DEVICES=0"
LimitNOFILE=1048576
LimitMEMLOCK=infinity
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
}

# Setup model configurations for 32GB RAM
setup_model_configs() {
    log_info "Creating optimized model configurations..."
    
    mkdir -p configs
    
    # Llama 3.1 70B configuration (4-bit quantization)
    cat > configs/llama3.1-70b-config.json << 'EOF'
{
    "model_id": "meta-llama/Llama-3.1-70B-Instruct",
    "quantization": "4bit",
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    "max_memory": {
        "0": "30GB",
        "cpu": "30GB"
    },
    "offload_folder": "./offload",
    "offload_state_dict": true,
    "low_cpu_mem_usage": true
}
EOF

    # Qwen2.5 72B configuration (4-bit quantization)
    cat > configs/qwen2.5-72b-config.json << 'EOF'
{
    "model_id": "Qwen/Qwen2.5-72B-Instruct",
    "quantization": "4bit",
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    "max_memory": {
        "0": "30GB",
        "cpu": "30GB"
    },
    "device_map": "auto",
    "torch_dtype": "float16"
}
EOF

    # DeepSeek V2.5 configuration (optimized for coding)
    cat > configs/deepseek-v2.5-config.json << 'EOF'
{
    "model_id": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "quantization": "4bit",
    "load_in_4bit": true,
    "rope_scaling": {
        "type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 8192
    },
    "max_memory": {
        "0": "30GB",
        "cpu": "30GB"
    },
    "torch_dtype": "float16",
    "trust_remote_code": true
}
EOF
}

# Create model loader script
create_model_loader() {
    log_info "Creating advanced model loader..."
    
    cat > modules/advanced_model_loader.py << 'EOF'
import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModelLoader:
    """Advanced model loader with quantization support for 32GB RAM"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Load model with 4-bit quantization"""
        logger.info(f"Loading {self.config['model_id']} with 4-bit quantization...")
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_id'],
            trust_remote_code=self.config.get('trust_remote_code', False)
        )
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_id'],
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=self.config.get('max_memory', {"0": "30GB", "cpu": "30GB"}),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=self.config.get('trust_remote_code', False),
            offload_folder=self.config.get('offload_folder', './offload'),
            offload_state_dict=self.config.get('offload_state_dict', True)
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info("Model loaded successfully!")
        
    def generate(self, prompt: str, max_length: int = 2048, temperature: float = 0.7) -> str:
        """Generate text with the loaded model"""
        if not self.pipeline:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        outputs = self.pipeline(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs[0]['generated_text']
    
    def unload_model(self):
        """Unload model to free memory"""
        del self.model
        del self.tokenizer
        del self.pipeline
        torch.cuda.empty_cache()
        logger.info("Model unloaded and memory cleared")

class MultiModelManager:
    """Manage multiple large models efficiently on 32GB RAM"""
    
    def __init__(self):
        self.models = {}
        self.active_model = None
        
    def load_model(self, model_name: str, config_path: str):
        """Load a specific model"""
        if self.active_model:
            logger.info(f"Unloading {self.active_model} to make room...")
            self.unload_active_model()
            
        logger.info(f"Loading {model_name}...")
        loader = AdvancedModelLoader(config_path)
        loader.load_model()
        self.models[model_name] = loader
        self.active_model = model_name
        
    def unload_active_model(self):
        """Unload the currently active model"""
        if self.active_model and self.active_model in self.models:
            self.models[self.active_model].unload_model()
            del self.models[self.active_model]
            self.active_model = None
            
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with the active model"""
        if not self.active_model:
            raise ValueError("No active model loaded")
            
        return self.models[self.active_model].generate(prompt, **kwargs)
    
    def switch_model(self, model_name: str, config_path: str):
        """Switch to a different model"""
        self.load_model(model_name, config_path)

# Usage example
if __name__ == "__main__":
    manager = MultiModelManager()
    
    # Load Llama 3.1 70B for general tasks
    manager.load_model("llama3.1-70b", "configs/llama3.1-70b-config.json")
    response = manager.generate("Create a game design document for a puzzle game")
    print(response)
    
    # Switch to DeepSeek for coding
    manager.switch_model("deepseek-v2.5", "configs/deepseek-v2.5-config.json")
    code = manager.generate("Write a Godot script for player movement")
    print(code)
EOF
}

# Create Ollama model installers
create_ollama_installers() {
    log_info "Creating Ollama model installation scripts..."
    
    # Llama 3.1 70B installer
    cat > install_llama3_1_70b.sh << 'EOF'
#!/bin/bash
echo "Installing Llama 3.1 70B (quantized)..."
ollama pull llama3.1:70b-instruct-q4_0
echo "Testing model..."
ollama run llama3.1:70b-instruct-q4_0 "Hello, introduce yourself"
EOF
    
    # Qwen2.5 72B installer
    cat > install_qwen2_5_72b.sh << 'EOF'
#!/bin/bash
echo "Installing Qwen2.5 72B (quantized)..."
ollama pull qwen2.5:72b-instruct-q4_0
echo "Testing model..."
ollama run qwen2.5:72b-instruct-q4_0 "Hello, introduce yourself"
EOF
    
    # DeepSeek V2.5 installer
    cat > install_deepseek_v2_5.sh << 'EOF'
#!/bin/bash
echo "Installing DeepSeek V2.5 (quantized)..."
ollama pull deepseek-v2.5:latest
echo "Testing model..."
ollama run deepseek-v2.5:latest "Write a hello world in Python"
EOF
    
    chmod +x install_*.sh
}

# Create optimized AutoCI integration
create_autoci_integration() {
    log_info "Creating AutoCI integration for advanced models..."
    
    cat > modules/autoci_advanced_models.py << 'EOF'
"""
AutoCI Advanced Model Integration
Supports: Llama 3.1 70B/405B, Qwen2.5 72B, DeepSeek V2.5
Optimized for 32GB RAM
"""

import asyncio
import json
from typing import Dict, Any, Optional
from enum import Enum
import logging
from advanced_model_loader import MultiModelManager

logger = logging.getLogger(__name__)

class ModelType(Enum):
    LLAMA_3_1_70B = "llama3.1-70b"
    LLAMA_3_1_405B = "llama3.1-405b"  # Requires special handling
    QWEN_2_5_72B = "qwen2.5-72b"
    DEEPSEEK_V2_5 = "deepseek-v2.5"

class AdvancedAutoCI:
    """AutoCI with advanced model support"""
    
    def __init__(self):
        self.model_manager = MultiModelManager()
        self.current_model = None
        self.model_configs = {
            ModelType.LLAMA_3_1_70B: "configs/llama3.1-70b-config.json",
            ModelType.QWEN_2_5_72B: "configs/qwen2.5-72b-config.json",
            ModelType.DEEPSEEK_V2_5: "configs/deepseek-v2.5-config.json"
        }
        
    async def select_best_model(self, task: str) -> ModelType:
        """Select the best model for a given task"""
        task_lower = task.lower()
        
        # DeepSeek for coding tasks
        if any(keyword in task_lower for keyword in ['code', 'script', 'function', 'class', 'debug', 'implement']):
            return ModelType.DEEPSEEK_V2_5
            
        # Qwen for complex reasoning and multi-language
        elif any(keyword in task_lower for keyword in ['design', 'architecture', 'complex', 'analyze', '한국어']):
            return ModelType.QWEN_2_5_72B
            
        # Llama for general tasks
        else:
            return ModelType.LLAMA_3_1_70B
            
    async def process_request(self, user_input: str) -> str:
        """Process user request with optimal model selection"""
        # Select best model
        best_model = await self.select_best_model(user_input)
        
        # Switch to selected model if needed
        if self.current_model != best_model:
            logger.info(f"Switching to {best_model.value} for this task...")
            self.model_manager.switch_model(
                best_model.value,
                self.model_configs[best_model]
            )
            self.current_model = best_model
            
        # Generate response
        try:
            response = self.model_manager.generate(
                user_input,
                max_length=4096,
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
            
    async def create_game_with_advanced_ai(self, game_description: str):
        """Create a complete game using advanced models"""
        tasks = [
            ("Design game architecture", ModelType.QWEN_2_5_72B),
            ("Implement core mechanics", ModelType.DEEPSEEK_V2_5),
            ("Create game assets list", ModelType.LLAMA_3_1_70B),
            ("Write GDScript code", ModelType.DEEPSEEK_V2_5),
            ("Optimize and polish", ModelType.LLAMA_3_1_70B)
        ]
        
        results = {}
        for task_desc, model_type in tasks:
            logger.info(f"Processing: {task_desc} with {model_type.value}")
            
            # Switch model
            self.model_manager.switch_model(
                model_type.value,
                self.model_configs[model_type]
            )
            
            # Generate result
            prompt = f"{task_desc} for: {game_description}"
            result = self.model_manager.generate(prompt)
            results[task_desc] = result
            
        return results

# Integration with existing AutoCI
class AutoCIWithAdvancedModels:
    """Enhanced AutoCI with advanced model support"""
    
    def __init__(self, use_advanced_models=True):
        self.use_advanced_models = use_advanced_models
        self.advanced_ai = AdvancedAutoCI() if use_advanced_models else None
        
    async def process_user_input(self, user_input: str) -> str:
        """Process user input with advanced models when appropriate"""
        if self.use_advanced_models and self._should_use_advanced_model(user_input):
            return await self.advanced_ai.process_request(user_input)
        else:
            # Fall back to original Llama 7B for simple tasks
            return await self._process_with_base_model(user_input)
            
    def _should_use_advanced_model(self, user_input: str) -> bool:
        """Determine if advanced models are needed"""
        # Use advanced models for complex tasks
        complex_indicators = [
            'complex', 'advanced', 'sophisticated', 'detailed',
            'full game', 'complete implementation', 'optimize',
            'ai system', 'multiplayer', 'procedural generation'
        ]
        return any(indicator in user_input.lower() for indicator in complex_indicators)
        
    async def _process_with_base_model(self, user_input: str) -> str:
        """Process with original Llama 7B model"""
        # Original implementation
        pass
EOF
}

# Create memory optimization script
create_memory_optimizer() {
    log_info "Creating memory optimization utilities..."
    
    cat > modules/memory_optimizer.py << 'EOF'
"""Memory optimization for running large models on 32GB RAM"""

import torch
import gc
import os
import psutil
from typing import Dict, Any

class MemoryOptimizer:
    """Optimize memory usage for large models"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage"""
        # System RAM
        ram = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
            
        return {
            'ram_used_gb': ram.used / 1024**3,
            'ram_available_gb': ram.available / 1024**3,
            'ram_percent': ram.percent,
            **gpu_info
        }
        
    @staticmethod
    def optimize_before_load():
        """Optimize memory before loading a model"""
        # Clear Python garbage
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Set memory fraction for PyTorch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            
    @staticmethod
    def enable_memory_efficient_attention():
        """Enable memory efficient attention mechanisms"""
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Enable Flash Attention if available
        try:
            import flash_attn
            os.environ['USE_FLASH_ATTENTION'] = '1'
        except ImportError:
            pass
            
    @staticmethod
    def setup_quantization_cache():
        """Setup cache directory for quantized models"""
        cache_dir = os.path.expanduser("~/.cache/autoci/quantized_models")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        return cache_dir

# Usage
optimizer = MemoryOptimizer()
optimizer.optimize_before_load()
optimizer.enable_memory_efficient_attention()
print(optimizer.get_memory_info())
EOF
}

# Create benchmark script
create_benchmark_script() {
    log_info "Creating model benchmark script..."
    
    cat > benchmark_models.py << 'EOF'
#!/usr/bin/env python3
"""Benchmark different models for AutoCI tasks"""

import time
import json
from datetime import datetime
from modules.advanced_model_loader import MultiModelManager
from modules.memory_optimizer import MemoryOptimizer

def benchmark_model(model_name: str, config_path: str, test_prompts: list):
    """Benchmark a single model"""
    optimizer = MemoryOptimizer()
    manager = MultiModelManager()
    
    results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'memory_before': optimizer.get_memory_info(),
        'benchmarks': []
    }
    
    # Load model
    start_time = time.time()
    manager.load_model(model_name, config_path)
    load_time = time.time() - start_time
    results['load_time'] = load_time
    
    # Test each prompt
    for prompt in test_prompts:
        start_time = time.time()
        response = manager.generate(prompt, max_length=1024)
        gen_time = time.time() - start_time
        
        results['benchmarks'].append({
            'prompt': prompt[:50] + '...',
            'generation_time': gen_time,
            'response_length': len(response),
            'tokens_per_second': len(response.split()) / gen_time
        })
    
    # Memory after
    results['memory_after'] = optimizer.get_memory_info()
    
    # Cleanup
    manager.unload_active_model()
    
    return results

def main():
    test_prompts = [
        "Write a player movement script for Godot 4",
        "Design a puzzle game mechanic",
        "Implement an enemy AI state machine",
        "Create a game save/load system",
        "Optimize this code for better performance: [sample code]"
    ]
    
    models_to_test = [
        ("llama3.1-70b", "configs/llama3.1-70b-config.json"),
        ("qwen2.5-72b", "configs/qwen2.5-72b-config.json"),
        ("deepseek-v2.5", "configs/deepseek-v2.5-config.json")
    ]
    
    all_results = []
    
    for model_name, config_path in models_to_test:
        print(f"\nBenchmarking {model_name}...")
        try:
            result = benchmark_model(model_name, config_path, test_prompts)
            all_results.append(result)
            print(f"✓ Completed {model_name}")
        except Exception as e:
            print(f"✗ Failed {model_name}: {str(e)}")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nBenchmark complete! Results saved to benchmark_results.json")

if __name__ == "__main__":
    main()
EOF
    chmod +x benchmark_models.py
}

# Main installation
main() {
    log_info "Starting advanced model installation for AutoCI..."
    
    check_requirements
    install_dependencies
    install_ollama_advanced
    setup_model_configs
    create_model_loader
    create_ollama_installers
    create_autoci_integration
    create_memory_optimizer
    create_benchmark_script
    
    log_info "Creating model download helper..."
    cat > download_advanced_models.sh << 'EOF'
#!/bin/bash
echo "Downloading quantized models optimized for 32GB RAM..."
echo "This will download approximately 150GB of data."
echo ""
echo "Choose which models to download:"
echo "1) Llama 3.1 70B (4-bit) - ~35GB"
echo "2) Qwen2.5 72B (4-bit) - ~36GB"
echo "3) DeepSeek V2.5 (4-bit) - ~50GB"
echo "4) All models"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1) ./install_llama3_1_70b.sh ;;
    2) ./install_qwen2_5_72b.sh ;;
    3) ./install_deepseek_v2_5.sh ;;
    4) 
        ./install_llama3_1_70b.sh
        ./install_qwen2_5_72b.sh
        ./install_deepseek_v2_5.sh
        ;;
    *) echo "Invalid choice" ;;
esac
EOF
    chmod +x download_advanced_models.sh
    
    log_info "=== Installation Complete ==="
    log_info "Next steps:"
    log_info "1. Run ./download_advanced_models.sh to download models"
    log_info "2. Run python benchmark_models.py to test performance"
    log_info "3. Models will automatically switch based on task complexity"
    log_info ""
    log_info "Memory optimization tips for 32GB RAM:"
    log_info "- Only one large model loads at a time"
    log_info "- Models use 4-bit quantization"
    log_info "- Automatic model switching based on task"
    log_info "- CPU offloading for larger models"
}

# Run main installation
main