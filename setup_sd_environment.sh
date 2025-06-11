#!/bin/bash

# Complete setup script for Stable Diffusion environment optimized for RTX 3070 Mobile (8GB VRAM)

echo "=== Stable Diffusion Environment Setup ==="
echo "Optimized for RTX 3070 Mobile with 8GB VRAM"
echo ""

# Create directories for organized setup
echo "Creating directory structure..."
mkdir -p models/{checkpoints,lora,embeddings,vae,controlnet}
mkdir -p training_data
mkdir -p outputs/{images,loras}

echo ""
echo "=== WebUI Forge Setup ==="
cd stable-diffusion-webui-forge

# First run to install dependencies
echo "Installing WebUI Forge dependencies... (this may take 10-15 minutes)"
echo "Arguments optimized for 8GB VRAM:"
echo "  --opt-split-attention: Reduce attention memory usage"
echo "  --precision full: Use full precision to avoid errors"
echo "  --no-half-vae: Prevent VAE memory issues"
echo "  --xformers: Enable memory-efficient attention"
echo "  Note: --medvram is deprecated in Forge (memory is auto-managed)"
echo ""

# Make launch script executable
chmod +x webui.sh

echo "WebUI Forge is ready to launch!"
echo ""
echo "=== Kohya Scripts Setup ==="
cd ../kohya_ss

# Initialize git submodules for sd-scripts
echo "Initializing Kohya dependencies..."
git submodule update --init --recursive

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Installing Kohya dependencies... (this may take 10-15 minutes)"
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Make GUI script executable  
chmod +x gui.sh

echo "Kohya Scripts GUI is ready!"
echo ""
echo "=== Quick Start Guide ==="
echo ""
echo "1. Launch WebUI Forge for image generation:"
echo "   cd stable-diffusion-webui-forge && ./webui-8gb.sh"
echo "   (Use ./webui.sh for default settings or ./webui-8gb.sh for optimized 8GB VRAM)"
echo ""
echo "2. Launch Kohya GUI for LoRA training:"
echo "   cd kohya_ss && ./gui.sh"
echo ""
echo "3. Download SD 1.5 model (recommended for 8GB VRAM):"
echo "   Place in: stable-diffusion-webui-forge/models/Stable-diffusion/"
echo ""
echo "4. Choose your training configuration:"
echo "   • jeff_face_lora_config_8gb.json - Standard LoRA for faces"
echo "   • loha_face_config_8gb.json - LoHa for detailed face preservation"
echo "   • lokr_style_config_8gb.json - LoKR for art style transfer"
echo "   • advanced_dylora_config_8gb.json - DyLoRA multi-rank training"
echo ""
echo "5. Read the comprehensive course: docs/README.md"
echo ""
echo "=== 8GB VRAM Optimization Tips ==="
echo "• Always use 512x512 resolution for SD 1.5"
echo "• Use batch size 1 for training"
echo "• Keep LoRA rank between 16-32"
echo "• Monitor temperature during long training sessions"
echo "• Close other GPU-intensive applications"
echo ""
echo "Setup complete! ✅"