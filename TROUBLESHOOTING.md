# Troubleshooting Guide - LoRA Training Environment

## Issues Fixed

1. **WebUI Forge `--medvram` deprecated warning**
   - Removed `--medvram` flag as Forge now auto-manages memory
   - Created optimized launch script: `webui-8gb.sh`

2. **Kohya missing dependencies**
   - Added virtual environment setup
   - Fixed git submodule initialization
   - Automated dependency installation

## Quick Start (Fixed)

### 1. WebUI Forge Launch
```bash
cd stable-diffusion-webui-forge
./webui-8gb.sh  # Optimized for 8GB VRAM
# or
./webui.sh      # Default settings
```

### 2. Kohya LoRA Training
```bash
cd kohya_ss
source venv/bin/activate  # Activate virtual environment
./gui.sh                  # Launch GUI
```

## What's Changed

### Updated Setup Script
- Initializes git submodules automatically
- Creates Python virtual environment for Kohya
- Installs PyTorch with CUDA support
- Installs all Kohya requirements

### New Optimized WebUI Script
- `webui-8gb.sh` - Custom launch script with 8GB VRAM optimizations
- Removes deprecated flags
- Uses memory-efficient attention

### Training Configuration
- `lora_training_config_8gb.toml` - Pre-configured for 8GB VRAM
- Batch size 1, fp16 precision, gradient checkpointing enabled
- Network dim 32 for optimal memory usage

## Memory Optimization Tips

### For Training:
- Always use batch size 1
- Enable gradient checkpointing
- Use fp16 precision (not bf16 on RTX 3070)
- Start with 10-15 images maximum
- Use AdamW8bit optimizer

### For Generation:
- Stick to 512x512 resolution for SD 1.5
- Use SDXL only for special cases
- Close other GPU applications during training

## Troubleshooting

### If Kohya still fails:
```bash
cd kohya_ss
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### If WebUI shows warnings:
- Use `./webui-8gb.sh` instead of `./webui.sh`
- The memory management warnings are normal in Forge

## Setup Complete ✅

Both WebUI Forge and Kohya should now launch without errors!