# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete LoRA (Low-Rank Adaptation) training environment for Stable Diffusion, optimized for 8GB VRAM GPUs (RTX 3070 Mobile). The project combines:

1. **Stable Diffusion WebUI Forge** - For image generation and testing trained models
2. **Kohya_ss** - For LoRA training with GUI interface

## Common Commands

### Environment Setup
```bash
# Complete environment setup (run once)
./setup_sd_environment.sh

# Fix Kohya dependencies if issues occur
cd kohya_ss
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Launch Applications
```bash
# Launch WebUI Forge (optimized for 8GB VRAM)
cd stable-diffusion-webui-forge && ./webui-8gb.sh

# Launch WebUI Forge (default settings)
cd stable-diffusion-webui-forge && ./webui.sh

# Launch Kohya LoRA Training GUI
cd kohya_ss && source venv/bin/activate && ./gui.sh
```

### Training Workflow
```bash
# 1. Prepare dataset in training_data/
# 2. Select appropriate config: *_config_8gb.json
# 3. Launch Kohya GUI and load config
# 4. Start training
# 5. Test in WebUI Forge: <lora:model_name:0.8>
```

## Architecture & Code Structure

### Main Components
- **kohya_ss/** - Complete Kohya LoRA training suite with GUI
  - **sd-scripts/** - Core training scripts (git submodule)
  - **kohya_gui/** - Python GUI interface modules
  - **logs/** - Training logs and outputs
  - **outputs/** - Generated LoRA models
- **stable-diffusion-webui-forge/** - WebUI for image generation
  - **models/** - Model storage (checkpoints, LoRAs, VAE, etc.)
  - **extensions-builtin/** - Built-in extensions including ControlNet
- **training_data/** - Organized training datasets
- **docs/** - Comprehensive 8-module training course
- **models/** - Shared model storage
- **outputs/** - Generated images and LoRAs

### Configuration Files (Root Level)
Pre-configured JSON files optimized for 8GB VRAM:
- **jeff_face_lora_config_8gb.json** - Standard LoRA for face training
- **loha_face_config_8gb.json** - LoHa architecture for detailed faces  
- **lokr_style_config_8gb.json** - LoKR for art style transfer
- **advanced_dylora_config_8gb.json** - DyLoRA with multi-rank training
- **advanced_block_weight_config_8gb.json** - Block-weighted training
- **embedding_config_8gb.json** - Textual inversion training

### Key Settings for 8GB VRAM
- **Batch size**: 1 (always)
- **Precision**: fp16 (not bf16 for RTX 3070)
- **Resolution**: 512x512 for SD 1.5
- **Network dim**: 16-32 for optimal memory usage
- **Optimizer**: AdamW8bit for memory efficiency
- **Gradient checkpointing**: enabled

## Training Types & Use Cases

### Face/Character Training
- Use **loha_face_config_8gb.json** (LoHa architecture)
- 10-20 high-quality images
- Conservative learning rate (3e-05)
- Include trigger word like "jeff_person"

### Style Transfer Training  
- Use **lokr_style_config_8gb.json** (LoKR architecture)
- 20-50 images in consistent style
- Higher learning rate (1e-04)
- Include style trigger like "style_name style"

### Advanced Techniques
- **DyLoRA**: Multi-rank training with advanced_dylora_config_8gb.json
- **Block weighting**: Fine-tuned layer control with advanced_block_weight_config_8gb.json

## Hardware Optimization

### Target Hardware
- **Minimum**: GTX 1060 6GB, 16GB RAM, 50GB storage
- **Recommended**: RTX 3070 8GB+, 32GB RAM, 100GB+ SSD
- **Optimized for**: RTX 3070 Mobile 8GB VRAM

### Memory Management
- WebUI Forge auto-manages memory (no --medvram needed)
- Always close other GPU applications during training
- Monitor GPU temperature during long sessions
- Use cache_latents_to_disk if OOM occurs

## Troubleshooting

### Common Issues
- **Kohya fails to start**: Recreate venv and reinstall PyTorch with CUDA
- **WebUI warnings**: Use webui-8gb.sh instead of webui.sh
- **OOM during training**: Reduce network_dim or enable cache_latents_to_disk
- **Training failures**: Check logs/ directory for detailed error messages

### File Paths
- LoRA models: `kohya_ss/outputs/` → `stable-diffusion-webui-forge/models/Lora/`
- Training logs: `kohya_ss/logs/`
- Generated images: `outputs/images/`

## Documentation Structure

The docs/ directory contains an 8-module comprehensive course:
1. **Module 0**: Course overview and prerequisites
2. **Module 1**: LoRA foundations and theory
3. **Module 2**: Basic training workflow
4. **Module 3**: Dataset mastery
5. **Module 4**: Training types (Standard, LoHa, LoKR, DyLoRA)
6. **Module 5**: Advanced techniques
7. **Module 6**: Optimization strategies
8. **Module 7**: Troubleshooting guide
9. **Module 8**: Production pipeline

Additional guides:
- **README.md** - Main entry point and quick start guide
- **training_data/LORA_TRAINING_GUIDE.md** - Face vs Style training comparison
- **TROUBLESHOOTING.md** - Setup troubleshooting and common issues