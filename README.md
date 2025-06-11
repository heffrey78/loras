# LoRA Training Environment

> **Complete LoRA training setup for Stable Diffusion, optimized for RTX 3070 Mobile (8GB VRAM)**

A comprehensive, production-ready environment for training LoRA (Low-Rank Adaptation) models with Stable Diffusion. Includes educational course materials, optimized configurations, and automated setup scripts.

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Clone this repository
git clone https://github.com/heffrey78/loras.git
cd loras

# Run the complete setup (installs all dependencies)
./setup_sd_environment.sh
```

### 2. Launch Applications
```bash
# For image generation and testing
cd stable-diffusion-webui-forge && ./webui-8gb.sh

# For LoRA training
cd kohya_ss && source venv/bin/activate && ./gui.sh
```

### 3. Start Training
1. Prepare your dataset in `training_data/`
2. Choose a configuration from the root directory
3. Load it in Kohya GUI and start training
4. Test your LoRA in WebUI Forge: `<lora:model_name:0.8>`

## 📚 What's Included

### 🎓 Comprehensive 8-Module Course
- **[Module 0](docs/00_course_overview.md)**: Course overview and prerequisites
- **[Module 1](docs/01_module_foundations.md)**: LoRA foundations and theory
- **[Module 2](docs/02_module_basic_training.md)**: Your first LoRA training
- **[Module 3](docs/03_module_dataset_mastery.md)**: Dataset preparation mastery
- **[Module 4](docs/04_module_training_types.md)**: LoRA architecture deep dive
- **[Module 5](docs/05_module_advanced_techniques.md)**: Advanced training techniques
- **[Module 6](docs/06_module_optimization.md)**: Performance optimization
- **[Module 7](docs/07_module_troubleshooting.md)**: Troubleshooting guide
- **[Module 8](docs/08_module_production_pipeline.md)**: Production pipeline

**Start here:** [📖 Course Overview](docs/README.md)

### ⚙️ Pre-Configured Training Setups

All configurations optimized for **8GB VRAM (RTX 3070 Mobile)**:

| Configuration | Best For | Architecture | Network Dim | Learning Rate |
|---------------|----------|--------------|-------------|---------------|
| `jeff_face_lora_config_8gb.json` | **Face/Character training** | Standard LoRA | 32 | 5e-05 |
| `loha_face_config_8gb.json` | **Detailed face preservation** | LoHa | 32 | 3e-05 |
| `lokr_style_config_8gb.json` | **Art style transfer** | LoKR | 64 | 1e-04 |
| `advanced_dylora_config_8gb.json` | **Multi-rank training** | DyLoRA | 32 | 5e-05 |
| `advanced_block_weight_config_8gb.json` | **Fine-tuned control** | Block-weighted | 32 | 5e-05 |
| `embedding_config_8gb.json` | **Textual inversion** | Embeddings | - | 5e-03 |

### 🛠️ Core Components

- **[Kohya_ss](kohya_ss/)** - Complete LoRA training suite with GUI
- **[Stable Diffusion WebUI Forge](stable-diffusion-webui-forge/)** - Image generation and testing
- **[Training Data](training_data/)** - Organized dataset structure
- **[Documentation](docs/)** - Comprehensive learning materials

## 🎯 Training Workflows

### 👤 Face/Character LoRA
```bash
# Best configuration: loha_face_config_8gb.json
# 10-20 high-quality face images
# Conservative learning rate for detailed preservation
```

### 🎨 Art Style LoRA  
```bash
# Best configuration: lokr_style_config_8gb.json
# 20-50 images in consistent style
# Higher learning rate for style learning
```

### 🚀 Advanced Techniques
```bash
# DyLoRA: Multi-rank training with dynamic rank selection
# Block-weighted: Fine-tuned layer control
# Embeddings: Textual inversion for concepts
```

## 💾 Hardware Requirements

### Minimum
- **GPU**: NVIDIA GTX 1060 6GB
- **RAM**: 16GB
- **Storage**: 50GB SSD

### Recommended  
- **GPU**: NVIDIA RTX 3070 8GB+ 
- **RAM**: 32GB
- **Storage**: 100GB+ NVMe SSD

### Optimized For
- **RTX 3070 Mobile 8GB VRAM**
- All configurations tested and optimized for this hardware

## 🔧 Memory Optimization

### Training Settings
- **Batch size**: Always 1
- **Precision**: fp16 (not bf16 for RTX 3070)
- **Resolution**: 512x512 for SD 1.5
- **Optimizer**: AdamW8bit for memory efficiency
- **Gradient checkpointing**: Enabled

### Generation Settings
- Use `./webui-8gb.sh` for optimized launch
- Stick to 512x512 resolution
- Close other GPU applications during training

## 📖 Documentation Structure

```
docs/
├── README.md                    # Course overview and navigation
├── 00_course_overview.md        # Prerequisites and objectives  
├── 01_module_foundations.md     # LoRA theory and basics
├── 02_module_basic_training.md  # First training workflow
├── 03_module_dataset_mastery.md # Advanced dataset techniques
├── 04_module_training_types.md  # Architecture comparison
├── 05_module_advanced_techniques.md # Optimization strategies
├── 06_module_optimization.md    # Performance tuning
├── 07_module_troubleshooting.md # Common issues and solutions
├── 08_module_production_pipeline.md # Scaling and automation
└── embeddings_guide.md         # Textual inversion guide
```

## 🚨 Troubleshooting

Having issues? Check our comprehensive troubleshooting guide: **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

Common solutions:
- **Kohya fails to start**: Recreate virtual environment
- **WebUI shows warnings**: Use `webui-8gb.sh` instead of `webui.sh`
- **Out of memory**: Reduce network dimensions or enable disk caching
- **Training fails**: Check logs in `kohya_ss/logs/`

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kohya_ss](https://github.com/kohya-ss/sd-scripts) - LoRA training framework
- [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) - Optimized WebUI
- The Stable Diffusion community for continuous innovation

---

**Ready to start?** → [📖 Begin with the Course Overview](docs/README.md)

**Need help?** → [🚨 Check Troubleshooting Guide](TROUBLESHOOTING.md)

**Want to train now?** → Run `./setup_sd_environment.sh` and get started!