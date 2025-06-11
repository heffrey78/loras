# LoRA Training Mastery Course - Table of Contents

Welcome to the comprehensive LoRA training course! This course takes you from beginner to advanced practitioner in training custom LoRA models for Stable Diffusion.

## Course Modules

### [Module 0: Course Overview](00_course_overview.md)
- Course structure and objectives
- Prerequisites and requirements
- Learning outcomes
- Time commitment

### [Module 1: LoRA Foundations](01_module_foundations.md)
- What is LoRA and how it works
- Understanding network dimensions
- LoRA variants (Standard, LoCon, LoHa, LoKR, DyLoRA)
- Essential training parameters
- First look at Kohya configuration

### [Module 2: Your First LoRA Training](02_module_basic_training.md)
- Planning your first project
- Dataset preparation basics
- Caption writing fundamentals
- Configuration setup
- Training execution and monitoring
- Testing and evaluation

### [Module 3: Dataset Mastery](03_module_dataset_mastery.md)
- Advanced dataset theory
- Image curation strategies
- Resolution and preprocessing
- Advanced captioning techniques
- Dataset augmentation
- Regularization images
- Dataset validation

### [Module 4: LoRA Architecture Deep Dive](04_module_training_types.md)
- Understanding LoRA variants in detail
- Standard LoRA vs LoCon vs LoHa vs LoKR
- Architecture selection guide
- Performance benchmarks
- Advanced architecture techniques

### [Module 5: Advanced Training Techniques](05_module_advanced_techniques.md)
- Multi-resolution training
- Differential learning rates
- Advanced regularization (noise offset, Min-SNR)
- Advanced optimizers (AdamW, Lion, Prodigy)
- Advanced captioning strategies
- Memory optimization
- Experimental techniques

### [Module 6: Training Optimization](06_module_optimization.md)
- Hardware-specific optimization
- Speed optimization techniques
- Multi-GPU training
- Disk and I/O optimization
- Training scheduling
- Quality vs speed trade-offs
- Monitoring and logging

### [Module 7: Troubleshooting Guide](07_module_troubleshooting.md)
- Common training failures (OOM, NaN loss)
- Quality issues (overfitting, underfitting)
- Performance problems
- Data issues
- Configuration errors
- Recovery strategies
- Debugging tools

### [Module 8: Production Pipeline](08_module_production_pipeline.md)
- Automation framework
- Batch processing systems
- Quality assurance
- Continuous integration
- Monitoring and analytics
- Scaling strategies
- Best practices and security

## Quick Start Guides

### For Beginners
1. Start with [Module 0](00_course_overview.md) for overview
2. Complete [Module 1](01_module_foundations.md) to understand basics
3. Follow [Module 2](02_module_basic_training.md) for your first LoRA
4. Practice with provided exercises

### For Intermediate Users
1. Review [Module 3](03_module_dataset_mastery.md) for dataset optimization
2. Explore [Module 4](04_module_training_types.md) for different architectures
3. Apply techniques from [Module 5](05_module_advanced_techniques.md)

### For Advanced Users
1. Focus on [Module 6](06_module_optimization.md) for performance
2. Use [Module 7](07_module_troubleshooting.md) as reference
3. Implement [Module 8](08_module_production_pipeline.md) for scale

## Configuration Templates

The course includes ready-to-use configuration templates:
- `/jeff_face_lora_config_8gb.json` - Basic face training
- `/loha_face_config_8gb.json` - LoHa architecture for faces
- `/lokr_style_config_8gb.json` - LoKR for style training
- `/advanced_dylora_config_8gb.json` - DyLoRA configuration
- `/advanced_block_weight_config_8gb.json` - Block-weighted training

## Practical Exercises

Each module includes hands-on exercises:
- Configuration analysis
- Dataset preparation challenges
- A/B testing experiments
- Debugging scenarios
- Optimization tasks

## Hardware Recommendations

### Minimum Requirements
- GPU: NVIDIA GTX 1060 6GB
- RAM: 16GB
- Storage: 50GB SSD

### Recommended Setup
- GPU: NVIDIA RTX 3070 8GB or better
- RAM: 32GB
- Storage: 100GB+ NVMe SSD

## Community and Support

- **Issues**: Report problems in the GitHub repository
- **Discussions**: Join the community Discord
- **Updates**: Follow the project for new modules

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This course is provided under the MIT License. Feel free to use, modify, and share!

---

**Ready to start?** → [Begin with the Course Overview](00_course_overview.md)