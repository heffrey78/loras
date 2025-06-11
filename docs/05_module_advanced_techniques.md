# Module 5: Advanced Training Techniques

## 5.1 Multi-Resolution Training

### The Power of Resolution Diversity

Training at multiple resolutions creates more robust and flexible LoRAs that work across different generation sizes.

### Bucketing Strategy

```json
{
    "enable_bucket": true,
    "min_bucket_reso": 512,
    "max_bucket_reso": 1024,
    "bucket_reso_steps": 64,
    "bucket_no_upscale": false
}
```

**How Bucketing Works:**
1. Groups images by aspect ratio
2. Creates resolution buckets
3. Maintains aspect ratios
4. Prevents distortion

### Advanced Bucket Configuration

```python
# Custom bucket distribution
def calculate_buckets(min_res=512, max_res=1024, steps=64):
    buckets = []
    for width in range(min_res, max_res + 1, steps):
        for height in range(min_res, max_res + 1, steps):
            aspect = width / height
            if 0.5 <= aspect <= 2.0:  # Reasonable aspects
                buckets.append((width, height))
    return buckets

# Result: [(512,512), (512,576), ..., (1024,1024)]
```

### Progressive Resolution Training

```python
# Stage 1: Low resolution for structure
stage1_config = {
    "max_resolution": "512,512",
    "epoch": 5,
    "learning_rate": "1e-4"
}

# Stage 2: Medium resolution for details
stage2_config = {
    "max_resolution": "768,768",
    "epoch": 5,
    "learning_rate": "5e-5"
}

# Stage 3: High resolution for refinement
stage3_config = {
    "max_resolution": "1024,1024",
    "epoch": 5,
    "learning_rate": "1e-5"
}
```

## 5.2 Differential Learning Rates

### Layer-Specific Learning

Different parts of the model learn at different rates:

```json
{
    "text_encoder_lr": "2e-5",
    "unet_lr": "1e-4",
    
    // Advanced: Block-specific
    "down_lr_weight": "1,1,1,1,1,1,1,1,1,1,1,1",
    "mid_lr_weight": "1",
    "up_lr_weight": "1,1,1,1,1,1,1,1,1,1,1,1"
}
```

### Learning Rate Strategies

**1. Text Encoder vs U-Net**
```json
{
    // Conservative text encoder
    "text_encoder_lr": "1e-5",
    
    // Aggressive U-Net
    "unet_lr": "5e-4",
    
    // Stop text encoder early
    "stop_text_encoder_training": 10
}
```

**2. Layer-Wise Decay**
```python
def get_layer_lr(base_lr, layer_idx, total_layers):
    # Exponential decay from input to output
    decay_factor = 0.95 ** layer_idx
    return base_lr * decay_factor
```

**3. Adaptive Scheduling**
```json
{
    "lr_scheduler": "polynomial",
    "lr_scheduler_power": 2.0,
    "lr_scheduler_num_cycles": 3,
    "lr_warmup_steps": 100
}
```

## 5.3 Advanced Regularization

### Noise-Based Regularization

**1. Noise Offset**
```json
{
    "noise_offset": 0.1,
    "adaptive_noise_scale": 0.005,
    "noise_offset_random_strength": true
}
```

**2. Multi-Resolution Noise**
```json
{
    "multires_noise_iterations": 8,
    "multires_noise_discount": 0.3
}
```

**3. Pyramid Noise**
```python
def apply_pyramid_noise(latent, iterations=6):
    noise_scales = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    for i, scale in enumerate(noise_scales[:iterations]):
        # Apply noise at different frequencies
        freq = 2 ** i
        noise = generate_noise(latent.shape, frequency=freq)
        latent += noise * scale
    return latent
```

### Min-SNR Gamma

Signal-to-Noise Ratio weighting for better gradient flow:

```json
{
    "min_snr_gamma": 5,
    "scale_v_pred_loss_like_noise_pred": true,
    "v_pred_like_loss": 0.5
}
```

**Understanding SNR Gamma:**
- Balances learning across timesteps
- Prevents gradient vanishing
- Improves detail preservation
- Values: 1-20 (5 is common)

### Dropout Strategies

```json
{
    // Caption dropout
    "caption_dropout_rate": 0.1,
    "caption_dropout_every_n_epochs": 2,
    "caption_tag_dropout_rate": 0.1,
    
    // Network dropout
    "network_dropout": 0.1,
    "rank_dropout": 0.1,
    "module_dropout": 0.05
}
```

## 5.4 Advanced Optimizers

### Optimizer Comparison

| Optimizer | Memory | Speed | Stability | Best For |
|-----------|---------|--------|-----------|----------|
| AdamW | High | Fast | Very Stable | General |
| AdamW8bit | Low | Fast | Stable | Limited VRAM |
| Lion | Medium | Very Fast | Medium | Experimentation |
| Prodigy | Medium | Medium | Adaptive | No LR tuning |
| Adafactor | Low | Medium | Stable | Large models |

### Optimizer Configurations

**1. AdamW with Decoupled Weight Decay**
```json
{
    "optimizer": "AdamW",
    "optimizer_args": "weight_decay=0.1 betas=0.9,0.999"
}
```

**2. Lion Optimizer**
```json
{
    "optimizer": "Lion",
    "learning_rate": "1e-5",  // Use 10x lower LR
    "optimizer_args": "weight_decay=0.01 betas=0.95,0.98"
}
```

**3. Prodigy (Self-Tuning)**
```json
{
    "optimizer": "Prodigy",
    "learning_rate": "1.0",  // Prodigy adjusts automatically
    "optimizer_args": "weight_decay=0.01 decouple=True use_bias_correction=True safeguard_warmup=True"
}
```

## 5.5 Advanced Captioning Techniques

### Dynamic Caption Templates

```python
class DynamicCaptioner:
    def __init__(self):
        self.templates = [
            "{trigger}, {subject}, {style}",
            "{style} {subject}, {trigger}",
            "{trigger} in {style} style, {subject}",
            "a {style} image of {trigger}, {subject}"
        ]
    
    def generate_captions(self, trigger, subject, style):
        captions = []
        for template in self.templates:
            caption = template.format(
                trigger=trigger,
                subject=subject,
                style=style
            )
            captions.append(caption)
        return captions
```

### Weighted Token Strategy

```python
# Advanced weighting
caption = "(trigger:1.2), (important_feature:1.1), normal_feature, (background:0.8)"

# Negative weighting
caption = "trigger, subject, [avoid:blurry:0.5], [avoid:low_quality:0.5]"

# Alternating concepts
caption = "[cat|dog], playing in park"  # Alternates between cat and dog
```

### Caption Augmentation

```python
def augment_caption(base_caption, variations):
    augmented = []
    
    # Synonym replacement
    for synonym in get_synonyms(base_caption):
        augmented.append(replace_with_synonym(base_caption, synonym))
    
    # Detail level variation
    augmented.append(simplify_caption(base_caption))
    augmented.append(elaborate_caption(base_caption))
    
    # Style variations
    for style in ["photorealistic", "artistic", "anime"]:
        augmented.append(f"{base_caption}, {style}")
    
    return augmented
```

## 5.6 Training Strategies

### Curriculum Learning

Train from simple to complex:

```python
curriculum_stages = [
    {
        "name": "Basic Structure",
        "dataset": "simple_poses",
        "epochs": 5,
        "learning_rate": "1e-4"
    },
    {
        "name": "Complex Features",
        "dataset": "detailed_images",
        "epochs": 10,
        "learning_rate": "5e-5"
    },
    {
        "name": "Fine Details",
        "dataset": "high_quality_subset",
        "epochs": 5,
        "learning_rate": "1e-5"
    }
]
```

### Ensemble Training

Train multiple LoRAs and merge:

```python
def ensemble_training(dataset, configs):
    loras = []
    
    for config in configs:
        lora = train_lora(dataset, config)
        loras.append(lora)
    
    # Merge strategies
    merged = merge_loras(loras, method="average")
    # or
    merged = merge_loras(loras, method="weighted", weights=[0.5, 0.3, 0.2])
    
    return merged
```

### Adversarial Training

```json
{
    // Add controlled noise
    "adversarial_noise": 0.01,
    
    // Random perturbations
    "random_perturbation": true,
    
    // Gradient penalty
    "gradient_penalty": 0.1
}
```

## 5.7 Memory Optimization

### Gradient Checkpointing Plus

```json
{
    "gradient_checkpointing": true,
    "xformers": true,
    "sdpa": true,  // Scaled Dot Product Attention
    "memory_efficient_attention": true
}
```

### Mixed Precision Strategies

```json
{
    // FP16 with gradient scaling
    "mixed_precision": "fp16",
    "full_fp16": false,
    "gradient_accumulation_steps": 2,
    
    // BF16 for newer GPUs
    "mixed_precision": "bf16",
    "full_bf16": true
}
```

### Cache Management

```json
{
    "cache_latents": true,
    "cache_latents_to_disk": true,
    "cache_text_encoder_outputs": true,
    "cache_text_encoder_outputs_to_disk": true
}
```

## 5.8 Advanced Validation

### Validation During Training

```json
{
    "validation_prompt": "trigger_word, high quality portrait",
    "validation_epochs": 2,
    "validation_steps": 100,
    "save_sample_prompt": "trigger_word in different style"
}
```

### A/B Testing Framework

```python
class LoRAValidator:
    def __init__(self, test_prompts):
        self.test_prompts = test_prompts
        self.metrics = []
    
    def validate(self, lora_path):
        results = {}
        
        for prompt in self.test_prompts:
            # Generate images
            images = generate_batch(prompt, lora_path, count=4)
            
            # Calculate metrics
            results[prompt] = {
                "clip_score": calculate_clip_score(images, prompt),
                "aesthetic_score": calculate_aesthetic_score(images),
                "fid_score": calculate_fid(images, reference_set)
            }
        
        return results
```

## 5.9 Experimental Techniques

### 1. Attention Coupling

```json
{
    "cross_attention_coupling": true,
    "coupling_factor": 0.1,
    "couple_latents": true
}
```

### 2. Feature Injection

```python
def inject_features(model, feature_extractor, injection_layers):
    features = feature_extractor.extract()
    
    for layer_idx in injection_layers:
        model.layers[layer_idx].inject(features[layer_idx])
```

### 3. Progressive Rank Growth

```python
# Start small, grow rank
growth_schedule = [
    {"epoch": 0, "rank": 4},
    {"epoch": 5, "rank": 8},
    {"epoch": 10, "rank": 16},
    {"epoch": 15, "rank": 32}
]
```

## 5.10 Practice Challenges

### Challenge 1: Multi-Stage Training

Create a 3-stage training pipeline:
1. Low-res structure learning
2. Mid-res detail capture
3. High-res refinement

### Challenge 2: Optimizer Shootout

Train same dataset with:
- AdamW
- Lion
- Prodigy

Compare convergence and quality.

### Challenge 3: Advanced Regularization

Implement training with:
- Min-SNR gamma
- Multi-res noise
- Progressive dropout

## Next Module Preview

Module 6 explores optimization strategies:
- Hardware-specific tuning
- Distributed training
- Performance profiling
- Production optimization