# Module 2: Your First LoRA Training

## 2.1 Planning Your First LoRA

### Choosing a Subject

**Good First Projects:**
- A specific person (face LoRA)
- A distinctive art style
- A unique object
- A consistent character

**Avoid for First Project:**
- Multiple concepts
- Abstract ideas
- Inconsistent styles
- Low-quality sources

### Success Criteria

Before starting, define success:
- Clear reproduction of subject
- Flexibility in poses/contexts
- No artifacts or distortions
- Reasonable file size

## 2.2 Dataset Preparation

### Image Requirements

**Quality Standards:**
- Resolution: 512×512 minimum
- Format: PNG or high-quality JPG
- Consistency: Similar lighting/quality
- Diversity: Various angles/poses

**Quantity Guidelines:**
- Face/Character: 15-20 images
- Style: 20-50 images
- Object: 10-30 images
- Concept: 30-100 images

### Dataset Structure

```
training_data/
└── my_first_lora/
    └── 10_subject_name/
        ├── img001.png
        ├── img001.txt
        ├── img002.png
        ├── img002.txt
        └── ...
```

**Folder Naming:** `[repeats]_[trigger_word]`
- Repeats: How many times to use each image
- Trigger word: Activation phrase for your LoRA

### Image Preparation Checklist

- [ ] Crop to subject (no unnecessary background)
- [ ] Consistent aspect ratio
- [ ] Remove duplicates
- [ ] Check for blur/artifacts
- [ ] Ensure variety in dataset

## 2.3 Caption Writing

### Caption Basics

**Structure:** `trigger_word, description, style, quality`

**Examples:**
```
# Character
john_doe, man in business suit, standing, photo, high quality

# Style  
artestyle, landscape painting, mountains, oil on canvas

# Object
vintage_camera, old camera on wooden table, product photo
```

### Caption Best Practices

1. **Consistency**
   - Always start with trigger word
   - Use consistent terminology
   - Maintain similar detail level

2. **Descriptiveness**
   - Describe what you see
   - Include relevant details
   - Avoid subjective terms

3. **Flexibility**
   - Don't over-specify
   - Allow for variations
   - Keep some captions simple

### Automated Captioning

```bash
# Using BLIP for basic captions
python caption_blip.py --input_dir ./images --output_dir ./captions

# Manual refinement is always needed!
```

## 2.4 Configuration Setup

### Starter Configuration

```json
{
    "LoRA_type": "Standard",
    "adaptive_noise_scale": 0,
    "additional_parameters": "",
    "block_alphas": "",
    "block_dims": "",
    "bucket_no_upscale": true,
    "bucket_reso_steps": 64,
    "cache_latents": true,
    "caption_dropout_every_n_epochs": 0,
    "caption_dropout_rate": 0,
    "caption_extension": ".txt",
    "clip_skip": 2,
    "color_aug": false,
    "enable_bucket": true,
    "epoch": 10,
    "flip_aug": false,
    "full_fp16": false,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "keep_tokens": 0,
    "learning_rate": "5e-05",
    "lr_scheduler": "cosine",
    "lr_warmup": "0",
    "max_data_loader_n_workers": "0",
    "max_resolution": "512,512",
    "max_token_length": "75",
    "mixed_precision": "fp16",
    "network_alpha": 16,
    "network_dim": 32,
    "no_token_padding": false,
    "noise_offset": 0,
    "optimizer": "AdamW8bit",
    "persistent_data_loader_workers": false,
    "prior_loss_weight": 1,
    "random_crop": false,
    "save_every_n_epochs": 2,
    "save_precision": "fp16",
    "seed": "1234",
    "shuffle_caption": false,
    "stop_text_encoder_training": 0,
    "text_encoder_lr": "5e-05",
    "train_batch_size": 1,
    "unet_lr": "5e-05",
    "xformers": true
}
```

### Key Parameters Explained

**For 8GB VRAM:**
- `train_batch_size`: Keep at 1
- `gradient_checkpointing`: Always true
- `mixed_precision`: "fp16"
- `cache_latents`: true
- `max_resolution`: "512,512" to start

## 2.5 Training Execution

### Pre-Training Checklist

- [ ] Dataset prepared and structured
- [ ] Captions written and saved
- [ ] Configuration file created
- [ ] Base model downloaded
- [ ] Output directories exist
- [ ] GPU memory cleared

### Starting Training

1. **Launch Kohya GUI**
   ```bash
   cd kohya_ss
   ./gui.sh
   ```

2. **Load Configuration**
   - Navigate to LoRA tab
   - Click "Open" under Configuration
   - Select your JSON file

3. **Verify Settings**
   - Check paths are correct
   - Confirm dataset detected
   - Verify output location

4. **Start Training**
   - Click "Start training"
   - Monitor progress
   - Watch for errors

### Monitoring Progress

**Good Signs:**
- Loss decreasing steadily
- No CUDA errors
- Regular checkpoint saves
- Sample images improving

**Warning Signs:**
- Loss increasing or NaN
- CUDA out of memory
- Very slow progress
- Artifacts in samples

## 2.6 Testing Your LoRA

### Installation

1. Copy LoRA file to:
   ```
   stable-diffusion-webui/models/Lora/
   ```

2. Restart WebUI or refresh models

### Testing Prompts

**Basic Test:**
```
<lora:your_lora_name:1.0> trigger_word
```

**Strength Testing:**
```
<lora:your_lora_name:0.5> trigger_word  # Lower strength
<lora:your_lora_name:1.2> trigger_word  # Higher strength
```

**Integration Testing:**
```
masterpiece, <lora:your_lora_name:0.8> trigger_word in a forest
<lora:your_lora_name:0.7> trigger_word, cyberpunk style
```

### Evaluation Criteria

1. **Accuracy**
   - Does it match your subject?
   - Are details preserved?
   - Is it recognizable?

2. **Flexibility**
   - Works in different contexts?
   - Responds to style prompts?
   - Maintains quality?

3. **Artifacts**
   - Any distortions?
   - Color shifts?
   - Unwanted patterns?

## 2.7 Common First Training Issues

### Issue: Overtraining

**Symptoms:**
- Exact recreation of training images
- No flexibility
- "Burned in" backgrounds

**Solutions:**
- Reduce epochs
- Lower learning rate
- Add caption variety
- Increase dataset size

### Issue: Undertraining

**Symptoms:**
- Weak effect
- Doesn't match subject
- Requires high strength

**Solutions:**
- Increase epochs
- Raise learning rate
- Check captions
- Improve dataset quality

### Issue: Artifacts

**Symptoms:**
- Strange colors
- Distortions
- Noise patterns

**Solutions:**
- Check dataset quality
- Reduce noise offset
- Lower learning rate
- Enable SNR gamma

## 2.8 Practice Project

### Project: Personal Avatar LoRA

**Goal:** Create a LoRA of yourself or a character

**Steps:**
1. Take/collect 15 photos
   - Different angles
   - Various expressions
   - Consistent lighting

2. Prepare dataset
   - Crop to 512×512
   - Name folder: `10_yourname_person`

3. Write captions
   ```
   yourname_person, smiling, front view
   yourname_person, profile, serious expression
   ```

4. Configure training
   - Use starter configuration
   - Set epochs to 10
   - Save every 2 epochs

5. Train and test
   - Monitor loss
   - Test each checkpoint
   - Find optimal version

### Success Metrics

- [ ] Recognizable likeness
- [ ] Works with different prompts
- [ ] No major artifacts
- [ ] File size under 50MB
- [ ] Optimal strength ~0.7-0.9

## 2.9 Optimization Tips

### Quick Improvements

1. **Dataset Quality > Quantity**
   - 10 great images > 50 poor ones
   - Consistency matters
   - Remove outliers

2. **Caption Strategy**
   - Simple often works better
   - Test with/without details
   - Keep trigger word consistent

3. **Iterative Refinement**
   - Start conservative
   - Test early checkpoints
   - Adjust based on results

## Next Module Preview

Module 3 will cover advanced dataset preparation:
- Multi-resolution training
- Advanced captioning strategies
- Dataset augmentation
- Regularization images
- A/B testing methods