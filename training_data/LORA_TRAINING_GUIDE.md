# LoRA Training Guide - Face vs Style

## Face Training with LoHa (Hadamard Product)

### Configuration: `loha_face_config_8gb.json`

**Best for:** Detailed face/character preservation
- **LoRA Type:** LyCORIS/LoHa
- **Network Dim:** 32 (balanced detail)
- **Conv Dim:** 8 (captures fine facial features)
- **Learning Rate:** 3e-05 (conservative for faces)
- **Key Features:**
  - `use_cp: true` - Compression for efficiency
  - `min_snr_gamma: 5` - Better gradient flow
  - `shuffle_caption: true` - Prevents overfitting
  - `noise_offset: 0.05` - Subtle variation

### Dataset Preparation for Faces:
1. **Image Requirements:**
   - 10-20 high-quality face shots
   - Various angles and expressions
   - Consistent lighting preferred
   - 768x768 or higher resolution

2. **Captioning for Faces:**
   ```
   jeff_person, portrait photo, looking at camera
   jeff_person, side profile, natural lighting
   jeff_person wearing glasses, smiling
   ```

3. **Folder Structure:**
   ```
   training_data/
   └── face_dataset/
       └── 10_jeff_person/
           ├── image1.jpg
           ├── image1.txt
           └── ...
   ```

---

## Style Transfer with LoKR (Low-Rank Kronecker)

### Configuration: `lokr_style_config_8gb.json`

**Best for:** Art styles, visual aesthetics
- **LoRA Type:** LyCORIS/LoKr
- **Network Dim:** 64 (captures style complexity)
- **Factor:** 8 (Kronecker decomposition)
- **Learning Rate:** 1e-04 (higher for style learning)
- **Key Features:**
  - `color_aug: true` - Style variation
  - `flip_aug: true` - More training data
  - `random_crop: true` - Focus on style patterns
  - `noise_offset: 0.1` - Style flexibility

### Dataset Preparation for Styles:
1. **Image Requirements:**
   - 20-50 images in target style
   - Diverse subjects, consistent style
   - Higher resolution better (768x768+)
   - Include various colors/compositions

2. **Captioning for Styles:**
   ```
   a house, painted in style_name style
   landscape scene, style_name art style
   portrait of woman, style_name technique
   ```

3. **Folder Structure:**
   ```
   training_data/
   └── style_dataset/
       └── 30_style_name/
           ├── artwork1.jpg
           ├── artwork1.txt
           └── ...
   ```

---

## Training Tips

### For Face Training:
- Use consistent trigger word (e.g., "jeff_person")
- Include varied expressions and angles
- Lower learning rates prevent "burning"
- Test with prompts like: "jeff_person as a superhero"

### For Style Training:
- Use descriptive style trigger (e.g., "style_name style")
- Include diverse subjects in same style
- Higher network dims capture complexity
- Test with prompts like: "cat in style_name style"

### Memory Optimization for Higher Resolution:
```json
"max_resolution": "896,896",     // For 8GB VRAM limit
"cache_latents_to_disk": true,   // If OOM occurs
"gradient_accumulation_steps": 2, // Virtual larger batch
"network_dim": 24,               // Reduce if needed
```

### Usage in WebUI:
1. Copy trained LoRA to: `stable-diffusion-webui-forge/models/Lora/`
2. In prompt: `<lora:jeff_person:0.8>` or `<lora:style_name:1.0>`
3. Adjust strength (0.5-1.0 typically)

### Combining LoRAs:
```
portrait of <lora:jeff_person:0.8> in <lora:style_name:0.6> style
```