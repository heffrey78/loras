# Textual Inversion (Embeddings) Guide

## Quick Comparison: Embeddings vs LoRA

| Aspect | Embeddings | LoRA |
|--------|------------|------|
| **File Size** | 1-100KB | 2-200MB |
| **Training Time** | 30min-2hrs | 1-12hrs |
| **What It Modifies** | Text tokens only | Model weights |
| **Best For** | Styles, objects, concepts | Faces, characters |
| **Flexibility** | Medium | High |
| **Compatibility** | Universal | Model-specific |

## When to Choose Embeddings

### Perfect For:
- **Art Styles**: watercolor, oil painting, specific artistic techniques
- **Objects**: unique items, buildings, vehicles
- **Abstract Concepts**: moods, atmospheres, lighting
- **Quick Experiments**: testing concepts rapidly
- **Small Datasets**: 5-15 images work well

### Not Ideal For:
- Complex character features
- Facial identity training
- Large anatomical changes
- Detailed transformations

## Embedding Creation Workflow

### 1. Dataset Preparation

```
embeddings_training/
└── concept_name/
    ├── 001.jpg
    ├── 001.txt  → "concept_name"
    ├── 002.jpg  
    ├── 002.txt  → "concept_name"
    └── ...
```

**Caption Rules:**
- Use ONLY your token word
- No additional descriptions
- Keep it simple: just "mystyle" or "myobject"

### 2. Training Configuration

**Basic Embedding Config:**
```json
{
    "model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "train_data_dir": "./embeddings_training/concept_name",
    "output_dir": "./outputs/embeddings",
    "output_name": "concept_name",
    
    "token_string": "concept_name",
    "init_word": "style",
    "num_vectors_per_token": 1,
    
    "learning_rate": "5e-4",
    "train_batch_size": 1,
    "max_train_steps": 1000,
    "save_every_n_steps": 100,
    
    "mixed_precision": "fp16",
    "gradient_checkpointing": true,
    "use_8bit_adam": true
}
```

### 3. Key Parameters Explained

**Token String:**
```json
"token_string": "mystyle"  // Your activation word
```
- Choose unique, memorable names
- Avoid existing words
- Use lowercase
- No spaces or special characters

**Initialization Word:**
```json
"init_word": "painting"   // Starting point for learning
```
Common choices:
- `"style"` - for art styles
- `"person"` - for faces/characters  
- `"object"` - for things/items
- `"concept"` - for abstract ideas

**Vector Count:**
```json
"num_vectors_per_token": 1    // Simple concepts
"num_vectors_per_token": 3    // Medium complexity
"num_vectors_per_token": 8    // Complex concepts
```

### 4. Training Process

**Using Kohya GUI:**
1. Navigate to "Textual Inversion" tab
2. Load your configuration
3. Set paths and parameters
4. Click "Start Training"

**Monitor Training:**
- Loss should decrease gradually
- Save samples every 100 steps
- Total training: 500-2000 steps typical

## Advanced Techniques

### Multi-Vector Embeddings

For complex concepts, use multiple vectors:

```json
{
    "num_vectors_per_token": 4,
    "learning_rate": "3e-4",  // Lower LR for stability
    "max_train_steps": 1500   // More steps needed
}
```

### Template-Based Training

Create variations in your captions:

```
# Instead of just "mystyle"
mystyle
a mystyle painting
mystyle artwork
painting in mystyle
```

### Progressive Training

Start simple, add complexity:

```python
# Stage 1: Core concept
stage1_captions = ["mystyle"]

# Stage 2: Add context  
stage2_captions = ["mystyle painting", "mystyle art"]

# Stage 3: Add variety
stage3_captions = ["mystyle portrait", "mystyle landscape"]
```

## Practical Examples

### Example 1: Watercolor Style

**Dataset:** 12 watercolor paintings
**Token:** "mywater"
**Init:** "watercolor"
**Steps:** 800
**Usage:** `mywater painting of mountains`

### Example 2: Vintage Car

**Dataset:** 15 photos of specific car model
**Token:** "vintagecar"  
**Init:** "car"
**Steps:** 1000
**Usage:** `vintagecar driving through city`

### Example 3: Mood/Atmosphere

**Dataset:** 20 photos with specific lighting
**Token:** "goldenhour"
**Init:** "lighting"
**Steps:** 600
**Usage:** `portrait in goldenhour lighting`

## Using Your Embeddings

### Installation
```bash
# Copy the .pt file to WebUI embeddings folder
cp mystyle.pt stable-diffusion-webui/embeddings/
```

### In Prompts
```bash
# Basic usage
"mystyle portrait of woman"

# With strength control
"(mystyle:1.2) landscape painting"  

# Combined with other modifiers
"mystyle, highly detailed, masterpiece, 4k"

# Negative usage (rare but possible)
"portrait, (mystyle:0.8) in the style but subtle"
```

### Testing Your Embedding

**Test Prompts:**
1. `mystyle` - Basic activation
2. `mystyle portrait` - With subject  
3. `mystyle landscape` - Different context
4. `beautiful mystyle artwork` - With quality words

## Troubleshooting

### Common Issues

**Weak Effect:**
```json
// Solutions:
"learning_rate": "8e-4",  // Increase LR
"max_train_steps": 1500,  // More training
"num_vectors_per_token": 3  // More capacity
```

**Overtraining:**
```json
// Solutions:
"learning_rate": "3e-4",  // Lower LR  
"max_train_steps": 800,   // Less training
"save_every_n_steps": 50  // Check progress more often
```

**No Effect:**
- Check file placement in embeddings folder
- Verify token name matches filename
- Test with simple prompts first
- Check WebUI settings

### Quality Issues

**Too Literal:**
- Reduce training steps
- Use more varied dataset
- Lower learning rate

**Not Recognizable:**
- Increase training steps
- Check dataset quality
- Use better initialization word

## Embedding + LoRA Combinations

You can use both together effectively:

```bash
# Embedding for style + LoRA for character
"<lora:character_name:0.8> mystyle portrait of character_name"

# Multiple embeddings
"mystyle lighting_mood portrait"

# Embedding + multiple LoRAs
"<lora:face:0.7> <lora:clothing:0.5> mystyle portrait"
```

## Best Practices

### Dataset Quality
- Consistent style/concept across images
- Good variety in subjects/compositions  
- High resolution (512x512+)
- Clean, artifact-free images

### Training Strategy
- Start with fewer steps, increase if needed
- Monitor sample outputs regularly
- Test different initialization words
- Save multiple checkpoints

### Usage Tips
- Start with strength 1.0, adjust as needed
- Test in different contexts
- Combine thoughtfully with other techniques
- Document what works for future reference

## Automation Scripts

### Batch Embedding Creator
```python
def create_embedding_batch(concepts):
    for concept in concepts:
        config = base_embedding_config.copy()
        config.update({
            "token_string": concept["name"],
            "init_word": concept["init"],
            "train_data_dir": f"./data/{concept['name']}"
        })
        
        train_embedding(config)
```

### Quality Tester
```python
def test_embedding(embedding_path, test_prompts):
    results = []
    for prompt in test_prompts:
        image = generate_with_embedding(prompt, embedding_path)
        score = evaluate_quality(image, prompt)
        results.append(score)
    return np.mean(results)
```

Embeddings are perfect for quickly teaching new styles and concepts with minimal training time and data!