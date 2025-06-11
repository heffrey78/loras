# Module 4: LoRA Architecture Deep Dive

## 4.1 Understanding LoRA Variants

### The LoRA Family Tree

```
Standard LoRA
├── LoCon (LoRA + Convolution)
├── LoHa (Hadamard Product)
├── LoKR (Kronecker Product)
├── DyLoRA (Dynamic Rank)
├── IA3 (Infused Adapter)
└── Flux LoRA (Latest Architecture)
```

### Architecture Comparison

| Type | Best For | File Size | Training Speed | Flexibility |
|------|----------|-----------|----------------|-------------|
| Standard | General use | Medium | Fast | High |
| LoCon | Styles | Large | Medium | Very High |
| LoHa | Details | Small | Fast | High |
| LoKR | Efficiency | Very Small | Medium | Medium |
| DyLoRA | Experimentation | Variable | Slow | Very High |

## 4.2 Standard LoRA

### Architecture Details

```python
# Simplified LoRA implementation
class LoRALayer:
    def __init__(self, in_dim, out_dim, rank=16, alpha=16):
        self.down = Linear(in_dim, rank)
        self.up = Linear(rank, out_dim)
        self.scale = alpha / rank
        
    def forward(self, x, original_weight):
        # LoRA formula: W' = W + (BA) * scale
        return original_weight(x) + self.up(self.down(x)) * self.scale
```

### When to Use Standard LoRA

**Perfect for:**
- First-time trainers
- General subjects
- Maximum compatibility
- Predictable results

**Configuration Example:**
```json
{
    "LoRA_type": "Standard",
    "network_module": "networks.lora",
    "network_dim": 32,
    "network_alpha": 16,
    "conv_dim": 0,
    "conv_alpha": 0
}
```

### Training Tips

1. **Rank Selection**
   - Faces: 16-32
   - Styles: 32-64
   - Complex: 64-128

2. **Alpha Strategy**
   - Alpha = Rank/2 (conservative)
   - Alpha = Rank (standard)
   - Alpha = Rank*2 (aggressive)

## 4.3 LoCon (LoRA for Convolution)

### Architecture Innovation

LoCon extends LoRA to convolutional layers, not just attention:

```python
# LoCon adds to conv layers
class LoConLayer:
    def __init__(self, conv_in, conv_out, rank=16):
        # Standard LoRA on attention
        self.attn_down = Linear(attn_dim, rank)
        self.attn_up = Linear(rank, attn_dim)
        
        # NEW: LoRA on convolution
        self.conv_down = Conv2d(conv_in, rank, 1)
        self.conv_up = Conv2d(rank, conv_out, 1)
```

### When to Use LoCon

**Perfect for:**
- Art styles
- Texture training
- Visual effects
- Complex transformations

**Configuration Example:**
```json
{
    "LoRA_type": "Kohya LoCon",
    "network_module": "lycoris.kohya",
    "network_dim": 32,
    "network_alpha": 16,
    "conv_dim": 16,
    "conv_alpha": 8,
    "algo": "lora"
}
```

### LoCon-Specific Parameters

```json
{
    // Attention layers
    "network_dim": 32,
    "network_alpha": 16,
    
    // Convolution layers
    "conv_dim": 16,
    "conv_alpha": 8,
    
    // Advanced
    "dropout": 0.1,
    "cp_decomposition": true
}
```

## 4.4 LoHa (Hadamard Product)

### Mathematical Foundation

LoHa uses Hadamard (element-wise) product decomposition:

```python
# Simplified LoHa
class LoHaLayer:
    def __init__(self, dim, rank=16):
        self.w1 = Parameter(dim, rank)
        self.w2 = Parameter(rank, dim)
        self.b1 = Parameter(dim, rank)
        self.b2 = Parameter(rank, dim)
        
    def forward(self, x):
        # Hadamard product decomposition
        return (self.w1 @ self.w2) * (self.b1 @ self.b2)
```

### When to Use LoHa

**Perfect for:**
- Face details
- Character features
- Fine textures
- Efficient training

**Configuration Example:**
```json
{
    "LoRA_type": "LyCORIS/LoHa",
    "network_module": "lycoris.kohya",
    "algo": "loha",
    "network_dim": 32,
    "network_alpha": 16,
    "conv_dim": 8,
    "conv_alpha": 4,
    "use_cp": true
}
```

### LoHa Optimization

```json
{
    // LoHa specific
    "use_cp": true,  // Compression
    "use_tucker": false,
    "use_scalar": false,
    "rank_dropout": 0.1,
    "module_dropout": 0.1
}
```

## 4.5 LoKR (Kronecker Product)

### Mathematical Efficiency

LoKR uses Kronecker product for extreme efficiency:

```python
# Kronecker factorization
class LoKRLayer:
    def __init__(self, dim, rank=16, factor=8):
        self.factor = factor
        # Much smaller matrices
        self.w1_a = Parameter(dim//factor, rank//factor)
        self.w1_b = Parameter(factor, factor)
        self.w2_a = Parameter(rank//factor, dim//factor)
        self.w2_b = Parameter(factor, factor)
    
    def forward(self, x):
        # Kronecker product: A ⊗ B
        w1 = kronecker_product(self.w1_a, self.w1_b)
        w2 = kronecker_product(self.w2_a, self.w2_b)
        return w2 @ w1 @ x
```

### When to Use LoKR

**Perfect for:**
- Limited storage
- Style transfer
- Efficient inference
- Large-scale deployment

**Configuration Example:**
```json
{
    "LoRA_type": "LyCORIS/LoKr",
    "network_module": "lycoris.kohya",
    "algo": "lokr",
    "network_dim": 64,
    "network_alpha": 64,
    "factor": 8,
    "decompose_both": false,
    "use_scalar": false
}
```

### LoKR Tuning

```json
{
    // LoKR specific
    "factor": 8,  // Decomposition factor
    "decompose_both": false,
    "decompose_factor": -1,
    "dora_wd": false,
    "use_scalar": false
}
```

## 4.6 DyLoRA (Dynamic LoRA)

### Multi-Rank Training

DyLoRA trains multiple ranks simultaneously:

```python
# DyLoRA concept
class DyLoRALayer:
    def __init__(self, dim, ranks=[4,8,16,32]):
        self.ranks = ranks
        # Nested matrices
        self.down = Linear(dim, max(ranks))
        self.up = Linear(max(ranks), dim)
    
    def forward(self, x, selected_rank):
        # Use only selected rank
        down_truncated = self.down[:, :selected_rank]
        up_truncated = self.up[:selected_rank, :]
        return up_truncated @ down_truncated @ x
```

### When to Use DyLoRA

**Perfect for:**
- Experimentation
- Finding optimal rank
- Flexible deployment
- Research purposes

**Configuration Example:**
```json
{
    "LoRA_type": "DyLoRA",
    "network_module": "networks.dylora",
    "network_dim": 64,
    "network_alpha": 32,
    "dylora_unit": 4,
    "conv_dim": 32,
    "conv_alpha": 16
}
```

## 4.7 Architecture Selection Guide

### Decision Tree

```
What are you training?
├── Face/Character
│   ├── High detail needed → LoHa
│   └── Standard quality → Standard LoRA
├── Style/Artistic
│   ├── Complex style → LoCon
│   └── Simple style → LoKR
├── Object/Product
│   └── Standard LoRA
└── Experimental
    └── DyLoRA
```

### Comparative Configurations

**Face Training Comparison:**

```json
// Standard LoRA
{
    "network_dim": 32,
    "file_size": "~40MB"
}

// LoHa
{
    "network_dim": 32,
    "conv_dim": 8,
    "file_size": "~25MB"
}

// LoKR
{
    "network_dim": 64,
    "factor": 8,
    "file_size": "~15MB"
}
```

## 4.8 Advanced Architecture Techniques

### 1. Hybrid Architectures

```json
{
    // Mix LoRA types per layer
    "block_lora_types": "lora,lora,loha,loha,lokr,lokr",
    "block_dims": "16,16,32,32,64,64",
    "block_alphas": "8,8,16,16,32,32"
}
```

### 2. Progressive Architecture

```python
# Train different architectures sequentially
training_stages = [
    {"type": "Standard", "epochs": 5},
    {"type": "LoCon", "epochs": 5},
    {"type": "LoHa", "epochs": 5}
]
```

### 3. Architecture Ensemble

```python
# Combine multiple LoRAs
def ensemble_inference(prompt, loras):
    results = []
    for lora in loras:
        results.append(generate(prompt, lora))
    return merge_results(results)
```

## 4.9 Performance Benchmarks

### Training Performance

| Architecture | VRAM Usage | Speed | Quality |
|--------------|------------|-------|---------|
| Standard | 100% | 100% | 100% |
| LoCon | 150% | 80% | 110% |
| LoHa | 90% | 95% | 105% |
| LoKR | 70% | 85% | 95% |
| DyLoRA | 120% | 60% | 100% |

### Inference Performance

| Architecture | Load Time | Generation Speed | Flexibility |
|--------------|-----------|------------------|-------------|
| Standard | Fast | Fast | High |
| LoCon | Medium | Medium | Very High |
| LoHa | Fast | Fast | High |
| LoKR | Very Fast | Fast | Medium |
| DyLoRA | Slow | Variable | Very High |

## 4.10 Practice Projects

### Project 1: Architecture Comparison

Train the same dataset with:
1. Standard LoRA
2. LoHa
3. LoKR

Compare:
- File sizes
- Training time
- Quality results
- Flexibility

### Project 2: Optimal Architecture Finding

1. Start with DyLoRA
2. Test different ranks
3. Choose best rank
4. Train final with optimal architecture

### Project 3: Hybrid Training

1. Train style with LoCon
2. Train details with LoHa
3. Merge for final result

## Next Module Preview

Module 5 covers advanced training techniques:
- Multi-resolution strategies
- Differential learning rates
- Advanced regularization
- Training optimization