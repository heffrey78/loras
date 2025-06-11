# Module 1: LoRA Foundations

## 1.1 What is LoRA?

LoRA (Low-Rank Adaptation) is a training technique that allows you to fine-tune large models like Stable Diffusion efficiently by training only a small number of parameters.

### Key Concepts

**Traditional Fine-tuning:**
- Modifies entire model (3-7GB)
- Requires massive GPU memory
- Slow training and inference

**LoRA Advantages:**
- Small file size (2-200MB)
- Trains only "difference" from base model
- Can be mixed and matched
- Hardware efficient

### How LoRA Works

```
Original Weight Matrix (W) = Base Model Weight + LoRA Weight (BA)
Where: B is down-projection, A is up-projection
```

The "rank" determines the bottleneck size between B and A matrices.

## 1.2 LoRA Architecture

### Network Dimensions

**Network Dim (Rank):**
- Controls model capacity
- Common values: 4, 8, 16, 32, 64, 128
- Higher = more capacity but larger file

**Network Alpha:**
- Scaling factor for LoRA weights
- Usually set to 1/2 of network dim
- Affects learning dynamics

### LoRA Variants

1. **Standard LoRA**
   - Original implementation
   - Works on attention layers
   - Most compatible

2. **LoCon (LoRA for Convolution)**
   - Includes convolutional layers
   - Better for styles
   - Larger file size

3. **LoHa (LoRA with Hadamard Product)**
   - Uses Hadamard product decomposition
   - More parameter efficient
   - Good for details

4. **LoKR (LoRA with Kronecker Product)**
   - Uses Kronecker decomposition
   - Very parameter efficient
   - Good for styles

5. **DyLoRA (Dynamic LoRA)**
   - Trains multiple ranks simultaneously
   - Flexible at inference
   - Experimental

## 1.3 Understanding Training Parameters

### Essential Parameters

**Learning Rate:**
- How fast the model learns
- Too high = unstable/artifacts
- Too low = slow/no learning
- Typical: 1e-4 to 1e-5

**Batch Size:**
- Images processed together
- Limited by VRAM
- Affects gradient quality
- Usually 1 for 8GB GPUs

**Epochs:**
- Complete passes through dataset
- More epochs ≠ better
- Watch for overfitting
- Typical: 10-20

**Steps:**
- Total training iterations
- Steps = (Images × Repeats × Epochs) / Batch Size
- More important than epochs

### Advanced Parameters

**Gradient Accumulation:**
- Simulates larger batch size
- Trades speed for quality
- Useful for limited VRAM

**Mixed Precision:**
- fp16: Standard, fast, compatible
- bf16: Better range, needs newer GPU
- fp32: Full precision, slow

**Gradient Checkpointing:**
- Trades computation for memory
- Enables higher resolution
- 30-50% memory savings

## 1.4 First Look at Kohya Configuration

### Basic Configuration Structure

```json
{
    "LoRA_type": "Standard",
    "network_dim": 32,
    "network_alpha": 16,
    "learning_rate": "5e-05",
    "train_batch_size": 1,
    "epoch": 10,
    "save_every_n_epochs": 2,
    "mixed_precision": "fp16",
    "optimizer": "AdamW8bit"
}
```

### Configuration Sections

1. **Model Settings**
   - Base model selection
   - Model version (SD 1.5, SDXL, etc.)

2. **Network Settings**
   - LoRA type and dimensions
   - Alpha values
   - Module selection

3. **Training Settings**
   - Learning rates
   - Schedulers
   - Optimizers

4. **Data Settings**
   - Resolution
   - Augmentation
   - Captions

5. **Output Settings**
   - Save frequency
   - File format
   - Naming

## 1.5 Practice Exercise

### Exercise 1: Configuration Analysis

Compare these two configurations and identify the differences:

**Config A:**
```json
{
    "network_dim": 64,
    "learning_rate": "1e-04",
    "epoch": 20
}
```

**Config B:**
```json
{
    "network_dim": 16,
    "learning_rate": "5e-05",
    "epoch": 10
}
```

**Questions:**
1. Which will produce a larger LoRA file?
2. Which will train faster?
3. Which is more likely to overfit?

### Exercise 2: Parameter Calculation

Given:
- 20 images
- 10 repeats per image
- Batch size of 1
- 15 epochs

Calculate:
1. Total steps
2. Steps per epoch
3. Training time estimate (2 sec/step)

## 1.6 Key Takeaways

1. LoRA trains a small "adapter" instead of the full model
2. Rank (network_dim) controls capacity vs. file size
3. Learning rate is crucial - start conservative
4. Different LoRA types suit different use cases
5. Configuration is key to successful training

## Next Module Preview

In Module 2, we'll train your first LoRA from start to finish, covering:
- Dataset preparation
- Caption writing
- Training execution
- Testing and evaluation

## Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Kohya Documentation](https://github.com/kohya-ss/sd-scripts)
- [Community LoRA Collection](https://civitai.com/models)