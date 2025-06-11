# Module 7: Troubleshooting Guide

## 7.1 Common Training Failures

### CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Diagnostic Steps:**
```python
import torch

def diagnose_oom():
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")
    
    # Find memory hogs
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(f"{type(obj).__name__}: {obj.size()}, {obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
```

**Solutions:**
```json
{
    // Immediate fixes
    "gradient_checkpointing": true,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    
    // Resolution reduction
    "max_resolution": "512,512",
    
    // Network size reduction
    "network_dim": 16,
    
    // Aggressive caching
    "cache_latents": true,
    "cache_latents_to_disk": true,
    "cache_text_encoder_outputs": true,
    "cache_text_encoder_outputs_to_disk": true
}
```

### NaN/Inf Loss

**Symptoms:**
```
Loss: nan, Step: 245
Warning: Gradient overflow detected
```

**Debugging:**
```python
class NaNDebugger:
    def __init__(self, model):
        self.model = model
        self.register_hooks()
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            module.register_forward_hook(
                lambda m, i, o, n=name: self.check_nan(n, o)
            )
    
    def check_nan(self, name, output):
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"NaN/Inf detected in {name}")
            print(f"Output stats: min={output.min()}, max={output.max()}")
            raise ValueError("NaN detected!")
```

**Solutions:**
```json
{
    // Stabilization settings
    "mixed_precision": "fp32",  // Temporarily
    "learning_rate": "1e-6",    // Much lower
    "gradient_clip_norm": 0.5,
    "max_grad_norm": 1.0,
    
    // Loss scaling
    "scale_weight_norms": 1.0,
    "min_snr_gamma": 5,
    
    // Disable problematic features
    "noise_offset": 0,
    "adaptive_noise_scale": 0
}
```

### Gradient Explosion

**Symptoms:**
- Loss suddenly spikes
- Training becomes unstable
- Model outputs noise

**Prevention:**
```python
def gradient_monitor(model, threshold=10.0):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    
    if total_norm > threshold:
        print(f"WARNING: Gradient norm {total_norm:.2f} exceeds threshold")
        return True
    return False
```

## 7.2 Quality Issues

### Overfitting

**Symptoms:**
- Exact recreation of training images
- No flexibility in generation
- "Burned" backgrounds

**Diagnosis:**
```python
def check_overfitting(lora_path, training_images, test_prompts):
    # Compare outputs to training data
    similarities = []
    
    for prompt in test_prompts:
        generated = generate_image(prompt, lora_path)
        
        for train_img in training_images:
            similarity = calculate_similarity(generated, train_img)
            similarities.append(similarity)
    
    overfitting_score = np.mean(similarities)
    return overfitting_score > 0.95  # Too similar
```

**Solutions:**
```json
{
    // Regularization
    "caption_dropout_rate": 0.1,
    "caption_dropout_every_n_epochs": 2,
    "shuffle_caption": true,
    
    // Early stopping
    "epoch": 10,  // Reduce from 20
    
    // Dataset augmentation
    "color_aug": true,
    "flip_aug": true,
    "random_crop": true,
    
    // Learning rate decay
    "lr_scheduler": "cosine",
    "lr_scheduler_num_cycles": 3
}
```

### Underfitting

**Symptoms:**
- Weak LoRA effect
- Requires very high strength
- Doesn't capture subject

**Solutions:**
```json
{
    // Increase capacity
    "network_dim": 64,  // From 32
    "network_alpha": 32,
    
    // More training
    "epoch": 20,
    "learning_rate": "1e-4",
    
    // Better optimization
    "optimizer": "AdamW",
    "lr_warmup_steps": 100
}
```

### Color Shift

**Symptoms:**
- Consistent color bias
- Tinted outputs
- Loss of color variety

**Diagnosis:**
```python
def analyze_color_shift(generated_images):
    color_histograms = []
    
    for img in generated_images:
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Analyze color channels
        l_mean, a_mean, b_mean = lab.mean(axis=(0,1))
        
        color_histograms.append({
            'luminance': l_mean,
            'green_red': a_mean,  # Negative = green, Positive = red
            'blue_yellow': b_mean  # Negative = blue, Positive = yellow
        })
    
    # Check for consistent bias
    return analyze_bias(color_histograms)
```

**Solutions:**
```json
{
    // Color normalization
    "color_aug": false,  // If causing issues
    "normalize_input": true,
    
    // Dataset balance
    "check_color_distribution": true,
    
    // Training adjustments
    "v_parameterization": false,
    "scale_v_pred_loss_like_noise_pred": true
}
```

## 7.3 Performance Issues

### Slow Training

**Symptoms:**
- <1 it/s on capable hardware
- Low GPU utilization
- CPU bottleneck

**Profiling:**
```python
import cProfile
import pstats

def profile_training_speed():
    profiler = cProfile.Profile()
    
    profiler.enable()
    # Run training for 10 steps
    train_steps(10)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

**Solutions:**
```json
{
    // I/O optimization
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": true,
    "cache_latents": true,
    
    // Computation optimization
    "xformers": true,
    "sdpa": true,
    "torch_compile": true,
    
    // Batch optimization
    "train_batch_size": 2,  // If memory allows
    "gradient_accumulation_steps": 2
}
```

### Memory Fragmentation

**Symptoms:**
- OOM despite low reported usage
- Gradually increasing memory
- Requires restarts

**Mitigation:**
```python
class MemoryManager:
    def __init__(self, threshold_gb=0.5):
        self.threshold = threshold_gb * 1024**3
        
    def check_fragmentation(self):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        fragmentation = (reserved - allocated) / reserved
        
        if fragmentation > 0.3:  # 30% fragmentation
            print(f"High fragmentation: {fragmentation:.1%}")
            self.defragment()
    
    def defragment(self):
        # Force cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Aggressive cleanup
        gc.collect()
        torch.cuda.empty_cache()
```

## 7.4 Data Issues

### Corrupted Images

**Detection:**
```python
from PIL import Image
import hashlib

def validate_dataset(dataset_path):
    corrupted = []
    
    for img_path in Path(dataset_path).glob("**/*.{png,jpg,jpeg,webp}"):
        try:
            # Try to open and verify
            img = Image.open(img_path)
            img.verify()
            
            # Check for truncation
            img = Image.open(img_path)
            img.load()
            
            # Check minimum size
            if min(img.size) < 256:
                print(f"Warning: {img_path} is too small: {img.size}")
                
        except Exception as e:
            corrupted.append((img_path, str(e)))
            
    return corrupted
```

### Caption Mismatches

**Validation:**
```python
def validate_caption_pairs(dataset_path):
    issues = []
    
    for img_path in Path(dataset_path).glob("**/*.png"):
        txt_path = img_path.with_suffix('.txt')
        
        if not txt_path.exists():
            issues.append(f"Missing caption: {img_path}")
            continue
            
        # Check caption content
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            
        if not caption:
            issues.append(f"Empty caption: {txt_path}")
        elif len(caption) < 3:
            issues.append(f"Caption too short: {txt_path}")
        elif not caption.startswith(expected_trigger):
            issues.append(f"Missing trigger word: {txt_path}")
            
    return issues
```

## 7.5 Configuration Errors

### JSON Parsing Errors

**Common Issues:**
```python
def validate_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON Error at line {e.lineno}, column {e.colno}")
        print(f"Message: {e.msg}")
        
        # Show context
        with open(config_path, 'r') as f:
            lines = f.readlines()
            error_line = e.lineno - 1
            
            for i in range(max(0, error_line - 2), 
                          min(len(lines), error_line + 3)):
                marker = ">>>" if i == error_line else "   "
                print(f"{marker} {i+1}: {lines[i].rstrip()}")
```

### Type Mismatches

**Validation:**
```python
CONFIG_SCHEMA = {
    "network_dim": (int, lambda x: 1 <= x <= 256),
    "learning_rate": (str, lambda x: float(x) > 0),
    "train_batch_size": (int, lambda x: x >= 1),
    "epoch": (int, lambda x: x >= 1),
    "mixed_precision": (str, lambda x: x in ["no", "fp16", "bf16"])
}

def validate_config_types(config):
    errors = []
    
    for key, (expected_type, validator) in CONFIG_SCHEMA.items():
        if key in config:
            value = config[key]
            
            # Type check
            if not isinstance(value, expected_type):
                errors.append(f"{key}: expected {expected_type.__name__}, got {type(value).__name__}")
            
            # Value check
            elif not validator(value):
                errors.append(f"{key}: invalid value {value}")
                
    return errors
```

## 7.6 Recovery Strategies

### Checkpoint Recovery

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def save_checkpoint(self, model, optimizer, epoch, step):
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'timestamp': time.time()
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, path)
        
        # Keep only last 3 checkpoints
        self.cleanup_old_checkpoints()
        
    def recover_from_checkpoint(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoints:
            return None
            
        latest = checkpoints[-1]
        print(f"Recovering from {latest}")
        
        return torch.load(latest)
```

### Partial Training Recovery

```python
def resume_training(config_path, checkpoint_path=None):
    config = load_config(config_path)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        
        # Adjust configuration
        config['resume_from_checkpoint'] = checkpoint_path
        config['start_epoch'] = checkpoint['epoch']
        config['start_step'] = checkpoint['step']
        
        # Reduce total epochs
        config['epoch'] = config['epoch'] - checkpoint['epoch']
        
    return config
```

## 7.7 Debugging Tools

### Interactive Debugger

```python
class InteractiveDebugger:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.breakpoints = {}
        
    def add_breakpoint(self, module_name, condition=None):
        self.breakpoints[module_name] = condition
        
    def debug_step(self):
        for batch in self.dataloader:
            # Forward with hooks
            with self.debugging_context():
                output = self.model(batch)
                
            # Interactive inspection
            import ipdb; ipdb.set_trace()
            
            break
```

### Loss Analyzer

```python
class LossAnalyzer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.loss_history = []
        
    def analyze(self, loss):
        self.loss_history.append(loss)
        
        if len(self.loss_history) >= self.window_size:
            recent = self.loss_history[-self.window_size:]
            
            # Detect issues
            if np.std(recent) < 1e-6:
                print("WARNING: Loss plateaued")
            
            if any(np.isnan(recent)) or any(np.isinf(recent)):
                print("ERROR: NaN/Inf in loss")
            
            if recent[-1] > np.mean(recent) * 2:
                print("WARNING: Loss spike detected")
```

## 7.8 Common Error Messages

### Error Reference

**"Expected all tensors to be on the same device"**
```python
# Fix: Ensure all inputs on same device
def ensure_same_device(tensors, device='cuda'):
    return [t.to(device) if torch.is_tensor(t) else t for t in tensors]
```

**"cuDNN error: CUDNN_STATUS_NOT_SUPPORTED"**
```python
# Fix: Disable cuDNN for specific operations
torch.backends.cudnn.enabled = False
# or
with torch.backends.cudnn.flags(enabled=False):
    # Your operation
```

**"RuntimeError: element 0 of tensors does not require grad"**
```python
# Fix: Ensure model in training mode
model.train()
# and check requires_grad
for param in model.parameters():
    param.requires_grad = True
```

## 7.9 Prevention Checklist

### Pre-Training Validation
- [ ] Validate dataset integrity
- [ ] Check caption-image pairs
- [ ] Verify configuration JSON
- [ ] Test with minimal dataset
- [ ] Check GPU memory availability
- [ ] Validate base model path

### During Training Monitoring
- [ ] Watch loss progression
- [ ] Monitor GPU memory
- [ ] Check gradient norms
- [ ] Validate sample outputs
- [ ] Track training speed
- [ ] Save regular checkpoints

### Post-Training Verification
- [ ] Test LoRA loading
- [ ] Verify output quality
- [ ] Check file integrity
- [ ] Test with various prompts
- [ ] Compare to baseline
- [ ] Document issues

## 7.10 Emergency Procedures

### Training Crash Recovery

1. **Don't Panic** - Most crashes are recoverable
2. **Save Logs** - Copy all error messages
3. **Check Checkpoints** - Find latest valid checkpoint
4. **Diagnose Issue** - Use debugging tools
5. **Adjust Config** - Fix identified problems
6. **Resume Training** - Start from checkpoint

### Quality Failure Recovery

1. **Analyze Outputs** - Identify specific issues
2. **Review Dataset** - Check for problems
3. **Adjust Parameters** - Based on analysis
4. **A/B Test** - Compare approaches
5. **Incremental Changes** - One variable at a time

## Practice Exercises

### Exercise 1: Debugging Challenge
Given a training that produces NaN loss:
1. Add debugging hooks
2. Identify the layer causing issues
3. Fix the configuration
4. Successfully complete training

### Exercise 2: Recovery Simulation
1. Intentionally crash training mid-way
2. Implement checkpoint recovery
3. Resume and complete training
4. Verify final quality matches expected

### Exercise 3: Optimization Debug
Given a training running at 0.5 it/s:
1. Profile the bottleneck
2. Implement fixes
3. Achieve >2 it/s
4. Maintain quality

## Next Module Preview

Module 8 covers production pipelines:
- Automation strategies
- CI/CD for LoRA training
- Batch processing
- Quality assurance