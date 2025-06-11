# Module 6: Training Optimization

## 6.1 Hardware-Specific Optimization

### GPU Memory Hierarchy

Understanding your GPU's memory architecture is crucial for optimization:

```
GPU Memory Layout:
├── L1 Cache (per SM): ~128KB
├── L2 Cache (shared): 4-40MB
├── VRAM: 6-80GB
└── System RAM (fallback): 16-128GB
```

### VRAM Optimization by GPU

**6-8GB GPUs (RTX 3060, 3070 Mobile)**
```json
{
    "train_batch_size": 1,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "fp16",
    "max_resolution": "768,768",
    "cache_latents": true,
    "cache_latents_to_disk": true,
    "network_dim": 32,
    "xformers": true
}
```

**10-12GB GPUs (RTX 3080, 4070)**
```json
{
    "train_batch_size": 2,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "fp16",
    "max_resolution": "1024,1024",
    "cache_latents": true,
    "network_dim": 64,
    "xformers": true
}
```

**16-24GB GPUs (RTX 4090, 3090)**
```json
{
    "train_batch_size": 4,
    "gradient_checkpointing": false,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "max_resolution": "1024,1024",
    "cache_latents": true,
    "network_dim": 128,
    "sdpa": true
}
```

### Memory Profiling

```python
import torch
import nvidia_ml_py as nvml

class VRAMProfiler:
    def __init__(self):
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_memory_info(self):
        info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "total": info.total / 1024**3,  # GB
            "used": info.used / 1024**3,
            "free": info.free / 1024**3
        }
    
    def profile_training_step(self, train_fn):
        torch.cuda.empty_cache()
        before = self.get_memory_info()
        
        result = train_fn()
        
        after = self.get_memory_info()
        
        return {
            "peak_usage": after["used"],
            "step_allocation": after["used"] - before["used"],
            "remaining": after["free"]
        }
```

## 6.2 Speed Optimization

### Bottleneck Analysis

```python
import time
import torch.profiler

class TrainingProfiler:
    def profile_training(self, model, dataloader, steps=10):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=2, warmup=2, active=6
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
        ) as prof:
            for i, batch in enumerate(dataloader):
                if i >= steps:
                    break
                
                # Training step
                loss = model.training_step(batch)
                loss.backward()
                
                prof.step()
```

### Data Loading Optimization

```json
{
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": true,
    "pin_memory": true,
    "prefetch_factor": 2
}
```

**Custom Data Pipeline:**
```python
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size=1, num_workers=4):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
            # Custom sampler for better GPU utilization
            sampler=SmartSampler(dataset)
        )
    
    def prefetch_to_gpu(self, batch):
        # Move data to GPU while previous batch processes
        return {k: v.cuda(non_blocking=True) for k, v in batch.items()}
```

### Computation Optimization

**1. Attention Mechanisms**
```json
{
    // Choose based on GPU
    "xformers": true,           // General purpose, good compatibility
    "sdpa": true,              // PyTorch 2.0+, newer GPUs
    "flash_attention": true,    // Best performance, limited compatibility
    "memory_efficient_attention": true
}
```

**2. Gradient Accumulation Strategy**
```python
def optimized_gradient_accumulation(model, optimizer, accumulation_steps=4):
    # Reduce memory allocation frequency
    optimizer.zero_grad(set_to_none=True)
    
    for i in range(accumulation_steps):
        loss = compute_loss() / accumulation_steps
        loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

## 6.3 Multi-GPU Training

### Data Parallel Setup

```python
# DDP Setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    
    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
```

### Multi-GPU Configuration

```json
{
    // Accelerate config
    "multi_gpu": true,
    "num_processes": 2,
    "distributed_type": "MULTI_GPU",
    "gradient_accumulation_steps": 2,
    "fp16": true,
    "split_batches": false
}
```

### GPU Assignment Strategy

```python
# Optimal GPU assignment for different architectures
def assign_layers_to_gpus(model, gpu_memory_gb):
    assignments = {}
    
    if len(gpu_memory_gb) == 2:
        # 2 GPU setup
        if gpu_memory_gb[0] >= gpu_memory_gb[1]:
            # Put text encoder on smaller GPU
            assignments["text_encoder"] = 1
            assignments["unet"] = 0
        else:
            assignments["text_encoder"] = 0
            assignments["unet"] = 1
    
    return assignments
```

## 6.4 Disk and I/O Optimization

### Caching Strategy

```python
class SmartCacheManager:
    def __init__(self, cache_dir="./cache", max_size_gb=50):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_gb * 1024**3
        
    def cache_latents(self, dataset, vae):
        cache_path = self.cache_dir / "latents"
        cache_path.mkdir(exist_ok=True)
        
        for idx, image in enumerate(dataset):
            latent_path = cache_path / f"{idx}.pt"
            
            if not latent_path.exists():
                # Compute and cache
                with torch.no_grad():
                    latent = vae.encode(image).latent_dist.sample()
                torch.save(latent, latent_path)
    
    def get_cached_latent(self, idx):
        return torch.load(self.cache_dir / "latents" / f"{idx}.pt")
```

### Storage Optimization

```json
{
    // Efficient storage settings
    "save_precision": "fp16",
    "save_model_as": "safetensors",
    "save_state": false,  // Don't save optimizer state
    "save_every_n_epochs": 5,
    "save_last_n_epochs": 2,
    "save_last_n_steps": 1000
}
```

## 6.5 Training Scheduling

### Adaptive Scheduling

```python
class AdaptiveScheduler:
    def __init__(self, base_config):
        self.config = base_config
        self.history = []
        
    def adjust_based_on_loss(self, current_loss):
        self.history.append(current_loss)
        
        if len(self.history) > 10:
            # Check if plateauing
            recent_losses = self.history[-10:]
            if np.std(recent_losses) < 0.001:
                # Reduce learning rate
                self.config["learning_rate"] *= 0.5
                print(f"Plateauing detected, reducing LR to {self.config['learning_rate']}")
        
        # Check for instability
        if current_loss > self.history[-1] * 1.5:
            self.config["gradient_clip_norm"] = 0.5
            print("Instability detected, enabling gradient clipping")
```

### Dynamic Batch Sizing

```python
def dynamic_batch_size(gpu_memory_gb, resolution, network_dim):
    # Base memory usage (GB)
    base_memory = 2.5
    
    # Memory per image (GB)
    memory_per_image = (resolution / 512) ** 2 * 0.5
    
    # Network dimension impact
    network_factor = network_dim / 32
    
    # Calculate max batch size
    available_memory = gpu_memory_gb - base_memory
    max_batch = int(available_memory / (memory_per_image * network_factor))
    
    return max(1, max_batch)
```

## 6.6 Quality vs Speed Trade-offs

### Configuration Presets

**Speed Priority**
```json
{
    "name": "Speed Priority",
    "gradient_checkpointing": false,
    "cache_latents": true,
    "cache_text_encoder_outputs": true,
    "mixed_precision": "fp16",
    "xformers": true,
    "persistent_data_loader_workers": true,
    "compile_unet": true
}
```

**Quality Priority**
```json
{
    "name": "Quality Priority",
    "gradient_checkpointing": true,
    "mixed_precision": "fp32",
    "optimizer": "AdamW",
    "learning_rate": "1e-5",
    "min_snr_gamma": 5,
    "multires_noise_iterations": 8
}
```

**Balanced**
```json
{
    "name": "Balanced",
    "gradient_checkpointing": true,
    "mixed_precision": "fp16",
    "optimizer": "AdamW8bit",
    "cache_latents": true,
    "xformers": true,
    "min_snr_gamma": 5
}
```

## 6.7 Monitoring and Logging

### Comprehensive Monitoring

```python
class TrainingMonitor:
    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.metrics = defaultdict(list)
        
    def log_step(self, step, metrics):
        for key, value in metrics.items():
            self.metrics[key].append((step, value))
        
        # Real-time monitoring
        if step % 10 == 0:
            self.print_summary(step)
        
        # Checkpointing
        if step % 100 == 0:
            self.save_checkpoint(step)
    
    def print_summary(self, step):
        recent_loss = np.mean([v for s, v in self.metrics["loss"][-10:]])
        lr = self.metrics["lr"][-1][1]
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"Step {step}: Loss={recent_loss:.4f}, LR={lr:.2e}, VRAM={gpu_memory:.1f}GB")
```

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_training_step(self, step, loss, images=None):
        self.writer.add_scalar("Loss/train", loss, step)
        
        if images is not None:
            self.writer.add_images("Training/samples", images, step)
        
        # Log system metrics
        self.writer.add_scalar("System/gpu_memory_gb", 
                              torch.cuda.memory_allocated() / 1024**3, step)
        self.writer.add_scalar("System/gpu_utilization", 
                              torch.cuda.utilization(), step)
```

## 6.8 Production Pipeline

### Automated Training Pipeline

```python
class ProductionTrainer:
    def __init__(self, base_config):
        self.base_config = base_config
        self.validator = LoRAValidator()
        
    def train_with_validation(self, dataset):
        # Pre-training validation
        dataset_quality = self.validator.validate_dataset(dataset)
        if dataset_quality < 0.8:
            raise ValueError("Dataset quality too low")
        
        # Adaptive configuration
        config = self.optimize_config_for_hardware()
        
        # Training with monitoring
        model = self.train_with_monitoring(dataset, config)
        
        # Post-training validation
        lora_quality = self.validator.validate_lora(model)
        
        return model if lora_quality > 0.9 else None
    
    def optimize_config_for_hardware(self):
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory < 8:
            return self.get_low_memory_config()
        elif gpu_memory < 16:
            return self.get_medium_memory_config()
        else:
            return self.get_high_memory_config()
```

## 6.9 Optimization Checklists

### Pre-Training Optimization
- [ ] Profile baseline memory usage
- [ ] Enable appropriate attention mechanism
- [ ] Configure caching strategy
- [ ] Set optimal batch size
- [ ] Choose correct mixed precision
- [ ] Enable gradient checkpointing if needed
- [ ] Configure data loader workers

### During Training Optimization
- [ ] Monitor GPU utilization (target >90%)
- [ ] Watch for memory fragmentation
- [ ] Check for I/O bottlenecks
- [ ] Monitor loss stability
- [ ] Track training speed (it/s)
- [ ] Adjust parameters if needed

### Post-Training Optimization
- [ ] Analyze training metrics
- [ ] Identify bottlenecks
- [ ] Document optimal settings
- [ ] Create configuration template
- [ ] Plan improvements

## 6.10 Advanced Tips

### 1. Compilation (PyTorch 2.0+)
```python
# Compile model for faster training
import torch._dynamo as dynamo
model = torch.compile(model, mode="reduce-overhead")
```

### 2. Custom CUDA Kernels
```python
# For specific operations
from torch.utils.cpp_extension import load_inline

custom_kernel = load_inline(
    name='custom_ops',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['custom_attention']
)
```

### 3. Memory Pool Management
```python
# Reduce allocation overhead
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()
```

## Practice Exercises

### Exercise 1: Optimization Benchmark
Create benchmarks for your hardware:
1. Test different batch sizes
2. Compare attention mechanisms
3. Measure caching impact
4. Find optimal configuration

### Exercise 2: Multi-Resolution Optimization
Train same LoRA at:
- 512x512
- 768x768
- 1024x1024

Compare memory usage and quality.

### Exercise 3: Speed Challenge
Optimize training to achieve:
- <2 seconds per iteration
- <5GB VRAM usage
- Quality score >0.9

## Next Module Preview

Module 7 covers troubleshooting:
- Common training failures
- Debugging techniques
- Quality issues
- Recovery strategies