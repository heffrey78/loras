# Cloud Training Guide for LoRA

## Overview

Training LoRAs in the cloud offers significant advantages for users with limited VRAM:
- Access to high-end GPUs (RTX 4090, A100)
- No local resource constraints
- Parallel training capabilities
- Cost-effective for occasional training

## Cloud Service Comparison

### RunPod (Recommended)

**Advantages:**
- Pre-built Kohya templates
- Reliable infrastructure
- Flexible pricing
- Easy file management

**GPU Options:**
- RTX 4090 24GB: $0.40-0.60/hour
- RTX 3090 24GB: $0.30-0.50/hour
- A100 40GB: $1.10-1.30/hour
- A100 80GB: $1.50-2.00/hour

**Setup Process:**
```bash
# 1. Create RunPod account
# 2. Deploy pod with Kohya template
# 3. Connect via web terminal or SSH

# Quick setup commands:
cd /workspace
git clone https://github.com/kohya-ss/sd-scripts
cd sd-scripts
pip install -r requirements.txt
```

### Google Colab Pro+

**Advantages:**
- Familiar Jupyter interface
- A100 access with Pro+
- Good for experimentation
- Integrated with Google Drive

**Pricing:**
- Pro: $9.99/month (limited A100)
- Pro+: $49.99/month (priority A100)

**Setup Notebook:**
```python
# Install Kohya in Colab
!git clone https://github.com/kohya-ss/sd-scripts
%cd sd-scripts
!pip install -r requirements.txt

# Mount Google Drive for data
from google.colab import drive
drive.mount('/content/drive')
```

### Vast.ai

**Advantages:**
- Cheapest option
- Wide GPU selection
- Spot pricing

**Disadvantages:**
- Variable reliability
- Setup complexity
- Limited support

**Pricing:**
- RTX 3090: $0.20-0.40/hour
- RTX 4090: $0.30-0.50/hour
- A100: $0.80-1.20/hour

### Lambda Labs

**Advantages:**
- Enterprise reliability
- Consistent performance
- Good support

**Pricing:**
- A100 40GB: $1.10/hour
- A100 80GB: $1.65/hour
- H100: $2.50/hour

## RunPod Setup (Detailed)

### Step 1: Account and Pod Creation

1. **Sign up** at runpod.io
2. **Add payment method** (credit required)
3. **Browse templates** → Find "Kohya SS"
4. **Select GPU** based on budget/needs
5. **Deploy pod** with adequate storage (50GB+)

### Step 2: Environment Setup

```bash
# Connect to pod via SSH or web terminal
ssh root@<pod-ip> -p <port>

# Navigate to workspace
cd /workspace

# Verify Kohya installation
ls sd-scripts/

# Test GPU
nvidia-smi
```

### Step 3: File Upload Methods

**Method 1: RunPod File Manager**
- Use web interface
- Drag and drop files
- Good for small datasets

**Method 2: SCP/RSYNC**
```bash
# From local machine
scp -P <port> -r ./training_data/ root@<pod-ip>:/workspace/data/

# Or using rsync
rsync -avz -e "ssh -p <port>" ./training_data/ root@<pod-ip>:/workspace/data/
```

**Method 3: Cloud Storage**
```bash
# Upload to Google Drive, then download on pod
pip install gdown
gdown <google-drive-share-url>

# Or use AWS S3
aws s3 sync s3://your-bucket/training-data ./data/
```

## Training Configuration for Cloud

### Optimized Cloud Config

```json
{
    "LoRA_type": "Standard",
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    
    // Take advantage of high VRAM
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "max_resolution": "1024,1024",
    "network_dim": 128,
    "network_alpha": 64,
    
    // Faster training with more resources
    "gradient_checkpointing": false,
    "mixed_precision": "bf16",
    "cache_latents": true,
    "cache_text_encoder_outputs": true,
    
    // Parallel processing
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": true,
    
    // Optimization
    "optimizer": "AdamW",
    "learning_rate": "1e-4",
    "lr_scheduler": "cosine_with_restarts",
    "epoch": 20,
    
    // Output
    "save_every_n_epochs": 2,
    "save_precision": "fp16"
}
```

### Multi-GPU Training

```bash
# For pods with multiple GPUs
accelerate config

# Configure for multi-GPU
# Then launch with accelerate
accelerate launch train_network.py --config_file config.toml
```

## Cost Optimization Strategies

### 1. Efficient Resource Usage

**Pre-processing Locally:**
```bash
# Do data prep on local machine
- Image resizing/cropping
- Caption generation
- Dataset validation

# Upload only final, optimized dataset
```

**Batch Multiple Trainings:**
```bash
# Train multiple LoRAs in single session
training_queue = [
    "character_lora_config.json",
    "style_lora_config.json", 
    "concept_lora_config.json"
]

for config in training_queue:
    train_lora(config)
```

### 2. Smart GPU Selection

**For Face LoRAs:** RTX 4090 24GB ($0.50/hr × 2hrs = $1.00)
**For Style LoRAs:** RTX 3090 24GB ($0.35/hr × 1hr = $0.35)
**For Complex Training:** A100 40GB ($1.20/hr × 1.5hrs = $1.80)

### 3. Spot Instance Usage

```python
# Monitor spot prices on Vast.ai
def find_cheapest_gpu(min_vram=24):
    # Query Vast.ai API
    offers = get_gpu_offers(vram_min=min_vram)
    return min(offers, key=lambda x: x['price'])
```

## Automated Cloud Training Pipeline

### Deployment Script

```python
import runpod
import time

class CloudTrainer:
    def __init__(self, api_key):
        self.client = runpod.API(api_key)
        
    def deploy_training_pod(self, config):
        # Create pod
        pod = self.client.create_pod(
            name="lora-training",
            image_name="kohya-ss:latest",
            gpu_type="RTX4090",
            container_disk_in_gb=50
        )
        
        # Wait for ready
        self.wait_for_ready(pod['id'])
        
        # Upload data
        self.upload_training_data(pod['id'], config['dataset'])
        
        # Start training
        self.start_training(pod['id'], config)
        
        return pod['id']
    
    def monitor_training(self, pod_id):
        while True:
            logs = self.client.get_pod_logs(pod_id)
            
            if "Training completed" in logs:
                break
                
            time.sleep(30)
        
        # Download results
        self.download_results(pod_id)
        
        # Terminate pod
        self.client.terminate_pod(pod_id)
```

### Training Orchestrator

```python
class TrainingOrchestrator:
    def __init__(self):
        self.queue = []
        self.results = {}
        
    def add_training_job(self, config):
        self.queue.append({
            'id': str(uuid.uuid4()),
            'config': config,
            'status': 'queued'
        })
    
    def process_queue(self):
        trainer = CloudTrainer(API_KEY)
        
        for job in self.queue:
            if job['status'] == 'queued':
                # Deploy and train
                pod_id = trainer.deploy_training_pod(job['config'])
                
                # Monitor
                trainer.monitor_training(pod_id)
                
                # Update status
                job['status'] = 'completed'
                
    def batch_train_loras(self, configs):
        for config in configs:
            self.add_training_job(config)
        
        self.process_queue()
```

## Data Management

### Efficient Upload/Download

```bash
# Compress before upload
tar -czf training_data.tar.gz ./training_data/
scp training_data.tar.gz root@pod:/workspace/

# Extract on cloud
tar -xzf training_data.tar.gz

# Compress results for download
tar -czf lora_results.tar.gz ./outputs/
scp root@pod:/workspace/lora_results.tar.gz ./
```

### Cloud Storage Integration

```python
# AWS S3 integration
import boto3

def sync_to_cloud(local_path, s3_bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, s3_bucket, s3_key)

def sync_from_cloud(s3_bucket, s3_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_key, local_path)

# Usage in training pipeline
def cloud_training_workflow(dataset_path, config):
    # Upload dataset
    sync_to_cloud(dataset_path, "training-bucket", "dataset.tar.gz")
    
    # Train on cloud (dataset downloads from S3)
    result = train_on_cloud(config)
    
    # Download results
    sync_from_cloud("training-bucket", "results.tar.gz", "./results.tar.gz")
```

## Monitoring and Logging

### Real-time Monitoring

```python
class TrainingMonitor:
    def __init__(self, pod_id):
        self.pod_id = pod_id
        self.metrics = []
        
    def monitor_training(self):
        while True:
            logs = get_pod_logs(self.pod_id)
            
            # Parse training metrics
            if "Loss:" in logs:
                loss = extract_loss(logs)
                self.metrics.append({
                    'timestamp': time.time(),
                    'loss': loss
                })
            
            # Check for completion
            if "Training finished" in logs:
                break
                
            time.sleep(10)
    
    def send_notification(self, message):
        # Discord webhook notification
        webhook_url = "your-discord-webhook"
        requests.post(webhook_url, json={"content": message})
```

### Cost Tracking

```python
class CostTracker:
    def __init__(self):
        self.sessions = []
        
    def start_session(self, gpu_type, hourly_rate):
        session = {
            'start_time': time.time(),
            'gpu_type': gpu_type,
            'hourly_rate': hourly_rate
        }
        self.sessions.append(session)
        return len(self.sessions) - 1
    
    def end_session(self, session_id):
        session = self.sessions[session_id]
        session['end_time'] = time.time()
        
        duration_hours = (session['end_time'] - session['start_time']) / 3600
        session['cost'] = duration_hours * session['hourly_rate']
        
        return session['cost']
```

## Security Best Practices

### API Key Management

```python
# Use environment variables
import os

API_KEY = os.getenv('RUNPOD_API_KEY')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')

# Never commit keys to git
# Use .env files locally
# Use secrets manager in production
```

### Data Privacy

```bash
# Encrypt sensitive data before upload
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 --force-mdc \
    --symmetric training_data.tar.gz

# Upload encrypted file
# Decrypt on cloud instance
gpg --decrypt training_data.tar.gz.gpg > training_data.tar.gz
```

## Troubleshooting

### Common Issues

**Connection Timeouts:**
```bash
# Use screen/tmux for long sessions
screen -S training
# Start training
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

**Out of Storage:**
```bash
# Monitor disk usage
df -h

# Clean up during training
rm -rf /tmp/*
rm -rf ~/.cache/*
```

**Network Issues:**
```bash
# Test connection
ping google.com

# Check bandwidth
speedtest-cli

# Resume interrupted downloads
wget -c <url>
```

## Cost Comparison

### Training Cost Examples

**Face LoRA (20 epochs, 15 images):**
- Local (RTX 3070 8GB): Free, 4 hours
- RunPod (RTX 4090): $1.00, 1 hour
- Colab Pro+: $49.99/month, 1.5 hours

**Style LoRA (15 epochs, 30 images):**
- Local: Limited by VRAM
- RunPod (RTX 3090): $0.70, 2 hours
- Lambda (A100): $2.20, 1 hour

**Complex Character (25 epochs, 50 images):**
- Local: Not feasible
- RunPod (A100 40GB): $3.60, 3 hours
- AWS (p3.2xlarge): $4.80, 3 hours

### ROI Calculation

```python
def calculate_training_roi(local_time, cloud_time, cloud_cost, hourly_rate=50):
    time_saved = local_time - cloud_time
    value_of_time = time_saved * hourly_rate
    net_benefit = value_of_time - cloud_cost
    return net_benefit

# Example: 4hr local vs 1hr cloud at $1.00
roi = calculate_training_roi(4, 1, 1.00, 50)  # $149 net benefit
```

## Next Steps

1. **Set up RunPod account** and test basic training
2. **Create automation scripts** for your common workflows  
3. **Experiment with different GPU types** to find optimal cost/performance
4. **Build monitoring system** for training progress
5. **Integrate with ComfyUI** for end-to-end workflow

Cloud training opens up possibilities for high-quality LoRAs that would be impossible on 8GB VRAM locally!