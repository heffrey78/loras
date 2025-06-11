# Module 3: Dataset Mastery

## 3.1 Advanced Dataset Theory

### The Foundation of Quality

Your dataset is the single most important factor in LoRA quality. No amount of parameter tuning can fix a poor dataset.

**Dataset Impact:**
- 70% of final quality
- Determines flexibility
- Sets capability limits
- Affects file size

### Dataset Philosophy

**Quality Hierarchy:**
1. Consistency
2. Diversity
3. Resolution
4. Quantity

**The 80/20 Rule:**
- 80% consistent style/quality
- 20% edge cases/variations

## 3.2 Image Curation Strategies

### Selection Criteria

**Technical Requirements:**
```python
# Minimum standards
resolution >= 512x512
file_format in ['png', 'jpg', 'webp']
no_compression_artifacts = True
proper_exposure = True
sharp_focus = True
```

**Content Requirements:**
- Clear subject visibility
- Varied poses/angles
- Consistent quality
- No duplicates
- Balanced representation

### Advanced Curation Techniques

**1. Histogram Analysis**
```python
import cv2
import numpy as np

def analyze_image_quality(image_path):
    img = cv2.imread(image_path)
    
    # Check exposure
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    
    # Balanced histogram = good exposure
    low_exposure = np.sum(hist[:50])
    high_exposure = np.sum(hist[200:])
    
    return {
        'underexposed': low_exposure > len(gray) * 0.5,
        'overexposed': high_exposure > len(gray) * 0.5,
        'balanced': abs(low_exposure - high_exposure) < len(gray) * 0.1
    }
```

**2. Similarity Detection**
```python
def find_duplicates(image_folder, threshold=0.95):
    # Use perceptual hashing
    from imagehash import average_hash
    from PIL import Image
    
    hashes = {}
    for img_path in image_folder:
        hash = average_hash(Image.open(img_path))
        # Group similar images
        for existing_hash, paths in hashes.items():
            similarity = 1 - (hash - existing_hash) / len(hash)
            if similarity > threshold:
                paths.append(img_path)
                break
        else:
            hashes[hash] = [img_path]
```

### Dataset Balancing

**Aspect Ratio Distribution:**
- Portrait: 30-40%
- Landscape: 30-40%
- Square: 20-40%

**Content Distribution:**
- Close-ups: 20-30%
- Medium shots: 40-50%
- Wide shots: 20-30%

## 3.3 Resolution and Preprocessing

### Multi-Resolution Strategy

**Benefits:**
- Better generalization
- Flexible inference
- Reduced artifacts
- Improved details

**Implementation:**
```json
{
    "enable_bucket": true,
    "min_bucket_reso": 512,
    "max_bucket_reso": 1024,
    "bucket_reso_steps": 64,
    "bucket_no_upscale": false
}
```

### Smart Preprocessing Pipeline

```python
import cv2
from PIL import Image
import numpy as np

class DatasetPreprocessor:
    def __init__(self, target_size=768, quality=95):
        self.target_size = target_size
        self.quality = quality
    
    def process_image(self, input_path, output_path):
        # Load image
        img = Image.open(input_path)
        
        # EXIF orientation fix
        img = self.fix_orientation(img)
        
        # Smart crop to subject
        img = self.smart_crop(img)
        
        # High-quality resize
        img = self.high_quality_resize(img)
        
        # Color correction
        img = self.auto_color_correct(img)
        
        # Save with optimal settings
        img.save(output_path, quality=self.quality, optimize=True)
    
    def smart_crop(self, img):
        # Detect subject using vision model
        # Crop with padding
        pass
    
    def high_quality_resize(self, img):
        # Lanczos for downscale
        # ESRGAN for upscale
        pass
```

### Resolution Guidelines

**By Training Type:**
- Face/Character: 768×768 optimal
- Style: 512×512 to 1024×1024
- Objects: 640×640 standard
- Concepts: Mixed resolutions

**By VRAM:**
- 6GB: 512×512 max
- 8GB: 768×768 comfortable
- 12GB: 1024×1024 feasible
- 24GB: 1024×1024+ with batch>1

## 3.4 Advanced Captioning

### Caption Engineering

**Structured Format:**
```
[trigger], [subject_type], [key_features], [action/pose], [environment], [style], [quality]
```

**Examples:**
```
# Character
john_doe, man, brown hair and beard, sitting at desk, modern office, photorealistic, high quality

# Style
gothic_style, architectural painting, cathedral with spires, dramatic lighting, oil on canvas, masterpiece

# Object
vintage_typewriter, antique typewriter, brass keys, on wooden desk, product photography, professional
```

### Captioning Strategies

**1. Hierarchical Captioning**
```
Level 1 (Simple): trigger_word
Level 2 (Basic): trigger_word, main subject
Level 3 (Detailed): trigger_word, subject, features, context
Level 4 (Complete): Full structured caption
```

**2. Weighted Captioning**
```
(trigger_word:1.2), (important_feature:1.1), normal_feature, (style:0.8)
```

**3. Negative Captioning**
```
trigger_word, subject, [avoid:blurry:0.5], [avoid:artifacts:0.5]
```

### Automated Caption Pipeline

```python
class AdvancedCaptioner:
    def __init__(self):
        self.blip_model = load_blip_model()
        self.clip_model = load_clip_model()
        self.custom_tagger = load_tagger()
    
    def generate_caption(self, image_path, trigger_word):
        # Base caption from BLIP
        base_caption = self.blip_model.caption(image_path)
        
        # Extract features with CLIP
        features = self.clip_model.extract_features(image_path)
        
        # Custom tags
        tags = self.custom_tagger.tag(image_path)
        
        # Combine intelligently
        return self.combine_captions(
            trigger_word, base_caption, features, tags
        )
    
    def combine_captions(self, trigger, base, features, tags):
        # Priority order
        caption_parts = [trigger]
        
        # Add subject from base
        caption_parts.append(self.extract_subject(base))
        
        # Add top features
        caption_parts.extend(features[:3])
        
        # Add relevant tags
        caption_parts.extend(self.filter_tags(tags))
        
        return ", ".join(caption_parts)
```

## 3.5 Dataset Augmentation

### Smart Augmentation

**When to Augment:**
- Limited dataset (<20 images)
- Lack of variety
- Specific pose gaps
- Style consistency needed

**Safe Augmentations:**
```python
safe_augmentations = {
    "horizontal_flip": True,  # For non-text images
    "rotation": (-5, 5),      # Subtle rotation
    "brightness": (0.9, 1.1), # Minor adjustment
    "contrast": (0.9, 1.1),   # Minor adjustment
    "crop": (0.9, 1.0),       # Slight zoom
}
```

**Risky Augmentations:**
```python
risky_augmentations = {
    "color_jitter": False,    # Can change style
    "perspective": False,     # Can distort
    "aggressive_crop": False, # Loses information
    "filters": False,         # Changes style
}
```

### Augmentation Implementation

```python
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    ShiftScaleRotate, RandomCrop, Resize
)

class SafeAugmenter:
    def __init__(self, preserve_aspect=True):
        self.transform = Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                p=0.3
            ),
        ])
    
    def augment_dataset(self, input_dir, output_dir, augment_factor=2):
        for image_path in Path(input_dir).glob("*.png"):
            img = cv2.imread(str(image_path))
            
            # Original
            cv2.imwrite(output_dir / image_path.name, img)
            
            # Augmented versions
            for i in range(augment_factor - 1):
                augmented = self.transform(image=img)['image']
                new_name = f"{image_path.stem}_aug{i}{image_path.suffix}"
                cv2.imwrite(output_dir / new_name, augmented)
```

## 3.6 Regularization Images

### Understanding Regularization

**Purpose:**
- Prevent overfitting
- Maintain model knowledge
- Improve generalization
- Balance training

**When to Use:**
- Small datasets (<20 images)
- Highly specific subjects
- Style preservation needed
- Preventing attribute bleed

### Regularization Strategy

```
training_data/
├── img/
│   └── 10_trigger_word/
│       ├── subject_001.png
│       └── subject_001.txt
└── reg/
    └── 1_class_word/
        ├── regular_001.png
        └── regular_001.txt
```

**Selection Criteria:**
- Same class as subject
- Different individuals
- Similar quality
- Diverse representation

### Regularization Configuration

```json
{
    "prior_loss_weight": 1.0,
    "reg_data_dir": "./training_data/reg",
    
    // For faces
    "class_prompt": "person",
    "num_class_images": 100,
    
    // For styles
    "class_prompt": "artwork",
    "num_class_images": 200
}
```

## 3.7 Dataset Validation

### Automated Validation Script

```python
class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.issues = []
    
    def validate(self):
        self.check_structure()
        self.check_images()
        self.check_captions()
        self.check_balance()
        self.generate_report()
    
    def check_structure(self):
        # Verify folder structure
        # Check naming convention
        # Ensure pairs exist
        pass
    
    def check_images(self):
        for img_path in self.dataset_path.glob("**/*.png"):
            # Resolution check
            # Quality check
            # Format check
            # Corruption check
            pass
    
    def check_captions(self):
        for txt_path in self.dataset_path.glob("**/*.txt"):
            # Length check
            # Trigger word check
            # Format check
            # Encoding check
            pass
    
    def generate_report(self):
        print("Dataset Validation Report")
        print(f"Total images: {self.total_images}")
        print(f"Issues found: {len(self.issues)}")
        for issue in self.issues:
            print(f"- {issue}")
```

### Quality Metrics

**Objective Metrics:**
- Resolution distribution
- Aspect ratio variety
- Color histogram spread
- Sharpness scores
- File size consistency

**Subjective Metrics:**
- Style consistency
- Subject clarity
- Pose diversity
- Background variety
- Lighting quality

## 3.8 Practice Exercises

### Exercise 1: Dataset Curation

Given 50 images of varying quality:
1. Select best 20 for training
2. Document selection criteria
3. Create augmentation plan
4. Write sample captions

### Exercise 2: Multi-Resolution Setup

1. Create buckets for 512, 640, 768, 896
2. Distribute images appropriately
3. Configure bucket training
4. Test impact on quality

### Exercise 3: A/B Testing

Create two datasets:
- Dataset A: 15 high-quality images
- Dataset B: 30 mixed-quality images

Train both and compare:
- Final quality
- Flexibility
- File size
- Training time

## 3.9 Advanced Tips

### Pro Dataset Strategies

1. **Progressive Dataset Building**
   - Start with core 10 images
   - Test and identify gaps
   - Add specific images to fill gaps
   - Iterate until satisfied

2. **Style Clustering**
   - Group similar images
   - Balance clusters
   - Caption by cluster
   - Train with awareness

3. **Synthetic Data**
   - Generate variations with img2img
   - Use ControlNet for poses
   - Carefully curate results
   - Mix with real data (20% max)

## Next Module Preview

Module 4 explores different LoRA architectures:
- LoCon for better style capture
- LoHa for efficient details
- LoKR for parameter efficiency
- Choosing the right architecture