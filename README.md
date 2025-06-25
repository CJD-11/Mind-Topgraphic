# Mind-Topgraphic

# MiDaS Depth Scanner
## AI-Powered Artistic Video Generation with Depth-Based Scanning Effects

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![MiDaS](https://img.shields.io/badge/MiDaS-DPT_Large-orange.svg)](https://github.com/isl-org/MiDaS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Project Vision

MiDaS Depth Scanner transforms static images into dynamic artistic videos using AI-powered depth estimation. The system creates mesmerizing scanning animations that reveal depth contours through progressive red-line sweeps, producing cinematic 4K videos with sophisticated visual effects.

### Key Features

- **AI Depth Estimation**: Advanced MiDaS DPT_Large model for precise depth mapping
- **4K Video Output**: Professional-quality 3840x2160 resolution rendering
- **Dynamic Scanning Animation**: Customizable scanning patterns with depth-aware highlighting
- **Artistic Effects Pipeline**: Layered visual effects including glow, fade, and contour detection
- **Batch Processing**: Automated processing of multiple images
- **GPU Acceleration**: CUDA support for high-performance rendering
- **Configurable Parameters**: Extensive customization options for creative control

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for 4K processing)
- 10GB+ free disk space for output videos

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/midas-depth-scanner.git
   cd midas-depth-scanner
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU setup (optional but recommended):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Run the scanner:**
   ```bash
   python src/depth_scanner.py --input assets/sample_images --output output/videos
   ```

### Basic Usage

1. **Prepare Images:**
   - Place your images in the input folder
   - Supported formats: JPG, PNG, BMP, TIFF
   - Any resolution (automatically scaled to 4K)

2. **Configure Settings:**
   - Edit `config/default.json` or use command-line options
   - Adjust scanning speed, effects intensity, and output quality

3. **Generate Videos:**
   ```bash
   # Basic usage
   python src/depth_scanner.py --input path/to/images

   # With custom configuration
   python src/depth_scanner.py --input path/to/images --config config/artistic.json

   # High-quality mode
   python src/depth_scanner.py --input path/to/images --quality high --gpu
   ```

## Core Features

### AI-Powered Depth Estimation
- **MiDaS DPT_Large Model**: State-of-the-art monocular depth estimation
- **High Accuracy**: Precise depth maps from single images
- **Robust Performance**: Works across diverse image types and scenes
- **GPU Acceleration**: Optimized for CUDA-enabled hardware

### Dynamic Scanning Animation
- **Progressive Sweep**: Horizontal scanning with customizable speed
- **Depth-Aware Highlighting**: Contour detection based on depth layers
- **Multi-Loop Animation**: Repeating scan cycles for hypnotic effect
- **Smooth Transitions**: Anti-aliased rendering for professional quality

### Visual Effects Pipeline
- **Red Line Glow**: Animated scanning line with configurable intensity
- **Fade Persistence**: Gradual decay of previous scan traces
- **Depth Contours**: Multi-level depth visualization
- **Black & White Imprint**: Subtle depth layer accumulation
- **Gaussian Blur**: Soft glow effects and edge smoothing

### Professional Video Output
- **4K Resolution**: 3840x2160 pixel output for maximum quality
- **Configurable Frame Rate**: 24, 30, or 60 FPS support
- **Aspect Ratio Preservation**: Intelligent scaling and padding
- **High-Quality Codecs**: H.264/H.265 encoding options
- **Batch Processing**: Multiple images to video sequences

## ⚙️ Configuration

### Basic Configuration
```json
{
  "input_folder": "assets/sample_images",
  "output_folder": "output/videos",
  "fps": 30,
  "scan_loops": 2,
  "final_hold_seconds": 5,
  "target_resolution": [3840, 2160]
}
```

### Visual Effects Settings
```json
{
  "contour_levels": 150,
  "scan_band_height": 8,
  "fade_decay": 0.94,
  "bw_opacity": 0.12,
  "red_strength": 1.5,
  "blur_radius": 1.5
}
```

### Performance Settings
```json
{
  "device": "auto",
  "batch_size": 1,
  "memory_optimization": true,
  "parallel_processing": false
}
```

## Technical Architecture

### Core Components

**Depth Estimator**
```python
class DepthEstimator:
    def __init__(self, model="DPT_Large", device="auto"):
        self.model = torch.hub.load("intel-isl/MiDaS", model)
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    def estimate_depth(self, image):
        # Returns normalized depth map
        pass
```

**Video Generator**
```python
class VideoGenerator:
    def __init__(self, config):
        self.fps = config.fps
        self.resolution = config.target_resolution
    
    def create_scanning_video(self, image, depth_map):
        # Generates animated scanning video
        pass
```

**Effects Pipeline**
```python
class EffectsPipeline:
    def apply_scanning_effect(self, frame, scan_position, depth_contours):
        # Applies visual effects for current scan position
        pass
```

### Processing Pipeline

1. **Image Loading**: Load and validate input images
2. **Preprocessing**: Resize, pad, and normalize for model input
3. **Depth Estimation**: Generate depth maps using MiDaS
4. **Contour Generation**: Extract depth-based contour lines
5. **Animation Generation**: Create frame-by-frame scanning animation
6. **Effects Application**: Apply visual effects and enhancements
7. **Video Encoding**: Combine frames into final video output

## Performance Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| **Input Resolution** | Any | Automatically scaled to 4K |
| **Output Resolution** | 3840x2160 | 4K Ultra HD |
| **Frame Rate** | 24-60 FPS | Configurable |
| **Processing Speed** | 2-10 sec/frame | Depends on GPU |
| **Memory Usage** | 4-8GB VRAM | For DPT_Large model |
| **File Formats** | JPG, PNG, BMP, TIFF | Input images |
| **Video Codecs** | H.264, H.265, AV1 | Output formats |

## Creative Applications

### Artistic Installations
- **Gallery Exhibitions**: Transform static artwork into dynamic displays
- **Digital Art**: Create mesmerizing depth-based animations
- **Interactive Displays**: Combine with motion sensors for responsive art
- **Projection Mapping**: Use depth data for 3D surface projection

### Commercial Applications
- **Product Visualization**: Highlight product depth and form
- **Architectural Presentation**: Showcase building depth and structure
- **Marketing Content**: Eye-catching social media and advertising videos
- **Documentary Production**: Enhance historical photographs with depth animation

### Educational Uses
- **Computer Vision Teaching**: Demonstrate depth estimation concepts
- **Art Education**: Explore relationships between 2D and 3D perception
- **Technology Demonstrations**: Show AI capabilities in visual processing
- **Research Applications**: Analyze depth perception and visual cognition

## Advanced Usage

### Custom Effect Development
```python
from src.effects_pipeline import EffectsPipeline

class CustomEffects(EffectsPipeline):
    def apply_custom_scan_effect(self, frame, scan_pos, depth_map):
        # Implement your own scanning effects
        enhanced_frame = self.apply_color_mapping(frame, depth_map)
        return self.add_particle_effects(enhanced_frame, scan_pos)
```

### Batch Processing
```python
from src.depth_scanner import DepthScanner

scanner = DepthScanner(config="config/batch.json")
scanner.process_directory("input/photos", "output/videos")
```

### Real-Time Processing
```python
# For webcam or video input
scanner.process_video_stream(source=0, output="live_scan.mp4")
```

## Dependencies

### Core Libraries
- **PyTorch 1.9+**: Deep learning framework for MiDaS model
- **OpenCV 4.5+**: Computer vision and video processing
- **NumPy**: Numerical operations and array processing
- **Pillow**: Image format support and manipulation

### AI/ML Libraries
- **torch-hub**: Pre-trained model loading
- **torchvision**: Image transformations and utilities
- **timm**: Vision transformer models support

### Optional Libraries
- **CUDA Toolkit**: GPU acceleration (recommended)
- **FFmpeg**: Advanced video encoding options
- **scikit-image**: Additional image processing tools
- **matplotlib**: Visualization and debugging tools

## Installation Options

### Standard Installation
```bash
pip install torch torchvision opencv-python numpy pillow
```

### Development Installation
```bash
git clone https://github.com/YOUR_USERNAME/midas-depth-scanner.git
cd midas-depth-scanner
pip install -e .
```

### Docker Installation
```bash
docker build -t depth-scanner .
docker run --gpus all -v /path/to/images:/input -v /path/to/output:/output depth-scanner
```


##  Contact

- **GitHub**: [https://github.com/CJD-11
- **Email**: coreydziadzio@c11visualarts.com
- **Project Link**: https://github.com/CJD-11/Mind-Topgraphic

## Project Status

- ✅ **Core Functionality**: Complete depth scanning system
- ✅ **4K Video Output**: Professional quality rendering
- ✅ **GPU Acceleration**: Optimized performance
- ✅ **Batch Processing**: Multiple image support
- 🔄 **Real-Time Processing**: In development
- 📋 **Web Interface**: Future development
- 📋 **Mobile Support**: Research phase
