"""
MiDaS Depth Scanner "Mind Topographic" - Main Application

AI-powered artistic video generation using depth-based scanning effects.
Transforms static images into dynamic 4K videos with progressive depth visualization.

Author: Corey Dziadzio
Email: coreydziadzio@c11visualarts.com
GitHub: https://github.com/CJD-11/Mind-Topgraphic

Features:
- MiDaS DPT_Large depth estimation
- 4K video output with customizable effects
- GPU acceleration for high performance
- Batch processing capabilities
- Configurable scanning parameters

"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import os
import torch
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Import custom modules
from depth_estimator import DepthEstimator
from video_generator import VideoGenerator
from image_processor import ImageProcessor
from effects_pipeline import EffectsPipeline
from config_manager import ConfigManager
from utils.file_utils import FileManager
from utils.video_utils import VideoEncoder
from utils.gpu_utils import GPUManager


class DepthScanner:
    """
    Main application class for depth-based scanning video generation.
    
    Coordinates depth estimation, effects processing, and video creation
    to transform static images into dynamic artistic animations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the depth scanner.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize GPU management
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_optimal_device()
        
        # Initialize core components
        self.depth_estimator = DepthEstimator(
            model_name=self.config.depth_model,
            device=self.device
        )
        
        self.image_processor = ImageProcessor(self.config)
        self.effects_pipeline = EffectsPipeline(self.config)
        self.video_generator = VideoGenerator(self.config)
        self.file_manager = FileManager()
        
        # Performance tracking
        self.processing_stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0,
            'depth_estimation_time': 0.0,
            'video_generation_time': 0.0
        }
        
        print("🧠 MiDaS Depth Scanner initialized")
        print(f"📱 Device: {self.device}")
        print(f"🎬 Output resolution: {self.config.target_width}x{self.config.target_height}")
        print(f"🔄 Scan loops: {self.config.scan_loops}")
        print(f"⚡ FPS: {self.config.fps}")
        print()
    
    def process_single_image(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """
        Process a single image to create scanning video.
        
        Args:
            image_path: Path to input image
            output_path: Optional custom output path
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            start_time = time.time()
            
            print(f"🖼️ Processing: {os.path.basename(image_path)}")
            
            # Load and validate image
            image = self.image_processor.load_image(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return False
            
            # Preprocess image (resize, pad to target resolution)
            processed_image = self.image_processor.preprocess_for_scanning(image)
            
            # Generate depth map
            depth_start = time.time()
            depth_map = self.depth_estimator.estimate_depth(processed_image)
            depth_time = time.time() - depth_start
            
            print(f"🏔️ Depth estimation completed in {depth_time:.2f}s")
            
            # Generate contours from depth map
            contours = self.effects_pipeline.generate_depth_contours(
                depth_map, 
                levels=self.config.contour_levels
            )
            
            print(f"📐 Generated {len(contours)} depth contours")
            
            # Create output path if not provided
            if output_path is None:
                output_path = self._generate_output_path(image_path)
            
            # Generate scanning video
            video_start = time.time()
            success = self.video_generator.create_scanning_video(
                processed_image,
                depth_map,
                contours,
                output_path
            )
            video_time = time.time() - video_start
            
            if success:
                total_time = time.time() - start_time
                print(f"✅ Video created: {output_path}")
                print(f"⏱️ Total processing time: {total_time:.2f}s")
                print(f"   - Depth estimation: {depth_time:.2f}s")
                print(f"   - Video generation: {video_time:.2f}s")
                
                # Update statistics
                self._update_processing_stats(total_time, depth_time, video_time)
                
                return True
            else:
                print(f"❌ Failed to create video for {image_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Optional output directory
            
        Returns:
            List of successfully created video paths
        """
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            return []
        
        # Find all image files
        image_files = self.file_manager.find_image_files(input_dir)
        
        if not image_files:
            print(f"⚠️ No image files found in {input_dir}")
            return []
        
        print(f"📁 Found {len(image_files)} image(s) in {input_dir}")
        print(f"🎯 Images: {[os.path.basename(f) for f in image_files[:5]]}")
        if len(image_files) > 5:
            print(f"    ... and {len(image_files) - 5} more")
        print()
        
        # Set up output directory
        if output_dir is None:
            output_dir = self.config.output_folder
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        successful_videos = []
        failed_count = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"📊 Progress: {i}/{len(image_files)}")
            
            # Generate output path
            output_path = os.path.join(
                output_dir,
                self._generate_output_filename(image_path)
            )
            
            # Process image
            if self.process_single_image(image_path, output_path):
                successful_videos.append(output_path)
            else:
                failed_count += 1
            
            print()  # Add spacing between images
        
        # Print summary
        print("📈 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"✅ Successful: {len(successful_videos)}")
        print(f"❌ Failed: {failed_count}")
        print(f"📊 Success rate: {len(successful_videos)/len(image_files)*100:.1f}%")
        
        if self.processing_stats['images_processed'] > 0:
            print(f"⏱️ Average time per image: {self.processing_stats['average_time_per_image']:.2f}s")
        
        print("=" * 50)
        
        return successful_videos
    
    def process_video_sequence(self, input_dir: str, output_path: str) -> bool:
        """
        Process multiple images into a single video sequence.
        
        Args:
            input_dir: Directory containing input images
            output_path: Path for output video
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image_files = self.file_manager.find_image_files(input_dir)
            
            if not image_files:
                print(f"❌ No images found in {input_dir}")
                return False
            
            print(f"🎬 Creating video sequence from {len(image_files)} images")
            
            # Sort files for consistent ordering
            image_files.sort()
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                self.config.fps, 
                (self.config.target_width, self.config.target_height)
            )
            
            # Process each image and add to video
            for i, image_path in enumerate(image_files):
                print(f"📊 Processing frame {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Load and process image
                image = self.image_processor.load_image(image_path)
                if image is None:
                    continue
                
                processed_image = self.image_processor.preprocess_for_scanning(image)
                depth_map = self.depth_estimator.estimate_depth(processed_image)
                contours = self.effects_pipeline.generate_depth_contours(depth_map, self.config.contour_levels)
                
                # Generate scanning frames for this image
                frames = self.video_generator.generate_scanning_frames(
                    processed_image, 
                    depth_map, 
                    contours
                )
                
                # Add frames to video
                for frame in frames:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"✅ Video sequence saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating video sequence: {str(e)}")
            return False
    
    def _generate_output_path(self, input_path: str) -> str:
        """Generate output path for processed video."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        filename = f"{base_name}_DepthScan_4K_{timestamp}.mp4"
        return os.path.join(self.config.output_folder, filename)
    
    def _generate_output_filename(self, input_path: str) -> str:
        """Generate output filename for processed video."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        return f"{base_name}_DepthScan_4K_{timestamp}.mp4"
    
    def _update_processing_stats(self, total_time: float, depth_time: float, video_time: float):
        """Update processing statistics."""
        self.processing_stats['images_processed'] += 1
        self.processing_stats['total_processing_time'] += total_time
        self.processing_stats['depth_estimation_time'] += depth_time
        self.processing_stats['video_generation_time'] += video_time
        
        # Calculate average
        count = self.processing_stats['images_processed']
        self.processing_stats['average_time_per_image'] = self.processing_stats['total_processing_time'] / count
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def save_depth_map(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Save depth map visualization for an image.
        
        Args:
            image_path: Path to input image
            output_dir: Optional output directory
            
        Returns:
            Path to saved depth map or None if failed
        """
        try:
            image = self.image_processor.load_image(image_path)
            if image is None:
                return None
            
            processed_image = self.image_processor.preprocess_for_scanning(image)
            depth_map = self.depth_estimator.estimate_depth(processed_image)
            
            # Convert to visualization
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Generate output path
            if output_dir is None:
                output_dir = os.path.join(self.config.output_folder, "depth_maps")
            
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_depth_map.jpg")
            
            cv2.imwrite(output_path, depth_vis)
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving depth map: {str(e)}")
            return None
    
    def preview_effects(self, image_path: str) -> bool:
        """
        Preview scanning effects without creating full video.
        
        Args:
            image_path: Path to input image
            
        Returns:
            bool: True if preview successful
        """
        try:
            print(f"👁️ Previewing effects for: {os.path.basename(image_path)}")
            
            # Load and process image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return False
            
            processed_image = self.image_processor.preprocess_for_scanning(image)
            depth_map = self.depth_estimator.estimate_depth(processed_image)
            contours = self.effects_pipeline.generate_depth_contours(depth_map, self.config.contour_levels)
            
            # Generate preview frames at different scan positions
            preview_positions = [0, 0.25, 0.5, 0.75, 1.0]
            
            for i, pos in enumerate(preview_positions):
                scan_y = int(pos * self.config.target_height)
                
                # Generate frame at this scan position
                frame = self.effects_pipeline.apply_scanning_effects(
                    processed_image.astype(np.float32),
                    scan_y,
                    contours,
                    frame_number=i
                )
                
                # Display preview
                preview_frame = cv2.resize(frame.astype(np.uint8), (960, 540))  # Scale for display
                cv2.imshow(f'Preview - Position {i+1}/5', preview_frame)
                cv2.waitKey(1000)  # Show for 1 second
            
            cv2.destroyAllWindows()
            print("✅ Preview completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in preview: {str(e)}")
            return False
