from analyzers.base_analyzer import GPUBatchAnalyzer
import cv2
import numpy as np
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

# GPU Forcing
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

class SceneSegmentationFixedAnalyzer(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__()
        self.analyzer_name = "scene_segmentation"
        self.sample_rate = 10  # Erhöht von 30 auf 10 (3 FPS statt 1 FPS)
        
        # Adaptive thresholds
        self.hist_threshold = 0.7  # Histogram similarity threshold
        self.ssim_threshold = 0.85  # SSIM threshold
        self.pixel_threshold = 30.0  # Pixel difference threshold
        
    def _load_model_impl(self):
        """No model to load for scene segmentation"""
        pass
    
    def calculate_histogram_difference(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """Calculate histogram difference between two frames"""
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = 0.299 * frame1[:,:,0] + 0.587 * frame1[:,:,1] + 0.114 * frame1[:,:,2]
            gray2 = 0.299 * frame2[:,:,0] + 0.587 * frame2[:,:,1] + 0.114 * frame2[:,:,2]
        else:
            gray1, gray2 = frame1, frame2
        
        # Calculate histograms
        hist1 = torch.histc(gray1, bins=256, min=0, max=255)
        hist2 = torch.histc(gray2, bins=256, min=0, max=255)
        
        # Normalize histograms
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Calculate correlation coefficient
        correlation = torch.sum((hist1 - hist1.mean()) * (hist2 - hist2.mean())) / (
            torch.sqrt(torch.sum((hist1 - hist1.mean())**2)) * 
            torch.sqrt(torch.sum((hist2 - hist2.mean())**2))
        )
        
        return correlation.item()
    
    def calculate_edge_difference(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """Calculate edge-based difference using Sobel filters"""
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = 0.299 * frame1[:,:,0] + 0.587 * frame1[:,:,1] + 0.114 * frame1[:,:,2]
            gray2 = 0.299 * frame2[:,:,0] + 0.587 * frame2[:,:,1] + 0.114 * frame2[:,:,2]
        else:
            gray1, gray2 = frame1, frame2
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).cuda()
        
        # Reshape for convolution
        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
        
        # Apply Sobel filters
        edges1_x = F.conv2d(gray1, sobel_x, padding=1)
        edges1_y = F.conv2d(gray1, sobel_y, padding=1)
        edges2_x = F.conv2d(gray2, sobel_x, padding=1)
        edges2_y = F.conv2d(gray2, sobel_y, padding=1)
        
        # Calculate edge magnitude
        edges1 = torch.sqrt(edges1_x**2 + edges1_y**2)
        edges2 = torch.sqrt(edges2_x**2 + edges2_y**2)
        
        # Calculate difference
        edge_diff = torch.abs(edges1 - edges2).mean()
        
        return edge_diff.item()
    
    def classify_scene_type(self, frames: List[torch.Tensor], start_idx: int, end_idx: int) -> str:
        """Classify the type of scene based on visual content"""
        if end_idx <= start_idx or start_idx >= len(frames):
            return "unknown"
        
        # Sample frames from the scene
        sample_indices = np.linspace(start_idx, min(end_idx, len(frames)-1), min(5, end_idx-start_idx+1), dtype=int)
        
        # Analyze visual characteristics
        avg_brightness = []
        color_variance = []
        edge_density = []
        
        for idx in sample_indices:
            frame = frames[idx]
            
            # Brightness
            if len(frame.shape) == 3:
                gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
            else:
                gray = frame
            avg_brightness.append(gray.mean().item())
            
            # Color variance
            if len(frame.shape) == 3:
                color_var = frame.std(dim=2).mean().item()
                color_variance.append(color_var)
            
            # Edge density (simplified)
            if idx > 0:
                edge_diff = torch.abs(frame - frames[idx-1]).mean().item()
                edge_density.append(edge_diff)
        
        # Classification based on characteristics
        avg_bright = np.mean(avg_brightness)
        avg_color_var = np.mean(color_variance) if color_variance else 0
        avg_edge = np.mean(edge_density) if edge_density else 0
        
        # Determine scene type
        if avg_bright < 50:
            return "dark_scene"
        elif avg_bright > 200:
            return "bright_scene"
        elif avg_color_var < 20:
            return "monochrome_scene"
        elif avg_edge > 50:
            return "action_scene"
        elif avg_edge < 10:
            return "static_scene"
        elif 100 < avg_bright < 150 and avg_color_var > 30:
            return "indoor_scene"
        elif avg_bright > 150 and avg_color_var > 40:
            return "outdoor_scene"
        else:
            return "general_scene"
    
    def detect_transition_type(self, frames: List[torch.Tensor], start_idx: int, end_idx: int) -> str:
        """Detect the type of transition between scenes"""
        if end_idx - start_idx <= 1:
            return "cut"
        
        # Analyze luminance changes
        luminances = []
        for i in range(start_idx, min(end_idx + 1, len(frames))):
            if len(frames[i].shape) == 3:
                lum = 0.299 * frames[i][:,:,0] + 0.587 * frames[i][:,:,1] + 0.114 * frames[i][:,:,2]
            else:
                lum = frames[i]
            luminances.append(lum.mean().item())
        
        # Check for fade patterns
        if len(luminances) >= 3:
            # Fade to black
            if luminances[0] > luminances[-1] * 2 and min(luminances) < 20:
                return "fade_out"
            # Fade from black
            elif luminances[-1] > luminances[0] * 2 and min(luminances) < 20:
                return "fade_in"
            # Cross dissolve
            elif abs(luminances[0] - luminances[-1]) < 50:
                return "dissolve"
        
        return "cut"
        
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        scenes = []
        current_scene_start = 0
        scene_confidences = []
        
        # Convert to tensor for GPU processing
        frame_tensors = []
        for frame in frames:
            tensor = torch.from_numpy(frame).float().cuda()
            frame_tensors.append(tensor)
        
        # Analyze every frame pair for better detection
        for i in range(1, len(frame_tensors)):
            # Multi-metric scene change detection
            
            # 1. Histogram difference
            hist_corr = self.calculate_histogram_difference(frame_tensors[i-1], frame_tensors[i])
            
            # 2. Pixel difference
            pixel_diff = torch.abs(frame_tensors[i] - frame_tensors[i-1]).mean().item()
            
            # 3. Edge difference
            edge_diff = self.calculate_edge_difference(frame_tensors[i-1], frame_tensors[i])
            
            # 4. SSIM (for CPU - convert small patches)
            patch1 = frame_tensors[i-1][:100, :100].cpu().numpy()
            patch2 = frame_tensors[i][:100, :100].cpu().numpy()
            
            # Ensure data is float32 and in correct range
            patch1 = patch1.astype(np.float32)
            patch2 = patch2.astype(np.float32)
            
            # Check if data needs normalization (if max > 1, it's in 0-255 range)
            if patch1.max() > 1.0:
                patch1 = patch1 / 255.0
                patch2 = patch2 / 255.0
            
            # Clip to ensure valid range
            patch1 = np.clip(patch1, 0, 1)
            patch2 = np.clip(patch2, 0, 1)
            
            # Calculate SSIM with proper parameters
            try:
                if len(patch1.shape) == 3:
                    ssim_score = ssim(patch1, patch2, channel_axis=2, data_range=1.0)
                else:
                    ssim_score = ssim(patch1, patch2, data_range=1.0)
            except Exception as e:
                # Fallback to simple correlation if SSIM fails
                ssim_score = np.corrcoef(patch1.flatten(), patch2.flatten())[0, 1]
            
            # Adaptive threshold based on multiple metrics
            is_scene_change = False
            confidence = 0.0
            
            if hist_corr < self.hist_threshold:
                confidence += 0.3
            if pixel_diff > self.pixel_threshold:
                confidence += 0.3
            if edge_diff > 50:
                confidence += 0.2
            if ssim_score < self.ssim_threshold:
                confidence += 0.2
            
            if confidence >= 0.5:  # Scene change detected
                is_scene_change = True
                
                # Detect transition type
                transition_type = self.detect_transition_type(frame_tensors, current_scene_start, i)
                
                scene_type = self.classify_scene_type(frame_tensors, current_scene_start, i)
                
                scenes.append({
                    "start_time": frame_times[current_scene_start],
                    "end_time": frame_times[i],
                    "duration": frame_times[i] - frame_times[current_scene_start],
                    "scene_id": len(scenes),
                    "scene_type": scene_type,
                    "transition_type": transition_type,
                    "confidence": confidence,
                    "description": f"Scene {len(scenes)+1}: {scene_type.replace('_', ' ').title()} with {transition_type} transition",
                    "scene_description": f"{scene_type.replace('_', ' ').title()} lasting {frame_times[i] - frame_times[current_scene_start]:.1f}s",
                    "metrics": {
                        "histogram_correlation": hist_corr,
                        "pixel_difference": pixel_diff,
                        "edge_difference": edge_diff,
                        "ssim_score": ssim_score
                    }
                })
                scene_confidences.append(confidence)
                current_scene_start = i
        
        # Add last scene
        if current_scene_start < len(frames) - 1:
            scene_type = self.classify_scene_type(frame_tensors, current_scene_start, len(frame_tensors)-1)
            
            scenes.append({
                "start_time": frame_times[current_scene_start],
                "end_time": frame_times[-1],
                "duration": frame_times[-1] - frame_times[current_scene_start],
                "scene_id": len(scenes),
                "scene_type": scene_type,
                "transition_type": "end",
                "confidence": 1.0,
                "description": f"Scene {len(scenes)+1}: {scene_type.replace('_', ' ').title()} (final scene)",
                "scene_description": f"{scene_type.replace('_', ' ').title()} lasting {frame_times[-1] - frame_times[current_scene_start]:.1f}s"
            })
        
        # Calculate average scene duration and shot types
        avg_duration = np.mean([s["duration"] for s in scenes]) if scenes else 0
        
        # Classify shot lengths
        for scene in scenes:
            if scene["duration"] < 1.0:
                scene["shot_type"] = "quick_cut"
            elif scene["duration"] < 3.0:
                scene["shot_type"] = "short"
            elif scene["duration"] < 7.0:
                scene["shot_type"] = "medium"
            else:
                scene["shot_type"] = "long"
        
        return {
            "segments": scenes,
            "scenes": scenes,
            "total_scenes": len(scenes),
            "average_scene_duration": avg_duration,
            "average_confidence": np.mean(scene_confidences) if scene_confidences else 0,
            "transition_types": list(set([s.get("transition_type", "cut") for s in scenes])),
            "shot_distribution": {
                "quick_cuts": sum(1 for s in scenes if s.get("shot_type") == "quick_cut"),
                "short": sum(1 for s in scenes if s.get("shot_type") == "short"),
                "medium": sum(1 for s in scenes if s.get("shot_type") == "medium"),
                "long": sum(1 for s in scenes if s.get("shot_type") == "long")
            }
        }
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main entry point"""
        print(f"[SceneSegmentation-Fixed] Analyzing {video_path}")
        frames, frame_times = self.extract_frames(video_path)
        print(f"[SceneSegmentation-Fixed] Extracted {len(frames)} frames for analysis")
        return self.process_batch_gpu(frames, frame_times)