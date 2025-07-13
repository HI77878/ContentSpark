from analyzers.base_analyzer import GPUBatchAnalyzer
import cv2
import numpy as np
from typing import Dict, Any, List
import torch

# GPU Forcing
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

class CutAnalysisFixedAnalyzer(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__()
        self.analyzer_name = "cut_analysis"
        self.threshold = 30.0
        self.sample_rate = 30  # Für 1:1 Rekonstruktion
        self.jump_cut_threshold = 15.0  # Lower threshold for jump cuts
        self.last_jump_cut_time = 0
        
    def _load_model_impl(self):
        """No model to load for cut analysis"""
        pass
    
    def analyze_cut_characteristics(self, diff: float, frame1: torch.Tensor, frame2: torch.Tensor, timestamp: float = 0) -> Dict[str, Any]:
        """Analyze the characteristics of a cut"""
        # Calculate additional metrics
        color_diff = torch.abs(frame1.mean(dim=(0,1)) - frame2.mean(dim=(0,1))).max().item()
        luminance_diff = torch.abs(frame1.mean() - frame2.mean()).item()
        
        # Edge detection for motion blur
        edge1 = torch.abs(frame1[1:] - frame1[:-1]).mean() + torch.abs(frame1[:,1:] - frame1[:,:-1]).mean()
        edge2 = torch.abs(frame2[1:] - frame2[:-1]).mean() + torch.abs(frame2[:,1:] - frame2[:,:-1]).mean()
        edge_diff = torch.abs(edge1 - edge2).item()
        
        # Histogram comparison for jump cut detection
        hist1_r = torch.histc(frame1[:,:,0], bins=32, min=0, max=1)
        hist1_g = torch.histc(frame1[:,:,1], bins=32, min=0, max=1)
        hist1_b = torch.histc(frame1[:,:,2], bins=32, min=0, max=1)
        
        hist2_r = torch.histc(frame2[:,:,0], bins=32, min=0, max=1)
        hist2_g = torch.histc(frame2[:,:,1], bins=32, min=0, max=1)
        hist2_b = torch.histc(frame2[:,:,2], bins=32, min=0, max=1)
        
        # Histogram correlation
        hist_corr_r = torch.corrcoef(torch.stack([hist1_r, hist2_r]))[0,1].item()
        hist_corr_g = torch.corrcoef(torch.stack([hist1_g, hist2_g]))[0,1].item()
        hist_corr_b = torch.corrcoef(torch.stack([hist1_b, hist2_b]))[0,1].item()
        hist_corr = (hist_corr_r + hist_corr_g + hist_corr_b) / 3
        
        # Check for jump cut (same scene, different framing)
        is_jump_cut = (self.jump_cut_threshold < diff < 30 and 
                      hist_corr > 0.7 and 
                      color_diff < 20 and
                      timestamp - self.last_jump_cut_time > 0.5)
        
        # Determine cut type
        if is_jump_cut:
            cut_type = "jump_cut"
            description = "Jump cut within same scene"
            self.last_jump_cut_time = timestamp
        elif diff > 80:
            cut_type = "smash_cut"
            description = "Abrupt transition with high visual contrast"
        elif diff > 50:
            cut_type = "hard_cut"
            description = "Standard hard cut between shots"
        elif diff > 30:
            cut_type = "match_cut"
            description = "Smooth transition with visual continuity"
        elif luminance_diff > 40:
            cut_type = "fade_transition"
            description = "Luminance-based transition"
        elif color_diff > 50:
            cut_type = "color_shift"
            description = "Color-based transition"
        elif edge_diff > 20:
            cut_type = "motion_cut"
            description = "Motion-based transition"
        else:
            cut_type = "soft_cut"
            description = "Subtle transition"
        
        return {
            "type": cut_type,
            "description": description,
            "intensity": diff,
            "color_change": color_diff,
            "luminance_change": luminance_diff,
            "motion_change": edge_diff
        }
        
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        cuts = []
        all_segments = []
        
        # Convert to tensor for GPU processing
        frame_tensors = []
        for frame in frames:
            tensor = torch.from_numpy(frame).float().cuda()
            frame_tensors.append(tensor)
        
        # Analyze every frame transition
        current_shot_start = 0
        shot_count = 0
        
        for i in range(1, len(frame_tensors)):
            # GPU-based difference calculation
            diff = torch.abs(frame_tensors[i] - frame_tensors[i-1])
            mean_diff = diff.mean().item()
            
            # Always create a segment for tracking
            segment = {
                "timestamp": frame_times[i],
                "start_time": frame_times[i-1],
                "end_time": frame_times[i],
                "frame_diff": mean_diff,
                "is_cut": mean_diff > self.threshold
            }
            
            # Also check for jump cuts with lower threshold
            if mean_diff > self.threshold or (mean_diff > self.jump_cut_threshold and i > 1):
                # Analyze cut characteristics
                cut_info = self.analyze_cut_characteristics(
                    mean_diff, 
                    frame_tensors[i-1], 
                    frame_tensors[i],
                    frame_times[i]
                )
                
                segment.update({
                    "type": cut_info["type"],
                    "description": cut_info["description"],
                    "confidence": min(mean_diff / 50.0, 1.0),
                    "intensity": cut_info["intensity"],
                    "color_change": cut_info["color_change"],
                    "luminance_change": cut_info["luminance_change"],
                    "motion_change": cut_info["motion_change"],
                    "shot_duration": frame_times[i] - frame_times[current_shot_start],
                    "shot_number": shot_count
                })
                
                cuts.append(segment)
                current_shot_start = i
                shot_count += 1
            else:
                segment.update({
                    "type": "no_cut",
                    "description": "Continuous shot",
                    "confidence": 0.0
                })
            
            all_segments.append(segment)
        
        # Calculate shot rhythm and pacing
        shot_durations = []
        if cuts:
            shot_durations = [cuts[i]["shot_duration"] for i in range(len(cuts)) if "shot_duration" in cuts[i]]
        
        avg_shot_duration = np.mean(shot_durations) if shot_durations else 0
        
        # Classify editing style
        if avg_shot_duration < 2:
            editing_style = "rapid_montage"
        elif avg_shot_duration < 4:
            editing_style = "fast_paced"
        elif avg_shot_duration < 8:
            editing_style = "moderate_pacing"
        else:
            editing_style = "slow_paced"
        
        # Cut type distribution
        cut_types = {}
        for cut in cuts:
            ct = cut.get("type", "unknown")
            cut_types[ct] = cut_types.get(ct, 0) + 1
        
        return {
            "segments": all_segments,
            "cuts": cuts,
            "total_cuts": len(cuts),
            "total_shots": shot_count + 1,
            "average_shot_duration": avg_shot_duration,
            "editing_style": editing_style,
            "cut_type_distribution": cut_types,
            "cuts_per_minute": (len(cuts) / (frame_times[-1] / 60)) if frame_times else 0
        }
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main entry point"""
        print(f"[CutAnalysis-Fixed] Analyzing {video_path}")
        frames, frame_times = self.extract_frames(video_path)
        return self.process_batch_gpu(frames, frame_times)