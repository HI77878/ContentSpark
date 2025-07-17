import logging
import re
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import os
import torch

# GPU Forcing
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# FFmpeg pthread fix
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

class GPUBatchCommentCTADetection(GPUBatchAnalyzer):
    """
    Detects references to comments and Call-to-Action elements in videos
    """
    
    def __init__(self):
        super().__init__()
        self.device_type = "cpu"  # Text analysis on CPU
        
        # Comment reference patterns
        self.comment_patterns = [
            r'kommentar',
            r'comments?',
            r'schreibt?\s*(mir|es|eure)',
            r'lasst?\s*(mir|uns)',
            r'sagt?\s*mir',
            r'teilt?\s*mir',
            r'was\s*denkt\s*ihr',
            r'eure\s*meinung',
            r'feedback',
            r'drop.*comment',
            r'let\s*me\s*know',
            r'tell\s*me',
            r'what\s*do\s*you\s*think'
        ]
        
        # CTA patterns
        self.cta_patterns = [
            r'folg(t|en)',
            r'follow',
            r'abonnier',
            r'subscribe',
            r'lik(e|t|en)',
            r'herz',
            r'heart',
            r'teilt?',
            r'share',
            r'klick',
            r'click',
            r'link\s*in\s*(der\s*)?bio',
            r'swipe\s*up',
            r'mehr\s*(dazu|infos|videos)',
            r'check\s*out',
            r'speichert?',
            r'save',
            r'merkt?\s*euch',
            r'remember'
        ]
        
        # Reply/response patterns
        self.reply_patterns = [
            r'antwort\s*auf',
            r'reply\s*to',
            r'reaktion\s*auf',
            r'response\s*to',
            r'ihr\s*habt\s*gefragt',
            r'you\s*asked',
            r'viele\s*haben\s*geschrieben',
            r'many\s*of\s*you'
        ]
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for comment references and CTAs"""
        logger.info(f"[CommentCTA] Analyzing {video_path}")
        
        try:
            # Get transcription data
            transcription_path = video_path.replace('.mp4', '_transcription.json')
            transcription_data = None
            
            if os.path.exists(transcription_path):
                import json
                with open(transcription_path, 'r') as f:
                    transcription_data = json.load(f)
            
            # Get text overlay data
            text_overlay_path = video_path.replace('.mp4', '_text_overlay.json')
            text_overlay_data = None
            
            if os.path.exists(text_overlay_path):
                import json
                with open(text_overlay_path, 'r') as f:
                    text_overlay_data = json.load(f)
            
            segments = []
            
            # Analyze transcription
            if transcription_data and 'segments' in transcription_data:
                for seg in transcription_data['segments']:
                    result = self._analyze_text_segment(seg, 'speech')
                    if result['has_reference']:
                        segments.append(result)
            
            # Analyze text overlays
            if text_overlay_data and 'segments' in text_overlay_data:
                for seg in text_overlay_data['segments']:
                    result = self._analyze_text_segment(seg, 'overlay')
                    if result['has_reference']:
                        segments.append(result)
            
            # Generate summary
            summary = self._generate_summary(segments)
            
            return {
                'segments': segments,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Comment/CTA detection failed: {e}")
            return {'segments': [], 'error': str(e)}
    
    def _analyze_text_segment(self, segment, source_type):
        """Analyze a text segment for references"""
        text = segment.get('text', '').lower()
        if not text:
            return {'has_reference': False}
        
        timestamp = float(segment.get('timestamp', segment.get('start', segment.get('start_time', 0))))
        
        # Check for comment references
        comment_refs = []
        for pattern in self.comment_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                comment_refs.append({
                    'pattern': pattern,
                    'type': 'comment_request'
                })
        
        # Check for CTAs
        ctas = []
        for pattern in self.cta_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                cta_type = self._classify_cta(pattern)
                ctas.append({
                    'pattern': pattern,
                    'type': cta_type
                })
        
        # Check for reply indicators
        replies = []
        for pattern in self.reply_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                replies.append({
                    'pattern': pattern,
                    'type': 'reply_indicator'
                })
        
        has_reference = bool(comment_refs or ctas or replies)
        
        return {
            'timestamp': timestamp,
            'start_time': timestamp,
            'end_time': timestamp + 2.0,
            'source': source_type,
            'text': segment.get('text', ''),
            'has_reference': has_reference,
            'comment_references': comment_refs,
            'ctas': ctas,
            'reply_indicators': replies,
            'interaction_type': self._determine_interaction_type(comment_refs, ctas, replies)
        }
    
    def _classify_cta(self, pattern):
        """Classify CTA type based on pattern"""
        if 'folg' in pattern or 'follow' in pattern or 'abonn' in pattern:
            return 'follow_request'
        elif 'lik' in pattern or 'herz' in pattern or 'heart' in pattern:
            return 'like_request'
        elif 'teil' in pattern or 'share' in pattern:
            return 'share_request'
        elif 'bio' in pattern or 'link' in pattern:
            return 'link_promotion'
        elif 'speicher' in pattern or 'save' in pattern or 'merk' in pattern:
            return 'save_request'
        else:
            return 'general_cta'
    
    def _determine_interaction_type(self, comments, ctas, replies):
        """Determine primary interaction type"""
        if replies:
            return 'community_response'
        elif comments and ctas:
            return 'engagement_request'
        elif comments:
            return 'comment_solicitation'
        elif ctas:
            return 'action_request'
        else:
            return 'none'
    
    def _generate_summary(self, segments):
        """Generate summary of interactions"""
        if not segments:
            return {
                'total_references': 0,
                'has_comment_requests': False,
                'has_ctas': False,
                'has_reply_indicators': False
            }
        
        # Count types
        comment_count = 0
        cta_count = 0
        reply_count = 0
        
        cta_types = {}
        interaction_types = {}
        
        for seg in segments:
            comment_count += len(seg.get('comment_references', []))
            cta_count += len(seg.get('ctas', []))
            reply_count += len(seg.get('reply_indicators', []))
            
            # Track CTA types
            for cta in seg.get('ctas', []):
                cta_type = cta['type']
                cta_types[cta_type] = cta_types.get(cta_type, 0) + 1
            
            # Track interaction types
            int_type = seg.get('interaction_type', 'none')
            if int_type != 'none':
                interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
        
        return {
            'total_references': len(segments),
            'has_comment_requests': comment_count > 0,
            'has_ctas': cta_count > 0,
            'has_reply_indicators': reply_count > 0,
            'comment_request_count': comment_count,
            'cta_count': cta_count,
            'reply_indicator_count': reply_count,
            'cta_types': cta_types,
            'interaction_types': interaction_types,
            'engagement_level': self._calculate_engagement_level(comment_count, cta_count, reply_count)
        }
    
    def _calculate_engagement_level(self, comments, ctas, replies):
        """Calculate overall engagement level"""
        total = comments + ctas + replies
        
        if total == 0:
            return 'none'
        elif total == 1:
            return 'minimal'
        elif total <= 3:
            return 'moderate'
        elif total <= 6:
            return 'high'
        else:
            return 'very_high'
    
    def process_batch_gpu(self, frames, frame_times):
        """Required by base class but not used"""
        return {'segments': []}