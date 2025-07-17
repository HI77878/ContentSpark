#!/usr/bin/env python3
"""Create human-readable video reconstruction"""
import json

# Load reconstruction data
with open('/home/user/tiktok_production/reconstruction_data.json') as f:
    data = json.load(f)

video_duration = data['video_duration']
timeline = data['timeline']

output_text = f"""VIDEO RECONSTRUCTION: Leon Schliebach TikTok
Duration: {video_duration:.1f} seconds
Source: Analysis data only (video not viewed)

SCENE-BY-SCENE BREAKDOWN:
========================
"""

# Sort timeline by second
for second in sorted(map(int, timeline.keys())):
    second_data = timeline[str(second)]
    
    output_text += f"\n[{second:02d}s] "
    
    # Build description from available data
    description_parts = []
    
    # Scene (if available from streaming dense captioning)
    if 'scene' in second_data:
        description_parts.append(second_data['scene'])
    
    # Text overlays (very prominent in TikTok)
    if 'text' in second_data:
        text_shown = ', '.join(second_data['text'])
        if not description_parts:  # If no scene description
            description_parts.append(f"Text on screen: \"{text_shown}\"")
        else:
            description_parts.append(f"Text overlay: \"{text_shown}\"")
    
    # Speech/Audio
    if 'speech' in second_data:
        description_parts.append(f"Speech: \"{second_data['speech']}\"")
    
    # Camera movement
    if 'camera' in second_data:
        camera_move = second_data['camera']
        if camera_move != 'static':
            description_parts.append(f"Camera {camera_move}")
    
    # Visual effects
    if 'effects' in second_data:
        effects = ', '.join(second_data['effects'])
        description_parts.append(f"Effects: {effects}")
    
    # Objects (if detected)
    if 'objects' in second_data:
        objects = ', '.join(second_data['objects'][:3])  # Top 3 objects
        description_parts.append(f"Visible: {objects}")
    
    # Combine all parts
    if description_parts:
        output_text += '. '.join(description_parts)
    else:
        output_text += "[No data]"

# Add summary
output_text += f"""

SUMMARY:
========
Based on the analysis, this appears to be a TikTok video by Leon Schliebach showing "a day in the life of a hardworking civil servant" (Beamter). 

The video includes:
- German text overlays throughout explaining his morning routine
- Speech narration matching the text
- Various camera movements (tilts, pans)
- Visual effects including motion blur and transitions
- Shows morning preparation, getting ready, making a shake
- Mentions being late and putting on a hat because no time for hair
- Heading to the office with music

The reconstruction captures the essential narrative and visual style of the video through text, speech, and technical details, even without scene descriptions from the StreamingDenseCaptioning analyzer.

Data Coverage: {data['metrics']['coverage_percentage']:.1f}%
Reconstruction Score: {data['reconstruction_score']:.0f}%
"""

# Save reconstruction
output_path = '/home/user/tiktok_production/video_reconstruction.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output_text)

print(f"Reconstruction saved to: {output_path}")
print(f"\nFirst 500 characters:")
print(output_text[:500] + "...")
print(f"\nTotal length: {len(output_text)} characters")