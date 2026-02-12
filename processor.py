import cv2
import numpy as np
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip
import moviepy.video.fx as vfx
from PIL import Image, ImageDraw, ImageFont
import os

# Face Detection Setup (MediaPipe with OpenCV Fallback)
try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import face_detection as mp_face_detection
    except ImportError:
        import mediapipe.solutions.face_detection as mp_face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    USE_MEDIAPIPE = True
except Exception as e:
    print(f"MediaPipe not available, falling back to OpenCV Haar Cascades: {e}")
    # Load OpenCV's pre-trained Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    USE_MEDIAPIPE = False

def create_text_image(text, fontsize, color, stroke_color, stroke_width, size=None):
    # Create a transparent image for text
    try:
        # Try to load a bold font, fallback to default
        font = ImageFont.truetype("arialbd.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    # Calculate text size
    dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)
    
    # Handle multi-line if size is provided
    if size:
        max_w = size[0]
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            w = draw.textlength(" ".join(current_line), font=font)
            if w > max_w:
                if len(current_line) > 1:
                    lines.append(" ".join(current_line[:-1]))
                    current_line = [word]
                else:
                    lines.append(" ".join(current_line))
                    current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        text = "\n".join(lines)

    # Re-calculate size for final image
    bbox = draw.multiline_textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20
    
    img = Image.new('RGBA', (int(w), int(h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw stroke
    for offset_x in range(-stroke_width, stroke_width + 1):
        for offset_y in range(-stroke_width, stroke_width + 1):
            draw.multiline_text((10 + offset_x, 10 + offset_y), text, font=font, fill=stroke_color, align="center")
    
    # Draw main text
    draw.multiline_text((10, 10), text, font=font, fill=color, align="center")
    
    return np.array(img)

def get_face_center(frame):
    if USE_MEDIAPIPE:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            best_face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
            bbox = best_face.location_data.relative_bounding_box
            center_x = bbox.xmin + bbox.width / 2
            return center_x
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            # Get the largest face by area
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            center_x = (x + w/2) / frame.shape[1]
            return center_x
    return None

def process_video_with_tracking(video_path, word_data, hook_text, output_path):
    clip = VideoFileClip(video_path)
    w, h = clip.size
    target_w, target_h = 720, 1280
    
    # Calculate crop width for 9:16 aspect ratio based on height
    crop_h = h
    crop_w = int(h * (9/16))
    
    if crop_w > w:
        crop_w = w
        crop_h = int(w * (16/9))

    # Face tracking and smoothing
    centers = []
    last_center = 0.5
    smoothing_window = 10
    
    fps = clip.fps
    total_frames = int(clip.duration * fps)
    
    for i in range(0, total_frames, 5): 
        t = i / fps
        frame = clip.get_frame(t)
        center_x = get_face_center(frame)
        
        if center_x is not None:
            if abs(center_x - last_center) > 0.3:
                centers.extend([center_x] * (i // 5 - len(centers)))
            last_center = center_x
        centers.append(last_center)

    full_centers = np.interp(np.arange(total_frames), np.arange(0, total_frames, 5), centers)
    full_centers = np.convolve(full_centers, np.ones(smoothing_window)/smoothing_window, mode='same')

    def make_frame(get_frame, t):
        frame = get_frame(t)
        frame_idx = min(int(t * fps), total_frames - 1)
        center_x_rel = full_centers[frame_idx]
        
        center_x_px = int(center_x_rel * w)
        x1 = max(0, center_x_px - crop_w // 2)
        x2 = min(w, x1 + crop_w)
        if x2 == w: x1 = w - crop_w
        
        cropped = frame[:, x1:x2]
        return cv2.resize(cropped, (target_w, target_h))

    # Background
    bg_clip = fg_clip.copy().fx(vfx.Resize, height=target_h).fx(vfx.Blur, 20)
    bg_clip = bg_clip.fx(vfx.Crop, x_center=bg_clip.w/2, y_center=bg_clip.h/2, width=target_w, height=target_h)

    # Foreground
    fg_clip = clip.fl(make_frame)
    
    final_clip = CompositeVideoClip([bg_clip, fg_clip.set_position("center")], size=(target_w, target_h))

    # Add Hook Text using PIL
    caption_clips = []
    if hook_text:
        hook_img = create_text_image(hook_text.upper(), 70, "yellow", "black", 3, size=(target_w*0.8, None))
        hook_clip = ImageClip(hook_img).set_start(0).set_duration(3).set_position(('center', 200))
        caption_clips.append(hook_clip)

    # Add Animated Captions using PIL
    for word in word_data:
        if word['start'] < final_clip.duration:
            word_img = create_text_image(word['word'].upper(), 60, "white", "black", 2)
            txt = ImageClip(word_img).set_start(word['start']).set_end(min(word['end'], final_clip.duration)).set_position(('center', target_h * 0.75))
            caption_clips.append(txt)
    
    final_render = CompositeVideoClip([final_clip] + caption_clips)
    
    final_render.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", 
                                 threads=2, logger=None, preset="ultrafast")
    
    clip.close()
    final_render.close()
