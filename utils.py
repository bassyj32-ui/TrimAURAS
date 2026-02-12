import yt_dlp
import google.generativeai as genai
import json
import os
import re
from faster_whisper import WhisperModel

def get_video_metadata(url):
    ydl_opts = {
        'get_subs': True,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info

def get_transcript(url):
    ydl_opts = {
        'skip_download': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'quiet': True,
        'no_warnings': True,
        'outtmpl': 'temp_subs.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Check for vtt files
    subs_file = None
    for f in os.listdir('.'):
        if f.startswith('temp_subs') and (f.endswith('.vtt') or f.endswith('.srt')):
            subs_file = f
            break
    
    if not subs_file:
        return None
    
    with open(subs_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Basic VTT parsing to text
    text = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])
    
    os.remove(subs_file)
    return text

def analyze_transcript(api_key, transcript):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Analyze the following video transcript and find 3 engaging segments under 60 seconds each.
    For each segment, provide:
    1. Start and end timestamps (in seconds).
    2. A catchy "Hook" text for the first 3 seconds of the clip (e.g., "HE ACTUALLY DID IT...", "THIS IS INSANE!").
    3. A brief description of why this segment is engaging.

    Transcript:
    {transcript[:15000]} # Limit transcript length for safety

    Return ONLY a JSON list of objects with keys: "start", "end", "hook", "description".
    """
    
    response = model.generate_content(prompt)
    try:
        # Clean response text in case Gemini adds markdown formatting
        json_str = response.text
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return None

def transcribe_full_video(video_path):
    # Use faster-whisper with tiny model and int8 quantization
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, beam_size=5)
    
    transcript_text = ""
    for segment in segments:
        transcript_text += f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n"
    return transcript_text

def download_section(url, start, end, output_path):
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': output_path,
        'download_sections': f'*[{start}-{end}]',
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def transcribe_clip(video_path):
    # Use faster-whisper with tiny model and int8 quantization
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, beam_size=5, word_timestamps=True)
    
    word_data = []
    for segment in segments:
        for word in segment.words:
            word_data.append({
                'word': word.word,
                'start': word.start,
                'end': word.end
            })
    return word_data
