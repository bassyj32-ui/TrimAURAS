import streamlit as st
import os
import shutil
from utils import get_video_metadata, get_transcript, analyze_transcript, download_section, transcribe_clip, transcribe_full_video
from processor import process_video_with_tracking
from moviepy import VideoFileClip
import moviepy.video.fx as vfx

# Set Page Config
st.set_page_config(page_title="TrimAURAS", page_icon="✂️", layout="wide")

# Custom CSS for "Cool" Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stSidebar {
        background-color: #1a1c24;
    }
    h1, h2, h3 {
        color: #ff4b4b !important;
    }
    .stAlert {
        background-color: #262730;
        color: white;
        border: 1px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# Load or Initialize API Key
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
    # Try to load from .env file
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    st.session_state.gemini_api_key = line.split("=")[1].strip()

# Sidebar for API Key
with st.sidebar:
    st.title("✂️ TrimAURAS")
    st.markdown("---")
    
    # API Key Input
    new_api_key = st.text_input(
        "Google Gemini API Key", 
        value=st.session_state.gemini_api_key, 
        type="password",
        help="Your key is saved locally in a .env file."
    )
    
    if new_api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = new_api_key
        # Save to .env
        with open(".env", "w") as f:
            f.write(f"GEMINI_API_KEY={new_api_key}")
        st.success("API Key saved locally!")

    st.info("Get your key at: [Google AI Studio](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    input_source = st.radio("Input Source", ["YouTube URL", "Upload Video"])
    
    if st.button("Clear Temp Files"):
        if os.path.exists("temp"):
            shutil.rmtree("temp")
            st.success("Temp files cleared!")

# Main UI
st.title("✂️ TrimAURAS: AI-Powered YouTube Shorts Creator")
st.markdown("Create 3 viral-ready Shorts from any YouTube video or local file in minutes.")

uploaded_file_path = None
youtube_url = None

if input_source == "YouTube URL":
    youtube_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file:
        if not os.path.exists("temp"):
            os.makedirs("temp")
        uploaded_file_path = os.path.join("temp", uploaded_file.name)
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")

if "segments" not in st.session_state:
    st.session_state.segments = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# Logic for Step 1
if st.session_state.gemini_api_key and (youtube_url or uploaded_file_path):
    if st.button("Step 1: Analyze Video") and not st.session_state.processing:
        with st.spinner("Processing transcript and analyzing with Gemini..."):
            try:
                transcript = None
                if input_source == "YouTube URL":
                    # 1. Fetch Metadata
                    info = get_video_metadata(youtube_url)
                    st.write(f"**Found Video:** {info.get('title')}")
                    # 2. Get Transcript
                    transcript = get_transcript(youtube_url)
                else:
                    # 1. Transcribe Uploaded Video
                    st.write("Transcribing uploaded video (this may take a minute)...")
                    transcript = transcribe_full_video(uploaded_file_path)

                if not transcript:
                    st.error("Could not generate/fetch transcript.")
                else:
                    # 3. Analyze with Gemini
                    segments = analyze_transcript(st.session_state.gemini_api_key, transcript)
                    if segments:
                        st.session_state.segments = segments
                        st.success("Analysis complete! Review the segments below.")
                    else:
                        st.error("AI analysis failed. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")

# Step 2: Verification
if st.session_state.segments:
    st.divider()
    st.subheader("Step 2: Verify & Customize Segments")
    
    selected_segments = []
    cols = st.columns(len(st.session_state.segments))
    
    for i, seg in enumerate(st.session_state.segments):
        with cols[i]:
            st.markdown(f"### Segment {i+1}")
            st.write(f"**Hook:** {seg['hook']}")
            st.write(f"**Duration:** {seg['end'] - seg['start']:.1f}s")
            st.write(f"*{seg['description']}*")
            
            start_time = st.number_input(f"Start (s) #{i+1}", value=float(seg['start']), key=f"start_{i}")
            end_time = st.number_input(f"End (s) #{i+1}", value=float(seg['end']), key=f"end_{i}")
            hook_text = st.text_input(f"Hook Text #{i+1}", value=seg['hook'], key=f"hook_{i}")
            
            if st.checkbox(f"Process Segment {i+1}", value=True, key=f"check_{i}"):
                selected_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "hook": hook_text,
                    "id": i
                })

    if st.button("Step 3: Generate Shorts"):
        st.session_state.processing = True
        
        # Create temp dir
        if not os.path.exists("temp"):
            os.makedirs("temp")
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for idx, seg in enumerate(selected_segments):
            try:
                # Update status
                progress = (idx) / len(selected_segments)
                progress_bar.progress(progress)
                
                clip_path = f"temp/raw_clip_{idx}.mp4"
                final_path = f"temp/trimauras_short_{idx}.mp4"
                
                # 1. Get Section
                if input_source == "YouTube URL":
                    status_text.text(f"Processing Clip {idx+1}: Downloading from YouTube...")
                    download_section(youtube_url, seg['start'], seg['end'], clip_path)
                else:
                    status_text.text(f"Processing Clip {idx+1}: Trimming local video...")
                    with VideoFileClip(uploaded_file_path) as video:
                        new_clip = video.subclipped(seg['start'], seg['end'])
                        new_clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
                
                # 2. Transcribe for Captions
                status_text.text(f"Processing Clip {idx+1}: Transcribing for captions...")
                word_data = transcribe_clip(clip_path)
                
                # 3. Render
                status_text.text(f"Processing Clip {idx+1}: Rendering & Face Tracking...")
                process_video_with_tracking(clip_path, word_data, seg['hook'], final_path)
                
                results.append(final_path)
                
            except Exception as e:
                st.error(f"Error processing clip {idx+1}: {e}")
        
        progress_bar.progress(1.0)
        status_text.text("All clips processed!")
        st.session_state.processing = False
        
        # Display Results
        st.divider()
        st.subheader("Step 4: Download Your Shorts")
        res_cols = st.columns(len(results))
        for i, path in enumerate(results):
            with res_cols[i]:
                st.video(path)
                with open(path, "rb") as file:
                    st.download_button(
                        label=f"Download Short {i+1}",
                        data=file,
                        file_name=f"TrimAURAS_Short_{i+1}.mp4",
                        mime="video/mp4"
                    )

else:
    if not st.session_state.gemini_api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    elif not youtube_url and not uploaded_file_path:
        st.warning("Please provide a YouTube URL or upload a video file.")
