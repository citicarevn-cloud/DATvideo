import streamlit as st
import os
import numpy as np
# --- VÁ LỖI TƯƠNG THÍCH PILLOW ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# ---------------------------------
from PIL import Image
from moviepy.editor import *
from gtts import gTTS
from huggingface_hub import InferenceClient
import tempfile
import math
import requests

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media Automation", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media - Hệ thống Tạo Video Tự Động")
st.markdown("---")

# --- SESSION STATE ---
if 'generated_bg' not in st.session_state: st.session_state['generated_bg'] = None
if 'bg_seed' not in st.session_state: st.session_state['bg_seed'] = 0

# --- XỬ LÝ API KEY TỰ ĐỘNG (AUTO-LOGIN) ---
# Kiểm tra xem trong Secrets có lưu Key chưa
sys_hf_token = st.secrets.get("HF_TOKEN", None)
sys_eleven_key = st.secrets.get("ELEVEN_KEY", None)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    
    # Xử lý Hugging Face Token
    if sys_hf_token:
        st.success("✅ HuggingFace: Đã kết nối")
        hf_token = sys_hf_token
    else:
        hf_token = st.text_input("🔑 Nhập Hugging Face Token:", type="password")

    # Xử lý ElevenLabs Key
    if sys_eleven_key:
        st.success("✅ ElevenLabs: Đã kết nối")
        elevenlabs_key = sys_eleven_key
    else:
        elevenlabs_key = st.text_input("🎤 Nhập ElevenLabs Key:", type="password")
    
    st.divider()
    st.header("⚙️ Thông số Video")
    video_ratio = st.radio("Tỷ lệ:", ("9:16 (Dọc)", "16:9 (Ngang)"))
    video_duration = st.slider("Thời lượng (s):", 10, 60, 20)
    mascot_scale = st.slider("Mascot Zoom:", 0.3, 1.0, 0.7)

# --- HÀM HỖ TRỢ ---
def generate_ai_background(prompt, token, seed=0):
    if not token: return None
    final_prompt = f"{prompt}, highly detailed, 8k, cinematic lighting, vivid colors"
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(final_prompt)
    except: return None

def clone_and_speak(api_key, text, sample_audio_path):
    if not api_key: return None
    add_url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": api_key}
    voice_name = f"Clone_{os.urandom(4).hex()}"
    files = {
        'files': open(sample_audio_path, 'rb'),
        'name': (None, voice_name),
        'description': (None, "DAT Media Clone")
    }
    try:
        response_add = requests.post(add_url, headers=headers, files=files)
        if response_add.status_code != 200: return None
        voice_id = response_add.json()['voice_id']
        
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        data = {
            "text": text, 
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }
        headers_json = {"xi-api-key": api_key, "Content-Type": "application/json"}
        response_tts = requests.post(tts_url, json=data, headers=headers_json)
        
        if response_tts.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(response_tts.content)
                output_path = fp.name
            requests.delete(f"https://api.elevenlabs.io/v1/voices/{voice_id}", headers=headers)
            return output_path
        else: return None
    except: return None

def create_video_v10(sim_img, mascot_img, logo_img, bg_img, audio_path, ratio, duration, scale):
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    audio_clip = AudioFileClip(audio_path)
    final_duration = min(audio_clip.duration, duration)
    if audio_clip.duration > final_duration:
        audio_clip = audio_clip.subclip(0, final_duration)
        
    layers = []
    # BG
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
        layers.append(bg_clip)
    else:
        layers.append(ColorClip(size=(w, h), color=(20,20,30)).set_duration(final_duration))

    # Mascot & Sim
    if mascot_img:
        mascot_final = mascot_img
        m_w = int(w * scale) 
        m_h = int(mascot_final.height * (m_w / mascot_final.width))
        mascot_resized = mascot_final.resize((m_w, m_h))
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)
        
        center_y = h * 0.6 
        mascot_anim = (mascot_clip
                       .set_position(lambda t: ('center', center_y - m_h/2 + 10 * math.sin(2*t)))
                       .resize(lambda t: 1 + 0.015 * math.sin(3*t)))
        layers.append(mascot_anim)

        s_w = int(m_w * 0.45)
        s_h = int(sim_img.height * (s_w / sim_img.width))
        sim_resized = sim_img.resize((s_w, s_h))
        sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
        sim_base_y = center_y + m_h * 0.15
        sim_anim = (sim_clip
                    .set_position(lambda t: ('center', sim_base_y + 10 * math.sin(2*t)))
                    .rotate(lambda t: 3 * math.sin(3*t)))
        layers.append(sim_anim)
    else:
        s_w = int(w * 0.65)
        s_h = int(sim_img.height * (s_w / sim_img.width))
        sim_resized = sim_img.resize((s_w, s_h))
        sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
        sim_anim = (sim_clip.set_position('center').resize(lambda t: 1 + 0.05 * math.sin(t)))
        layers.append(sim_anim)

    # Logo
    if logo_img:
        l_w = int(w * 0.18)
        l_h = int(logo_img.height * (l_w / logo_img.width))
        logo_resized = logo_img.resize((l_w, l_h))
        logo_clip = ImageClip(np.array(logo_resized)).set_duration(final_duration).set_position((30, 40)) 
        layers.append(logo_clip)

    final = CompositeVideoClip(layers, size=(w,h)).set_audio(audio_clip)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
    return out_path

# --- UI CHÍNH ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Hình ảnh (PNG)")
    sim_file = st.file_uploader("🖼️ Tải ảnh SIM:", type=['png'])
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot:", type=['png'])
    logo_file = st.file_uploader("©️ Tải Logo:", type=['png'])
    
with col2:
    st.subheader("2. Bối cảnh")
    bg_prompt = st.text_input("Mô tả bối cảnh:", value="neon sci-fi tunnel, blue lights, 3d render, 8k")
    if st.button("🎲 Tạo bối cảnh mới"):
        if hf_token:
            st.session_state['bg_seed'] += 1
            with st.spinner("Đang vẽ..."):
                bg = generate_ai_background(bg_prompt, hf_token, st.session_state['bg_seed'])
                st.session_state['generated_bg'] = bg
    if st.session_state['generated_bg']: st.image(st.session_state['generated_bg'], width=200)

st.markdown("---")
st.subheader("3. Âm thanh")
voice_mode = st.radio("Chọn chế độ âm thanh:", ["📝 AI Đọc (Google - Free)", "🤖 AI Clone (ElevenLabs - Pro)", "🎙️ File có sẵn"])

final_audio_path = None
input_script = ""

if voice_mode == "📝 AI Đọc (Google - Free)":
    input_script = st.text_area("Nhập kịch bản:", height=100)
elif voice_mode == "🤖 AI Clone (ElevenLabs - Pro)":
    input_script = st.text_area("Nhập kịch bản:", height=100)
    sample_voice = st.file_uploader("Mẫu giọng Clone (MP3):", type=['mp3'])
else:
    uploaded_audio = st.file_uploader("File thu âm (MP3):", type=['mp3'])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            fp.write(uploaded_audio.getvalue())
            final_audio_path = fp.name

st.markdown("---")
video_name = st.text_input("Tên file:", "dat_media_ads")
if st.button("🚀 XUẤT BẢN VIDEO", type="primary"):
    error = False
    if not sim_file: st.error("Thiếu ảnh SIM!"); error=True
    if voice_mode == "🤖 AI Clone (ElevenLabs - Pro)":
        if not elevenlabs_key: st.error("Thiếu ElevenLabs Key (Cài trong Secrets)!"); error=True
        if not sample_voice: st.error("Thiếu mẫu giọng!"); error=True
        
    if not error:
        status = st.empty()
        prog = st.progress(0)
        try:
            if voice_mode == "📝 AI Đọc (Google - Free)":
                status.text("🔊 Đang tạo giọng...")
                tts = gTTS(input_script, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    final_audio_path = fp.name
            elif voice_mode == "🤖 AI Clone (ElevenLabs - Pro)":
                status.text("🧬 Đang Clone giọng...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(sample_voice.getvalue())
                    final_audio_path = clone_and_speak(elevenlabs_key, input_script, tmp.name)
                if not final_audio_path: st.stop()
            
            prog.progress(30)
            bg_final = st.session_state['generated_bg']
            if not bg_final and hf_token:
                status.text("🎨 Đang vẽ bối cảnh...")
                bg_final = generate_ai_background(bg_prompt, hf_token)
            
            prog.progress(50)
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            status.text("🎬 Rendering...")
            out = create_video_v10(sim_pil, mascot_pil, logo_pil, bg_final, final_audio_path, video_ratio, 20, mascot_scale)
            
            prog.progress(100)
            status.success("Xong!")
            st.video(out)
            with open(out, "rb") as f: st.download_button("⬇️ Tải về", f, file_name=f"{video_name}.mp4")
        except Exception as e: st.error(f"Lỗi: {e}")
