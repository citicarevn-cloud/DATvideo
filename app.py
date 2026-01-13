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
import asyncio
import edge_tts # Thư viện mới: Microsoft Edge TTS

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media V12 - Edge TTS", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media V12 - Giọng Đọc Cao Cấp (Free)")
st.markdown("---")

# --- SESSION STATE ---
if 'generated_bg' not in st.session_state: st.session_state['generated_bg'] = None
if 'bg_seed' not in st.session_state: st.session_state['bg_seed'] = 0

# --- AUTO LOGIN ---
sys_hf_token = st.secrets.get("HF_TOKEN", None)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    if sys_hf_token:
        st.success("✅ HuggingFace: Connected")
        hf_token = sys_hf_token
    else:
        hf_token = st.text_input("🔑 Hugging Face Token:", type="password")

    st.divider()
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

# --- HÀM TẠO GIỌNG MICROSOFT EDGE (MỚI) ---
async def generate_edge_tts(text, voice_short_name, output_file):
    communicate = edge_tts.Communicate(text, voice_short_name)
    await communicate.save(output_file)

def get_audio_from_edge(text, gender):
    # Chọn giọng dựa trên giới tính
    if gender == "Nữ (Hoài My - Nhẹ nhàng)":
        voice = "vi-VN-HoaiMyNeural"
    else:
        voice = "vi-VN-NamMinhNeural"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        output_path = fp.name
    
    # Chạy hàm async trong môi trường sync
    try:
        asyncio.run(generate_edge_tts(text, voice, output_path))
        return output_path
    except Exception as e:
        st.error(f"Lỗi tạo giọng Edge TTS: {e}")
        return None

def create_video_v12(sim_img, mascot_img, logo_img, bg_img, audio_path, ratio, duration, scale):
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
st.subheader("3. Âm thanh (Microsoft Neural)")

# Menu chọn chế độ mới
voice_mode = st.radio("Chọn nguồn âm thanh:", 
                     ["💎 AI Đọc (Microsoft - Free & Hay)", 
                      "📝 Google Translate (Cũ)", 
                      "🎙️ File có sẵn"])

final_audio_path = None
input_script = ""

if voice_mode == "💎 AI Đọc (Microsoft - Free & Hay)":
    voice_gender = st.selectbox("Chọn giọng:", ["Nữ (Hoài My - Nhẹ nhàng)", "Nam (Nam Minh - Truyền cảm)"])
    input_script = st.text_area("Nhập kịch bản quảng cáo:", height=100)

elif voice_mode == "📝 Google Translate (Cũ)":
    input_script = st.text_area("Nhập kịch bản:", height=100)
    
else:
    uploaded_audio = st.file_uploader("Tải file MP3 đã thu âm:", type=['mp3'])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            fp.write(uploaded_audio.getvalue())
            final_audio_path = fp.name

st.markdown("---")
video_name = st.text_input("Tên file:", "dat_media_ads")

if st.button("🚀 XUẤT BẢN VIDEO", type="primary"):
    error = False
    if not sim_file: st.error("Thiếu ảnh SIM!"); error=True
    if (voice_mode != "🎙️ File có sẵn") and not input_script: st.error("Thiếu kịch bản!"); error=True
        
    if not error:
        status = st.empty()
        prog = st.progress(0)
        try:
            # 1. Xử lý Audio
            if voice_mode == "💎 AI Đọc (Microsoft - Free & Hay)":
                status.text("🔊 Đang tạo giọng Microsoft Neural...")
                final_audio_path = get_audio_from_edge(input_script, voice_gender)
                if not final_audio_path: st.stop()
                
            elif voice_mode == "📝 Google Translate (Cũ)":
                status.text("🔊 Đang tạo giọng Google...")
                tts = gTTS(input_script, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    final_audio_path = fp.name
            
            prog.progress(30)
            
            # 2. Background
            bg_final = st.session_state['generated_bg']
            if not bg_final and hf_token:
                status.text("🎨 Đang vẽ bối cảnh...")
                bg_final = generate_ai_background(bg_prompt, hf_token)
            
            prog.progress(50)
            
            # 3. Load Images
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            # 4. Render
            status.text("🎬 Rendering...")
            out = create_video_v12(sim_pil, mascot_pil, logo_pil, bg_final, final_audio_path, video_ratio, 20, mascot_scale)
            
            prog.progress(100)
            status.success("Xong!")
            st.video(out)
            with open(out, "rb") as f: st.download_button("⬇️ Tải về", f, file_name=f"{video_name}.mp4")
        except Exception as e: st.error(f"Lỗi: {e}")
