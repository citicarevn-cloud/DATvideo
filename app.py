import streamlit as st
import os
import numpy as np
# --- VÁ LỖI TƯƠNG THÍCH PILLOW ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# ---------------------------------
from PIL import Image
from rembg import remove
from moviepy.editor import *
from gtts import gTTS
from huggingface_hub import InferenceClient
import tempfile
import math

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media V7 - Final", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media V7 - Mascot Fix & Logo Back")
st.markdown("---")

# --- SESSION STATE ---
if 'generated_bg' not in st.session_state: st.session_state['generated_bg'] = None
if 'bg_seed' not in st.session_state: st.session_state['bg_seed'] = 0

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    hf_token = st.text_input("🔑 Hugging Face Token:", type="password")
    st.divider()
    video_ratio = st.radio("Tỷ lệ khung hình:", ("9:16 (Dọc - Tiktok)", "16:9 (Ngang - Youtube)"))
    video_duration = st.slider("Thời lượng (giây):", 10, 60, 20)
    st.divider()
    mascot_scale = st.slider("Độ lớn Mascot:", 0.3, 1.0, 0.7)
    
    # TÙY CHỌN QUAN TRỌNG ĐỂ SỬA LỖI MASCOT
    st.markdown("---")
    st.warning("🦖 **Cài đặt Mascot:**")
    remove_bg_mascot = st.checkbox("Dùng AI tách nền Mascot?", value=True, 
                                  help="Bỏ chọn nếu bạn tải lên ảnh PNG đã tách nền sẵn (để tránh bị lỗi mất hình)")

# --- HÀM HỖ TRỢ ---

def generate_ai_background(prompt, token, seed=0):
    if not token: return None
    final_prompt = f"{prompt}, highly detailed, 8k, cinematic lighting, vivid colors"
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(final_prompt)
    except: return None

def create_video_v7(sim_img, mascot_img, logo_img, bg_img, audio_path, ratio, duration, scale, do_remove_bg):
    # Setup kích thước
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    
    # Xử lý Audio
    audio_clip = AudioFileClip(audio_path)
    final_duration = min(audio_clip.duration, duration)
    if audio_clip.duration > final_duration:
        audio_clip = audio_clip.subclip(0, final_duration)
        
    layers = []
    
    # 1. Background Layer
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
    else:
        bg_clip = ColorClip(size=(w, h), color=(20,20,30)).set_duration(final_duration)
    layers.append(bg_clip)

    # 2. Mascot & Sim Logic
    if mascot_img:
        # XỬ LÝ TÁCH NỀN (THEO YÊU CẦU NGƯỜI DÙNG)
        if do_remove_bg:
            mascot_final = remove(mascot_img)
        else:
            mascot_final = mascot_img # Dùng nguyên ảnh gốc (PNG)
            
        # Resize Mascot
        m_w = int(w * scale) 
        m_h = int(mascot_final.height * (m_w / mascot_final.width))
        mascot_resized = mascot_final.resize((m_w, m_h))
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)
        
        # Vị trí: Đứng giữa màn hình (Center)
        center_y = h * 0.6 
        
        # Hiệu ứng: Thở & Trôi nhẹ
        mascot_anim = (mascot_clip
                       .set_position(lambda t: ('center', center_y - m_h/2 + 10 * math.sin(2*t)))
                       .resize(lambda t: 1 + 0.015 * math.sin(3*t))
                       )
        layers.append(mascot_anim)

        # Sim: Đặt trước ngực Mascot
        s_w = int(m_w * 0.45) # Sim to bằng 45% Mascot
        s_h = int(sim_img.height * (s_w / sim_img.width))
        sim_resized = sim_img.resize((s_w, s_h))
        sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
        
        # Vị trí sim chuyển động theo Mascot
        sim_base_y = center_y + m_h * 0.15 # Vị trí bụng
        
        sim_anim = (sim_clip
                    .set_position(lambda t: ('center', sim_base_y + 10 * math.sin(2*t)))
                    .rotate(lambda t: 3 * math.sin(3*t))
                    )
        layers.append(sim_anim)

    else:
        # Nếu không có Mascot -> Sim đứng 1 mình
        s_w = int(w * 0.65)
        s_h = int(sim_img.height * (s_w / sim_img.width))
        sim_resized = sim_img.resize((s_w, s_h))
        sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
        
        sim_anim = (sim_clip.set_position('center').resize(lambda t: 1 + 0.05 * math.sin(t)))
        layers.append(sim_anim)

    # 3. LOGO Layer (Đã khôi phục)
    if logo_img:
        l_w = int(w * 0.18) # Logo chiếm 18% chiều rộng
        l_h = int(logo_img.height * (l_w / logo_img.width))
        logo_resized = logo_img.resize((l_w, l_h))
        
        logo_clip = ImageClip(np.array(logo_resized)).set_duration(final_duration)
        # Đặt góc trái trên, cách lề 30px
        logo_clip = logo_clip.set_position((30, 40)) 
        layers.append(logo_clip)

    # Render
    final = CompositeVideoClip(layers, size=(w,h)).set_audio(audio_clip)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
        
    return out_path

# --- UI CHÍNH ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Hình ảnh")
    sim_file = st.file_uploader("🖼️ Tải ảnh SIM (PNG đã tách nền):", type=['png'])
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot:", type=['png', 'jpg'])
    logo_file = st.file_uploader("©️ Tải Logo (Sẽ hiện góc trái trên):", type=['png', 'jpg'])
    
with col2:
    st.subheader("2. Bối cảnh & Âm thanh")
    bg_prompt = st.text_input("Mô tả bối cảnh:", value="neon sci-fi tunnel, blue lights, 3d render, 8k")
    
    # Nút Random Background
    if st.button("🎲 Tạo bối cảnh mới"):
        if hf_token:
            st.session_state['bg_seed'] += 1
            with st.spinner("Đang vẽ..."):
                bg = generate_ai_background(bg_prompt, hf_token, st.session_state['bg_seed'])
                st.session_state['generated_bg'] = bg
    
    if st.session_state['generated_bg']:
        st.image(st.session_state['generated_bg'], width=200)

    st.markdown("---")
    voice_type = st.radio("Nguồn âm thanh:", ["🎙️ Tải file ghi âm", "📝 AI Đọc"], horizontal=True)
    
    final_audio = None
    input_script = ""
    
    if voice_type == "📝 AI Đọc":
        input_script = st.text_area("Nhập kịch bản (AI sẽ đọc):", height=100)
    else:
        uploaded_audio = st.file_uploader("Tải file MP3/WAV:", type=['mp3', 'wav'])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(uploaded_audio.getvalue())
                final_audio = fp.name

# Render Button
st.markdown("---")
video_name = st.text_input("Tên video:", "dat_media_ads")

if st.button("🚀 XUẤT BẢN VIDEO (RENDER)", type="primary"):
    if not hf_token or not sim_file:
        st.error("Thiếu Token hoặc Ảnh SIM!")
    elif voice_type == "📝 AI Đọc" and not input_script:
        st.error("Thiếu kịch bản!")
    elif voice_type == "🎙️ Tải file ghi âm" and not final_audio:
        st.error("Thiếu file âm thanh!")
    else:
        status = st.empty()
        prog = st.progress(0)
        
        try:
            # 1. Tạo Audio
            if voice_type == "📝 AI Đọc":
                status.text("🔊 Đang tạo giọng đọc...")
                tts = gTTS(input_script, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    final_audio = fp.name
            
            prog.progress(20)
            
            # 2. Check Background
            bg_final = st.session_state['generated_bg']
            if not bg_final:
                status.text("🎨 Đang vẽ bối cảnh...")
                bg_final = generate_ai_background(bg_prompt, hf_token)
                st.session_state['generated_bg'] = bg_final
            
            prog.progress(40)
            
            # 3. Load Images
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            # 4. Render
            status.text("🎬 Đang xử lý Video (Ghép Logo, Mascot)...")
            # Lấy cài đặt tách nền từ Sidebar
            should_remove_bg = st.sidebar.checkbox("Dùng AI tách nền Mascot?", value=True)
            
            out_vid = create_video_v7(
                sim_pil, mascot_pil, logo_pil, bg_final, final_audio, 
                video_ratio, video_duration, mascot_scale, should_remove_bg
            )
            
            prog.progress(100)
            status.success("Xong!")
            st.video(out_vid)
            
            with open(out_vid, "rb") as f:
                st.download_button("⬇️ Tải về", f, file_name=f"{video_name}.mp4", mime="video/mp4")
                
        except Exception as e:
            st.error(f"Lỗi: {e}")
