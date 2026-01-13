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
import random

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media Creator V5", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    /* Nút tạo nền màu xanh lá */
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    /* Nút xuất bản màu đỏ nổi bật */
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media - Video Creator V5 (Mascot MC)")
st.markdown("---")

# --- KHỞI TẠO SESSION STATE (LƯU TRẠNG THÁI) ---
if 'generated_bg' not in st.session_state:
    st.session_state['generated_bg'] = None
if 'bg_seed' not in st.session_state:
    st.session_state['bg_seed'] = 0

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình Kỹ thuật")
    hf_token = st.text_input("🔑 Hugging Face Token:", type="password")
    
    st.divider()
    
    video_ratio = st.radio("Tỷ lệ khung hình:", ("9:16 (Dọc - Tiktok)", "16:9 (Ngang - Youtube)"))
    video_duration = st.slider("Thời lượng video (giây):", 10, 60, 20)
    
    st.divider()
    st.subheader("🎭 Chế độ Diễn xuất")
    # Tùy chọn mới cho Mascot
    mascot_mode = st.radio("Vai trò của Mascot:", 
                          ["MC cầm SIM giới thiệu (Mới)", "Đứng góc phụ họa (Cũ)"])
    
    st.divider()
    effect_type = st.selectbox(
        "Hiệu ứng chuyển động:",
        ["Nhún nhảy (Bounce)", "Lắc lư (Shake)", "Zoom kịch tính", "Đứng yên"]
    )

# --- HÀM HỖ TRỢ ---

def get_dominant_color_hex(pil_img):
    img = pil_img.copy().convert("RGBA").resize((50, 50))
    pixels = img.getcolors(50 * 50)
    if not pixels: return None
    sorted_pixels = sorted(pixels, key=lambda t: t[0], reverse=True)
    for count, color in sorted_pixels:
        if len(color) == 4 and color[3] < 200: continue 
        if sum(color[:3]) > 700 or sum(color[:3]) < 50: continue
        return '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    return None

def generate_ai_background(prompt, token, color_hex=None, seed=0):
    if not token: return None
    final_prompt = prompt
    if color_hex:
        final_prompt = f"background theme color {color_hex}, {prompt}"
    
    # Thêm yếu tố ngẫu nhiên vào prompt để ảnh thay đổi
    random_styles = ["cinematic lighting", "studio lighting", "soft focus", "vibrant colors"]
    final_prompt += f", {random_styles[seed % len(random_styles)]}"
    
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(final_prompt)
    except: return None

# Hàm hiệu ứng nâng cao
def apply_advanced_effect(clip, effect_name, start_time=0):
    if effect_name == "Zoom kịch tính":
        return clip.resize(lambda t: 1 + 0.05 * t).set_position('center')
    elif effect_name == "Lắc lư (Shake)":
        return clip.rotate(lambda t: 5 * math.sin(2 * math.pi * t + start_time)).set_position('center')
    elif effect_name == "Nhún nhảy (Bounce)":
        # Nhún lên xuống
        return clip.set_position(lambda t: ('center', 100 + 20 * math.sin(5*t))) # Y offset relative
    else:
        return clip.set_position('center')

def create_video_v5(sim_img, mascot_img, logo_img, bg_img, audio_path, effect, ratio, duration, mode):
    # 1. Setup
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    
    # 2. Audio
    audio_clip = AudioFileClip(audio_path)
    final_duration = min(audio_clip.duration, duration)
    if audio_clip.duration > final_duration:
        audio_clip = audio_clip.subclip(0, final_duration)
        
    layers = []
    
    # 3. Background Layer
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
    else:
        bg_clip = ColorClip(size=(w, h), color=(20,20,20)).set_duration(final_duration)
    layers.append(bg_clip)

    # 4. XỬ LÝ MASCOT VÀ SIM (CORE LOGIC MỚI)
    
    # Chuẩn bị ảnh Mascot
    mascot_clip = None
    if mascot_img:
        mascot_nobg = remove(mascot_img)
        # Nếu chế độ MC: Mascot to hơn, đứng giữa
        m_scale = 0.65 if mode == "MC cầm SIM giới thiệu (Mới)" else 0.35
        m_w = int(w * m_scale)
        m_h = int(mascot_nobg.height * (m_w / mascot_nobg.width))
        mascot_resized = mascot_nobg.resize((m_w, m_h))
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)

    # Chuẩn bị ảnh SIM
    s_scale = 0.4 if mode == "MC cầm SIM giới thiệu (Mới)" else 0.5
    s_w = int(w * s_scale)
    s_h = int(sim_img.height * (s_w / sim_img.width))
    sim_resized = sim_img.resize((s_w, s_h))
    sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)

    # --- LOGIC GHÉP VÀ CHUYỂN ĐỘNG ---
    
    if mode == "MC cầm SIM giới thiệu (Mới)" and mascot_clip:
        # 1. Mascot đứng giữa dưới
        mascot_pos_y = h - m_h + 50 # Thụt xuống chút cho tự nhiên
        mascot_clip = mascot_clip.set_position(('center', mascot_pos_y))
        
        # Tạo hiệu ứng chuyển động cho Mascot (Ví dụ: Nhún nhảy)
        if effect == "Nhún nhảy (Bounce)":
            mascot_clip = mascot_clip.set_position(lambda t: ('center', mascot_pos_y + 10 * math.sin(4*t)))
        elif effect == "Lắc lư (Shake)":
            mascot_clip = mascot_clip.rotate(lambda t: 2 * math.sin(2*t)).set_position(('center', mascot_pos_y))
            
        layers.append(mascot_clip)
        
        # 2. SIM đặt đè lên Mascot (Vị trí tay cầm giả định)
        # Giả định tay cầm nằm ở khoảng 60% chiều cao mascot từ trên xuống
        sim_pos_y_base = mascot_pos_y + m_h * 0.4 
        
        # SIM chuyển động ĐỒNG BỘ với Mascot
        if effect == "Nhún nhảy (Bounce)":
            # Mascot nhún, SIM cũng phải nhún theo cùng nhịp (4*t)
            sim_clip = sim_clip.set_position(lambda t: ('center', sim_pos_y_base + 10 * math.sin(4*t)))
        elif effect == "Lắc lư (Shake)":
            # Mascot lắc, SIM lắc theo nhưng biên độ lớn hơn xíu để sinh động
            sim_clip = sim_clip.rotate(lambda t: 2 * math.sin(2*t)).set_position(lambda t: ('center', sim_pos_y_base))
        else:
             sim_clip = sim_clip.set_position(('center', sim_pos_y_base))
             
        # Thêm hiệu ứng SIM "nổi bật" (Zoom nhẹ độc lập)
        sim_clip = sim_clip.resize(lambda t: 1 + 0.02 * math.sin(t))
        
        layers.append(sim_clip)
        
    else:
        # Chế độ Cũ: Mascot góc, SIM giữa
        if mascot_clip:
            pos = ('right', 'bottom') if "16:9" in ratio else ('center', 'bottom')
            mascot_clip = mascot_clip.set_position(pos)
            layers.append(mascot_clip)
            
        # SIM độc lập
        sim_clip = sim_clip.set_position(('center', 'center'))
        if effect == "Nhún nhảy (Bounce)":
            sim_clip = sim_clip.set_position(lambda t: ('center', h/2 - s_h/2 + 20 * math.sin(2*t)))
        elif effect == "Lắc lư (Shake)":
             sim_clip = sim_clip.rotate(lambda t: 5 * math.sin(2*math.pi*t)).set_position('center')
        
        layers.append(sim_clip)

    # 5. Logo
    if logo_img:
        l_w = int(w * 0.15)
        l_h = int(logo_img.height * (l_w / logo_img.width))
        logo_resized = logo_img.resize((l_w, l_h))
        logo_clip = ImageClip(np.array(logo_resized)).set_duration(final_duration).set_position((30, 30))
        layers.append(logo_clip)
    
    # 6. Render
    final = CompositeVideoClip(layers, size=(w,h)).set_audio(audio_clip)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
        
    return out_path

# --- UI CHÍNH ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Hình ảnh & Thương hiệu")
    sim_file = st.file_uploader("🖼️ Tải ảnh SIM (PNG đã tách nền):", type=['png'])
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot (Nên dùng ảnh toàn thân):", type=['png', 'jpg'])
    logo_file = st.file_uploader("©️ Tải Logo:", type=['png', 'jpg'])
    
    logo_color_hint = None
    if logo_file:
        logo_color_hint = get_dominant_color_hex(Image.open(logo_file))

with col2:
    st.subheader("2. Bối cảnh (Background)")
    bg_prompt = st.text_input("Mô tả bối cảnh:", value="modern abstract technology background, blue lights, 3d render")
    
    # NÚT TẠO RIÊNG BIỆT CHO BACKGROUND
    col_bg_btn, col_bg_preview = st.columns([1, 2])
    with col_bg_btn:
        if st.button("🎲 Tạo lại bối cảnh mới"):
            if not hf_token:
                st.error("Cần nhập Token trước!")
            else:
                st.session_state['bg_seed'] += 1 # Tăng seed để ảnh khác đi
                with st.spinner("Đang vẽ nền mới..."):
                    # Lấy màu logo (nếu có)
                    if logo_file and not logo_color_hint:
                        logo_color_hint = get_dominant_color_hex(Image.open(logo_file))
                    
                    new_bg = generate_ai_background(bg_prompt, hf_token, logo_color_hint, st.session_state['bg_seed'])
                    st.session_state['generated_bg'] = new_bg
    
    with col_bg_preview:
        if st.session_state['generated_bg']:
            st.image(st.session_state['generated_bg'], caption="Bối cảnh hiện tại", width=200)
        else:
            st.info("Chưa có bối cảnh. Hãy bấm nút 'Tạo lại' hoặc chờ hệ thống tự tạo khi xuất video.")

    st.markdown("---")
    st.subheader("3. Âm thanh")
    voice_option = st.radio("Nguồn giọng đọc:", ["🎙️ Tải file ghi âm", "📝 AI Đọc"], horizontal=True)
    
    final_audio_path = None
    script_content = ""
    if voice_option == "📝 AI Đọc":
        script_content = st.text_area("Nhập kịch bản:", height=100)
    else:
        uploaded_audio = st.file_uploader("Tải file MP3/WAV:", type=['mp3', 'wav'])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(uploaded_audio.getvalue())
                final_audio_path = fp.name

st.markdown("---")
video_name_input = st.text_input("4. Đặt tên file video:", "video_dat_media_mascot")

# --- NÚT XỬ LÝ FINAL ---
if st.button("🚀 XUẤT BẢN VIDEO (RENDER)", type="primary"):
    # Kiểm tra input
    valid = True
    if not hf_token: st.error("Thiếu Token!"); valid = False
    if not sim_file: st.error("Thiếu ảnh SIM!"); valid = False
    if voice_option == "📝 AI Đọc" and not script_content: st.error("Thiếu kịch bản!"); valid = False
    if voice_option == "🎙️ Tải file ghi âm" and not final_audio_path: st.error("Thiếu file âm thanh!"); valid = False
    
    if valid:
        status = st.empty()
        progress = st.progress(0)
        
        try:
            # 1. Audio AI (nếu chọn)
            if voice_option == "📝 AI Đọc":
                status.text("🔊 Đang tạo giọng đọc...")
                tts = gTTS(script_content, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    final_audio_path = fp.name
            
            progress.progress(20)
            
            # 2. Background (Nếu chưa có thì tạo, nếu có rồi thì dùng lại)
            bg_to_use = st.session_state['generated_bg']
            if not bg_to_use:
                status.text("🎨 Đang vẽ bối cảnh lần đầu...")
                if logo_file and not logo_color_hint:
                     logo_color_hint = get_dominant_color_hex(Image.open(logo_file))
                bg_to_use = generate_ai_background(bg_prompt, hf_token, logo_color_hint, 0)
                st.session_state['generated_bg'] = bg_to_use
            
            progress.progress(40)
            
            # 3. Load Images
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            # 4. Render
            status.text("🎬 Đang diễn hoạt Mascot và SIM...")
            out_video = create_video_v5(
                sim_pil, mascot_pil, logo_pil, bg_to_use, 
                final_audio_path, effect_type, 
                video_ratio, video_duration, mascot_mode
            )
            
            progress.progress(100)
            status.success("✅ Thành công!")
            st.video(out_video)
            
            with open(out_video, "rb") as f:
                st.download_button("⬇️ Tải về", f, file_name=f"{video_name_input}.mp4", mime="video/mp4")
                
        except Exception as e:
            st.error(f"Lỗi: {e}")
