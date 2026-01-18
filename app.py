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
from huggingface_hub import InferenceClient
import tempfile
import math
import asyncio
import edge_tts
import random
import requests

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media V17 - Smart Layout", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

# --- NÚT RESET ---
col_title, col_reset = st.columns([3, 1])
with col_title:
    st.title("🎬 DAT Media V17 - Smart Layout")
with col_reset:
    if st.button("🔄 Làm mới (Reset)"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

st.markdown("---")

# --- SESSION STATE ---
if 'generated_bg' not in st.session_state: st.session_state['generated_bg'] = None
if 'current_prompt' not in st.session_state: st.session_state['current_prompt'] = ""

# --- AUTO LOGIN ---
sys_hf_token = st.secrets.get("HF_TOKEN", None)
sys_eleven_key = st.secrets.get("ELEVEN_KEY", None)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình API")
    if sys_hf_token:
        st.success("✅ HuggingFace: Connected")
        hf_token = sys_hf_token
    else:
        hf_token = st.text_input("🔑 Hugging Face Token:", type="password")

    if sys_eleven_key:
        st.success("✅ ElevenLabs: Connected")
        elevenlabs_key = sys_eleven_key
    else:
        elevenlabs_key = st.text_input("🎤 ElevenLabs Key:", type="password")
    
    st.divider()
    st.header("⚙️ Video & Hiệu ứng")
    video_ratio = st.radio("Tỷ lệ:", ("9:16 (Dọc)", "16:9 (Ngang)"))
    
    sim_effect_name = st.selectbox(
        "Hiệu ứng SIM:",
        [
            "1. Lơ lửng (Floating)",
            "2. Nảy tưng tưng (Bounce)",
            "3. Lắc lư (Swing)",
            "4. Phóng to thu nhỏ (Pulse)",
            "5. Xoay tròn 3D (Spin 3D)",
            "6. Trượt ngang (Slide)",
            "7. Rung lắc (Shake)"
        ]
    )
    mascot_scale = st.slider("Độ lớn Mascot:", 0.3, 1.2, 0.75)
    sim_scale_factor = st.slider("Độ lớn Sim:", 0.5, 1.2, 0.8)

# --- HÀM HỖ TRỢ ---
def get_smart_prompt(theme):
    lighting = random.choice(["cinematic lighting", "soft sunlight", "neon glow", "studio lighting", "golden hour"])
    detail = "highly detailed, 8k, professional photography, depth of field"
    
    if theme == "Văn phòng hiện đại":
        scene = random.choice(["modern office desk", "coworking space", "glass meeting room", "minimalist tech workspace"])
        return f"{scene}, blurred background, {lighting}, {detail}"
    elif theme == "Ngoài trời / Thiên nhiên":
        scene = random.choice(["beautiful park sunny day", "city street blurred", "beach sunny", "green garden"])
        return f"{scene}, bokeh background, natural light, {detail}"
    elif theme == "Trong nhà / Ấm cúng":
        scene = random.choice(["cozy living room", "coffee shop window", "wooden table shelf", "modern apartment"])
        return f"{scene}, warm tones, {lighting}, {detail}"
    elif theme == "Công nghệ / Trừu tượng":
        scene = random.choice(["abstract data stream", "blue digital tunnel", "futuristic circuit board", "3d render geometric"])
        return f"{scene}, neon blue and purple, cyber style, {detail}"
    else:
        return f"abstract background, professional, {detail}"

def generate_ai_background(prompt, token):
    if not token: return None
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(prompt)
    except: return None

# --- AUDIO ---
async def generate_edge_tts(text, voice_short_name, output_file):
    communicate = edge_tts.Communicate(text, voice_short_name)
    await communicate.save(output_file)

def get_audio_from_edge(text, gender):
    voice = "vi-VN-HoaiMyNeural" if "Nữ" in gender else "vi-VN-NamMinhNeural"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        output_path = fp.name
    try:
        asyncio.run(generate_edge_tts(text, voice, output_path))
        return output_path
    except Exception as e: st.error(f"Lỗi Edge TTS: {e}"); return None

def speak_with_elevenlabs(api_key, text, voice_id):
    if not api_key: return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    data = {"text": text, "model_id": "eleven_multilingual_v2"}
    try:
        r = requests.post(url, json=data, headers=headers)
        if r.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(r.content); return fp.name
        else: st.error(f"Lỗi ElevenLabs: {r.text}"); return None
    except Exception as e: st.error(f"Lỗi kết nối: {e}"); return None

# --- TRANSFORM SIM ---
def apply_sim_transform(clip, effect_name, cx, cy):
    if "Floating" in effect_name:
        return clip.set_position(lambda t: (cx, cy + 15*math.sin(2*t))).rotate(lambda t: 3*math.sin(t))
    elif "Bounce" in effect_name:
        return clip.set_position(lambda t: (cx, cy + abs(40*math.sin(3*t)) - 20))
    elif "Swing" in effect_name:
        return clip.rotate(lambda t: 15 * math.sin(2.5*t)).set_position((cx, cy))
    elif "Pulse" in effect_name:
        return clip.resize(lambda t: 1 + 0.05 * math.sin(4*t)).set_position('center').set_position((cx, cy))
    elif "Spin 3D" in effect_name:
        return clip.resize(lambda t: (abs(math.cos(2*t)) + 0.1, 1)).set_position('center').set_position((cx, cy))
    elif "Slide" in effect_name:
        return clip.set_position(lambda t: (cx + 60*math.sin(2*t), cy))
    elif "Shake" in effect_name:
        return clip.set_position(lambda t: (cx + 5*math.sin(20*t), cy + 5*math.cos(15*t)))
    else:
        return clip.set_position((cx, cy))

# --- VIDEO CORE V17 (SMART LAYOUT) ---
def create_video_v17(sim_img, mascot_img, logo_img, bg_img, audio_path, ratio, sim_effect_mode, m_scale, s_scale_input):
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    
    audio_clip = AudioFileClip(audio_path)
    final_duration = audio_clip.duration + 1
    
    layers = []
    
    # 1. Background
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
        layers.append(bg_clip)
    else:
        layers.append(ColorClip(size=(w, h), color=(20,20,30)).set_duration(final_duration))

    # --- THUẬT TOÁN SẮP XẾP VỊ TRÍ (SMART LAYOUT ENGINE) ---
    
    # Bước 1: Xác định Vùng An Toàn của Logo (Logo Safe Zone)
    logo_bottom_limit = 50 # Mặc định nếu không có logo
    if logo_img:
        l_w = int(w * 0.18)
        l_h = int(logo_img.height * (l_w / logo_img.width))
        logo_resized = logo_img.resize((l_w, l_h))
        logo_pos = (30, 40)
        
        # Vùng cấm: Y của logo + Chiều cao logo + 20px padding
        logo_bottom_limit = logo_pos[1] + l_h + 20
        
        logo_clip = ImageClip(np.array(logo_resized)).set_duration(final_duration).set_position(logo_pos)
        layers.append(logo_clip)

    # Bước 2: Tính toán SIM và Mascot (Dự kiến ban đầu)
    
    # -- Chuẩn bị Mascot --
    mascot_h_final = 0
    mascot_y_final = h # Mặc định ẩn
    
    if mascot_img:
        m_w = int(w * m_scale)
        m_h = int(mascot_img.height * (m_w / mascot_img.width))
        mascot_resized = mascot_img.resize((m_w, m_h))
        mascot_h_final = m_h
        # Vị trí dự kiến: Cách đáy màn hình 1 chút
        mascot_y_final = h - m_h * 0.85 

    # -- Chuẩn bị SIM --
    s_w = int(w * s_scale_input)
    s_h = int(sim_img.height * (s_w / sim_img.width))
    sim_resized = sim_img.resize((s_w, s_h))
    
    # Vị trí dự kiến của SIM:
    if mascot_img:
        # Nằm trên đầu Mascot, chồng lên nhau khoảng 50px để tạo liên kết
        sim_y_final = mascot_y_final - s_h + 50 
    else:
        # Giữa màn hình
        sim_y_final = (h - s_h) / 2

    # Bước 3: KIỂM TRA VA CHẠM (COLLISION CHECK) & ĐIỀU CHỈNH
    
    # Nếu Đỉnh SIM cao hơn Đáy Logo (tức là sim_y_final NHỎ HƠN logo_bottom_limit)
    if sim_y_final < logo_bottom_limit:
        # Tính khoảng cách bị chồng lấn
        overlap_distance = logo_bottom_limit - sim_y_final
        
        # Đẩy SIM xuống
        sim_y_final += overlap_distance
        
        # Nếu có Mascot, cũng phải đẩy Mascot xuống theo để giữ liên kết
        if mascot_img:
            mascot_y_final += overlap_distance

    # --- RENDER VỚI TỌA ĐỘ ĐÃ ĐIỀU CHỈNH ---

    # Render Mascot
    if mascot_img:
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)
        mascot_anim = (mascot_clip
                       .set_position(('center', mascot_y_final))
                       .resize(lambda t: 1 + 0.01 * math.sin(2*t)))
        layers.append(mascot_anim)

    # Render Sim
    sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
    sim_x_final = (w - s_w) / 2
    
    # Áp dụng hiệu ứng bay lượn (với tọa độ Y đã được fix lỗi chồng lấn)
    sim_final = apply_sim_transform(sim_clip, sim_effect_mode, sim_x_final, sim_y_final)
    
    # Fix X center cho hiệu ứng không trượt
    if "Slide" not in sim_effect_mode:
         sim_final = sim_final.set_position(lambda t: ('center', sim_y_final + (15*math.sin(2*t) if "Floating" in sim_effect_mode else 0)))
         
    layers.append(sim_final)

    final = CompositeVideoClip(layers, size=(w,h)).set_audio(audio_clip)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
    return out_path

# --- UI CHÍNH ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Hình ảnh")
    sim_file = st.file_uploader("🖼️ Tải ảnh SIM (Bắt buộc):", type=['png'])
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot (Tùy chọn):", type=['png'])
    logo_file = st.file_uploader("©️ Tải Logo (Tùy chọn):", type=['png'])
    
    st.markdown("---")
    st.subheader("2. Bối cảnh")
    bg_theme = st.selectbox("Chủ đề:", 
                           ["Văn phòng hiện đại", "Ngoài trời / Thiên nhiên", 
                            "Trong nhà / Ấm cúng", "Công nghệ / Trừu tượng"])
    if st.button("🎲 TẠO BỐI CẢNH (GENERATE)"):
        if hf_token:
            with st.spinner("Đang vẽ..."):
                smart_prompt = get_smart_prompt(bg_theme)
                st.session_state['current_prompt'] = smart_prompt
                bg = generate_ai_background(smart_prompt, hf_token)
                st.session_state['generated_bg'] = bg
    
    if st.session_state['generated_bg']:
        st.image(st.session_state['generated_bg'], width=200)

with col2:
    st.subheader("3. Âm thanh")
    voice_option = st.radio("Nguồn:", 
                           ["💎 Microsoft Edge TTS (Free)", 
                            "🚀 ElevenLabs (Voice ID)", 
                            "🎙️ Tải file ghi âm"])
    
    final_audio_path = None
    input_script = ""

    if "Microsoft" in voice_option:
        voice_gender = st.selectbox("Giọng:", ["Nữ (Hoài My)", "Nam (Nam Minh)"])
        input_script = st.text_area("Kịch bản:", height=120)
    elif "ElevenLabs" in voice_option:
        voice_id_input = st.text_input("Voice ID:")
        input_script = st.text_area("Kịch bản:", height=120)
    else:
        uploaded_audio = st.file_uploader("File Audio:", type=['mp3', 'wav'])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(uploaded_audio.getvalue())
                final_audio_path = fp.name

st.markdown("---")
video_name = st.text_input("Tên file:", "dat_media_v17")

if st.button("🚀 XUẤT BẢN VIDEO", type="primary"):
    error = False
    if not sim_file: st.error("❌ Thiếu ảnh SIM (Bắt buộc)!"); error=True
    if "Tải file" not in voice_option and not input_script: st.error("❌ Thiếu kịch bản!"); error=True
    
    if not error:
        status = st.empty()
        prog = st.progress(0)
        try:
            # AUDIO
            if "Microsoft" in voice_option:
                status.text("🔊 Creating Audio...")
                final_audio_path = get_audio_from_edge(input_script, voice_gender)
            elif "ElevenLabs" in voice_option:
                if not elevenlabs_key: st.error("Thiếu Key!"); st.stop()
                status.text("🔊 Creating Audio...")
                final_audio_path = speak_with_elevenlabs(elevenlabs_key, input_script, voice_id_input)
            
            if not final_audio_path: st.stop()
            prog.progress(30)
            
            # BG CHECK
            bg_final = st.session_state['generated_bg']
            if not bg_final and hf_token:
                status.text("🎨 Generating Background...")
                smart_prompt = get_smart_prompt(bg_theme)
                bg_final = generate_ai_background(smart_prompt, hf_token)
            prog.progress(50)
            
            # IMAGES
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            # RENDER
            status.text(f"🎬 Calculating Smart Layout & Rendering...")
            out = create_video_v17(
                sim_pil, mascot_pil, logo_pil, bg_final, final_audio_path, 
                video_ratio, sim_effect_name, mascot_scale, sim_scale_factor
            )
            
            prog.progress(100); status.success("Thành công!")
            st.video(out)
            with open(out, "rb") as f: st.download_button("⬇️ Tải về", f, file_name=f"{video_name}.mp4")
            
        except Exception as e: st.error(f"Lỗi: {e}")
