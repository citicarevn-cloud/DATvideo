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
st.set_page_config(page_title="DAT Media V14 - Pro", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media V14 - Pro Animation & Smart BG")
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
    mascot_scale = st.slider("Mascot Zoom:", 0.3, 1.0, 0.75)
    
    # 1. DANH SÁCH HIỆU ỨNG SIM (7 Loại)
    sim_effect_name = st.selectbox(
        "Hiệu ứng chuyển động SIM:",
        [
            "1. Lơ lửng (Floating) - Mặc định",
            "2. Nảy tưng tưng (Bounce)",
            "3. Lắc lư qua lại (Swing)",
            "4. Phóng to thu nhỏ (Pulse)",
            "5. Xoay tròn 3D (Spin 3D)",
            "6. Trượt ngang (Slide)",
            "7. Rung lắc mạnh (Shake)"
        ]
    )

# --- HÀM TẠO PROMPT THÔNG MINH ---
def get_smart_prompt(theme):
    # Từ khóa ngẫu nhiên để tạo sự khác biệt mỗi lần bấm
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
    
    else: # Mặc định
        return f"abstract background, professional, {detail}"

def generate_ai_background(prompt, token):
    if not token: return None
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(prompt)
    except: return None

# --- HÀM AUDIO ---
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

# --- HÀM XỬ LÝ HIỆU ỨNG SIM ---
def apply_sim_transform(clip, effect_name, w, h, center_pos):
    cx, cy = center_pos # Vị trí trung tâm (trước ngực Mascot)
    
    if "Floating" in effect_name: # Lơ lửng nhẹ
        return clip.set_position(lambda t: (cx, cy + 10*math.sin(2*t))).rotate(lambda t: 2*math.sin(t))
    
    elif "Bounce" in effect_name: # Nảy tưng tưng
        return clip.set_position(lambda t: (cx, cy + abs(30*math.sin(3*t)) - 15))
    
    elif "Swing" in effect_name: # Lắc qua lại như đồng hồ
        # Xoay quanh tâm phía trên của ảnh (cần logic phức tạp hơn, ở đây xoay tâm giữa)
        return clip.rotate(lambda t: 15 * math.sin(3*t)).set_position((cx, cy))
        
    elif "Pulse" in effect_name: # Phóng to thu nhỏ
        return clip.resize(lambda t: 1 + 0.05 * math.sin(4*t)).set_position('center').set_position(lambda t: (cx, cy))
        
    elif "Spin 3D" in effect_name: # Giả lập xoay 3D (bằng cách co giãn chiều ngang)
        # MoviePy cơ bản khó làm 3D thật, dùng hiệu ứng lật qua lại
        return clip.resize(lambda t: (abs(math.cos(2*t)) + 0.1, 1)).set_position('center').set_position((cx, cy))
        
    elif "Slide" in effect_name: # Trượt ngang qua lại
        return clip.set_position(lambda t: (cx + 50*math.sin(2*t), cy))
        
    elif "Shake" in effect_name: # Rung lắc mạnh (báo động)
        return clip.set_position(lambda t: (cx + 5*math.sin(20*t), cy + 5*math.cos(15*t)))
        
    else:
        return clip.set_position((cx, cy))

# --- HÀM VIDEO CORE ---
def create_video_v14(sim_img, mascot_img, logo_img, bg_img, audio_path, ratio, sim_effect_mode, scale):
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    
    # 1. Xử lý Audio & Thời lượng
    # Video sẽ dài bằng chính xác file audio
    audio_clip = AudioFileClip(audio_path)
    final_duration = audio_clip.duration + 1 # Cộng thêm 1s dư ra cho đẹp
    
    layers = []
    
    # 2. Background
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
        layers.append(bg_clip)
    else:
        layers.append(ColorClip(size=(w, h), color=(20,20,30)).set_duration(final_duration))

    # Tọa độ chuẩn
    center_y = h * 0.6 # Mascot đứng thấp hơn giữa chút
    
    # 3. Mascot (Idle Animation - Thở nhẹ)
    if mascot_img:
        m_w = int(w * scale)
        m_h = int(mascot_img.height * (m_w / mascot_img.width))
        mascot_resized = mascot_img.resize((m_w, m_h))
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)
        
        # Hiệu ứng thở: Phồng nhẹ + Trôi lên xuống cực nhẹ
        mascot_anim = (mascot_clip
                       .set_position(lambda t: ('center', center_y - m_h/2 + 5 * math.sin(1.5*t)))
                       .resize(lambda t: 1 + 0.01 * math.sin(2*t)))
        layers.append(mascot_anim)
        
        # Sim Base Position: Trước ngực Mascot
        sim_base_x = (w - int(m_w * 0.45)) / 2 # Căn giữa theo chiều ngang
        sim_base_y = center_y + m_h * 0.15 # Vị trí bụng/ngực
    else:
        sim_base_x = (w - int(w*0.6)) / 2
        sim_base_y = h/2 - int(w*0.6)/2

    # 4. SIM (Áp dụng hiệu ứng đã chọn)
    sim_ratio = 0.45 if mascot_img else 0.6
    s_w = int((w * scale * sim_ratio) if mascot_img else w * sim_ratio)
    s_h = int(sim_img.height * (s_w / sim_img.width))
    sim_resized = sim_img.resize((s_w, s_h))
    
    sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
    
    # Gọi hàm xử lý chuyển động SIM
    # Lưu ý: sim_base_x được tính toán để căn giữa, nhưng set_position('center') của moviepy đôi khi xung đột với lambda
    # Nên ta dùng vị trí tương đối
    sim_final = apply_sim_transform(sim_clip, sim_effect_mode, w, h, (sim_base_x, sim_base_y))
    
    # Do hàm transform trả về clip với pos function, ta cần căn lại center X nếu hàm không tự căn
    if "Slide" not in sim_effect_mode:
        # Ép căn giữa trục X cho các hiệu ứng không di chuyển ngang
        sim_final = sim_final.set_position(lambda t: ('center', sim_base_y + (10*math.sin(2*t) if "Floating" in sim_effect_mode else 0)))
        # (Logic trên là giản lược, trong thực tế ta tin tưởng hàm apply_sim_transform)
        
    layers.append(sim_final)

    # 5. Logo
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
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot (1 ảnh tĩnh):", type=['png'])
    logo_file = st.file_uploader("©️ Tải Logo:", type=['png'])
    
    st.markdown("---")
    st.subheader("2. Bối cảnh (Smart Generator)")
    
    # Dropdown Menu chọn chủ đề
    bg_theme = st.selectbox("Chọn chủ đề bối cảnh:", 
                           ["Văn phòng hiện đại", "Ngoài trời / Thiên nhiên", 
                            "Trong nhà / Ấm cúng", "Công nghệ / Trừu tượng"])
    
    if st.button("🎲 TẠO BỐI CẢNH MỚI (GENERATE)"):
        if hf_token:
            with st.spinner(f"AI đang vẽ bối cảnh {bg_theme}..."):
                # Tạo prompt mới hoàn toàn mỗi lần bấm
                smart_prompt = get_smart_prompt(bg_theme)
                st.session_state['current_prompt'] = smart_prompt # Lưu để debug xem chơi
                
                # Gọi AI vẽ
                bg = generate_ai_background(smart_prompt, hf_token)
                st.session_state['generated_bg'] = bg
    
    if st.session_state['generated_bg']:
        st.image(st.session_state['generated_bg'], width=250, caption="Bối cảnh vừa tạo")
        st.caption(f"Prompt: {st.session_state['current_prompt']}")

with col2:
    st.subheader("3. Âm thanh")
    voice_option = st.radio("Nguồn âm thanh:", 
                           ["💎 Microsoft Edge TTS (Free)", 
                            "🚀 ElevenLabs (Cần Voice ID)", 
                            "🎙️ Tải file ghi âm của tôi"])
    
    final_audio_path = None
    input_script = ""

    if "Microsoft" in voice_option:
        voice_gender = st.selectbox("Giọng đọc:", ["Nữ (Hoài My)", "Nam (Nam Minh)"])
        input_script = st.text_area("Nhập kịch bản quảng cáo:", height=150)
        
    elif "ElevenLabs" in voice_option:
        voice_id_input = st.text_input("Nhập Voice ID:", help="Lấy từ ElevenLabs -> Voices")
        input_script = st.text_area("Nhập kịch bản quảng cáo:", height=150)
        
    else: # Tải file
        uploaded_audio = st.file_uploader("Tải file MP3/WAV:", type=['mp3', 'wav'])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(uploaded_audio.getvalue())
                final_audio_path = fp.name
            st.success(f"Đã nhận file âm thanh! Video sẽ dài theo file này.")

st.markdown("---")
video_name = st.text_input("Tên file:", "dat_media_final")

if st.button("🚀 XUẤT BẢN VIDEO", type="primary"):
    error = False
    if not sim_file: st.error("Thiếu ảnh SIM!"); error=True
    if "Tải file" not in voice_option and not input_script: st.error("Thiếu kịch bản!"); error=True
    
    if not error:
        status = st.empty()
        prog = st.progress(0)
        try:
            # 1. AUDIO GENERATION
            if "Microsoft" in voice_option:
                status.text("🔊 Đang tạo giọng Microsoft...")
                final_audio_path = get_audio_from_edge(input_script, voice_gender)
            
            elif "ElevenLabs" in voice_option:
                if not elevenlabs_key: st.error("Chưa nhập API Key!"); st.stop()
                status.text("🔊 Đang tạo giọng ElevenLabs...")
                final_audio_path = speak_with_elevenlabs(elevenlabs_key, input_script, voice_id_input)
            
            if not final_audio_path: st.stop()
            prog.progress(30)
            
            # 2. BACKGROUND CHECK
            bg_final = st.session_state['generated_bg']
            if not bg_final and hf_token:
                status.text("🎨 Đang vẽ bối cảnh lần đầu...")
                smart_prompt = get_smart_prompt(bg_theme)
                bg_final = generate_ai_background(smart_prompt, hf_token)
            
            prog.progress(50)
            
            # 3. LOAD IMAGES
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            # 4. RENDER
            status.text(f"🎬 Đang xử lý hiệu ứng: {sim_effect_name}...")
            out = create_video_v14(
                sim_pil, mascot_pil, logo_pil, bg_final, final_audio_path, 
                video_ratio, sim_effect_name, mascot_scale
            )
            
            prog.progress(100); status.success("Xong!")
            st.video(out)
            with open(out, "rb") as f: st.download_button("⬇️ Tải về", f, file_name=f"{video_name}.mp4")
            
        except Exception as e: st.error(f"Lỗi: {e}")
