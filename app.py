import streamlit as st
import os
import numpy as np
# --- VÁ LỖI TƯƠNG THÍCH PILLOW ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# ---------------------------------
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
from moviepy.editor import *
from gtts import gTTS
from huggingface_hub import InferenceClient
import tempfile
import math
import requests

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media V6 - Subtitles", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; padding: 10px 0; }
    div[data-testid="stButton"] > button:first-child { background-color: #f0f2f6; color: black; border: 1px solid #ccc; }
    div[data-testid="stVerticalBlock"] > div:last-child > div > button { background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media V6 - Mascot Center & Subtitles")
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
    mascot_scale = st.slider("Độ lớn Mascot:", 0.3, 1.0, 0.7, help="Chỉnh độ to nhỏ của Mascot")
    st.divider()
    # Tùy chọn phụ đề
    use_subtitle = st.checkbox("Hiển thị phụ đề (Subtitle)", value=True)
    subtitle_color = st.color_picker("Màu chữ phụ đề:", "#FFFF00") # Vàng mặc định

# --- HÀM HỖ TRỢ HỆ THỐNG ---

# 1. Tải Font tiếng Việt (Tránh lỗi ô vuông)
def download_font():
    font_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    font_path = "Roboto-Bold.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except: pass
    return font_path

# 2. Tạo hình ảnh chứa Text (Thay thế TextClip của MoviePy hay lỗi)
def create_text_image(text, w, h, fontsize=40, color="yellow"):
    # Tạo ảnh nền trong suốt
    img = Image.new('RGBA', (w, int(h/5)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load font
    font_path = download_font()
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except:
        font = ImageFont.load_default()
        
    # Tính vị trí giữa
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
    except:
        # Fallback cho phiên bản Pillow cũ hơn
        text_w, text_h = draw.textsize(text, font=font)
        
    x = (w - text_w) / 2
    y = (int(h/5) - text_h) / 2
    
    # Vẽ viền đen cho chữ nổi
    outline_range = 3
    for dx in range(-outline_range, outline_range+1):
        for dy in range(-outline_range, outline_range+1):
            draw.text((x+dx, y+dy), text, font=font, fill="black")
            
    # Vẽ chữ chính
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img)

# 3. Tạo nền AI
def generate_ai_background(prompt, token, seed=0):
    if not token: return None
    final_prompt = f"{prompt}, highly detailed, 8k, cinematic lighting, vivid colors"
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(final_prompt)
    except: return None

# 4. CORE: Xử lý Video
def create_video_v6(sim_img, mascot_img, bg_img, audio_path, script_text, ratio, duration, scale, show_sub, sub_color):
    # Setup kích thước
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    
    # Xử lý Audio
    audio_clip = AudioFileClip(audio_path)
    final_duration = min(audio_clip.duration, duration)
    if audio_clip.duration > final_duration:
        audio_clip = audio_clip.subclip(0, final_duration)
        
    layers = []
    
    # Lớp 1: Background
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
    else:
        bg_clip = ColorClip(size=(w, h), color=(20,20,30)).set_duration(final_duration)
    layers.append(bg_clip)

    # Lớp 2: Mascot & Sim (CENTER STAGE)
    if mascot_img:
        mascot_nobg = remove(mascot_img)
        
        # Tăng kích thước Mascot lên (dựa vào biến scale từ slider)
        m_w = int(w * scale) 
        m_h = int(mascot_nobg.height * (m_w / mascot_nobg.width))
        mascot_resized = mascot_nobg.resize((m_w, m_h))
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)
        
        # Vị trí: Đứng giữa màn hình (Center)
        # Tính toán để Mascot đứng ở khoảng 2/3 màn hình từ trên xuống
        center_y = h * 0.6  # Hạ thấp trọng tâm xuống chút cho đẹp
        
        # Hiệu ứng "Idle Breathing" (Thở & Trôi)
        # Kết hợp Zoom nhẹ (thở) + Di chuyển lên xuống (trôi)
        mascot_anim = (mascot_clip
                       .set_position(lambda t: ('center', center_y - m_h/2 + 10 * math.sin(2*t))) # Trôi lên xuống
                       .resize(lambda t: 1 + 0.02 * math.sin(3*t)) # Phồng xẹp nhẹ
                       )
        layers.append(mascot_anim)

        # Sim: Đặt ngay trước ngực Mascot
        s_w = int(m_w * 0.4) # Sim nhỏ bằng 40% Mascot
        s_h = int(sim_img.height * (s_w / sim_img.width))
        sim_resized = sim_img.resize((s_w, s_h))
        sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
        
        # Sim chuyển động đồng bộ với Mascot
        # Vị trí sim = Vị trí mascot + offset
        sim_base_y = center_y + m_h * 0.1 # Đặt ở phần bụng/ngực
        
        sim_anim = (sim_clip
                    .set_position(lambda t: ('center', sim_base_y + 10 * math.sin(2*t))) # Trôi cùng mascot
                    .rotate(lambda t: 5 * math.sin(3*t)) # Lắc lư thêm chút cho vui
                    )
        layers.append(sim_anim)

    else:
        # Nếu không có Mascot thì để SIM giữa màn hình to đùng
        s_w = int(w * 0.6)
        s_h = int(sim_img.height * (s_w / sim_img.width))
        sim_resized = sim_img.resize((s_w, s_h))
        sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
        
        sim_anim = (sim_clip
                    .set_position('center')
                    .resize(lambda t: 1 + 0.05 * math.sin(t)) # Zoom in out
                    )
        layers.append(sim_anim)

    # Lớp 3: Phụ đề (Subtitles) - Giả lập Karaoke
    if show_sub and script_text:
        # Chia kịch bản thành các câu nhỏ (mỗi câu khoảng 5-6 từ)
        words = script_text.split()
        chunk_size = 6
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        if len(chunks) > 0:
            # Thời gian mỗi câu hiển thị
            chunk_duration = final_duration / len(chunks)
            
            sub_clips = []
            for i, chunk in enumerate(chunks):
                # Tạo ảnh chứa text bằng Pillow (An toàn hơn TextClip)
                txt_img = create_text_image(chunk, w, h, fontsize=50 if "9:16" in ratio else 40, color=sub_color)
                
                txt_clip = (ImageClip(txt_img)
                            .set_start(i * chunk_duration)
                            .set_duration(chunk_duration)
                            .set_position(('center', 'bottom' if "16:9" in ratio else 0.85), relative=True)) # 0.85 là gần đáy
                
                # Hiệu ứng chữ nảy lên (Pop up)
                txt_clip = txt_clip.resize(lambda t: 1 + 0.1 * math.sin(t*10) if t < 0.2 else 1)
                
                sub_clips.append(txt_clip)
                
            layers.extend(sub_clips)

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
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot (Toàn thân):", type=['png', 'jpg'])
    
with col2:
    st.subheader("2. Nội dung & Âm thanh")
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
        input_script = st.text_area("Nhập kịch bản (Để tạo giọng & Phụ đề):", height=100)
    else:
        uploaded_audio = st.file_uploader("Tải file MP3/WAV:", type=['mp3', 'wav'])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(uploaded_audio.getvalue())
                final_audio = fp.name
        # Dù tải file, vẫn cần nhập text để làm phụ đề
        input_script = st.text_area("Nhập lại nội dung file ghi âm (Để làm phụ đề):", height=100)

# Render Button
st.markdown("---")
video_name = st.text_input("Tên video:", "video_dat_media_v6")

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
        
        # --- ĐOẠN NÀY LÀ CHỖ ĐÃ SỬA LỖI INDENTATION (THỤT ĐẦU DÒNG) ---
        try:
            # 1. Tạo Audio nếu cần
            if voice_type == "📝 AI Đọc":
                status.text("🔊 Đang tạo giọng đọc...")
                tts = gTTS(input_script, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    final_audio = fp.name
            
            prog.progress(20)
            
            # 2. Check background
            bg_final = st.session_state['generated_bg']
            if not bg_final:
                status.text("🎨 Đang vẽ bối cảnh...")
                bg_final = generate_ai_background(bg_prompt, hf_token)
                st.session_state['generated_bg'] = bg_final
            
            prog.progress(40)
            
            # 3. Load Images
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            
            # 4. Render
            status.text("🎬 Đang xử lý Video & Phụ đề...")
            out_vid = create_video_v6(
                sim_pil, mascot_pil, bg_final, final_audio, 
                input_script, video_ratio, video_duration, 
                mascot_scale, use_subtitle, subtitle_color
            )
            
            prog.progress(100)
            status.success("Xong!")
            st.video(out_vid)
            
            with open(out_vid, "rb") as f:
                st.download_button("⬇️ Tải về", f, file_name=f"{video_name}.mp4", mime="video/mp4")
                
        except Exception as e:
            st.error(f"Lỗi: {e}")
