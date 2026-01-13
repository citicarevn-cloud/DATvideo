import streamlit as st
import os
import numpy as np
# --- VÁ LỖI TƯƠNG THÍCH PILLOW (GIỮ NGUYÊN) ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# ----------------------------------------------
from PIL import Image
from rembg import remove
from moviepy.editor import *
from gtts import gTTS
from huggingface_hub import InferenceClient
import tempfile
import math

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media Studio Pro V4", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0068C9; color: white; font-weight: bold; padding: 10px 0; }
    .stTextInput>div>div>input { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media - Video Creator (Branding Version)")
st.markdown("---")

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình Kỹ thuật")
    hf_token = st.text_input("🔑 Hugging Face Token:", type="password")
    
    st.divider()
    
    video_ratio = st.radio("Tỷ lệ khung hình:", ("9:16 (Dọc - Tiktok/Reels)", "16:9 (Ngang - Youtube)"))
    video_duration = st.slider("Thời lượng video (giây):", 10, 60, 20)
    
    st.divider()
    
    st.subheader("✨ Hiệu ứng SIM")
    effect_type = st.selectbox(
        "Chọn kiểu chuyển động:",
        ["Lắc lư (Shake)", "Trượt qua lại (Slide)", "Zoom nhẹ (Zoom In)", "Nhịp đập (Pulse)", "Đứng yên (Static)"]
    )

# --- HÀM HỖ TRỢ LOGIC ---

def get_dominant_color_hex(pil_img):
    """Lấy màu chủ đạo của Logo để AI vẽ nền đồng bộ"""
    img = pil_img.copy()
    img = img.convert("RGBA")
    img = img.resize((50, 50)) # Thu nhỏ để xử lý nhanh
    pixels = img.getcolors(50 * 50)
    if not pixels: return None
    
    # Sắp xếp màu xuất hiện nhiều nhất (bỏ qua màu trong suốt/trắng/đen)
    sorted_pixels = sorted(pixels, key=lambda t: t[0], reverse=True)
    
    for count, color in sorted_pixels:
        # color là (r, g, b, a)
        if len(color) == 4 and color[3] < 200: continue # Bỏ qua màu trong suốt
        if sum(color[:3]) > 700 or sum(color[:3]) < 50: continue # Bỏ qua trắng/đen quá
        
        # Chuyển sang Hex
        return '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    
    return None # Không tìm được màu đặc trưng

def generate_ai_background(prompt, token, color_hex=None):
    if not token: return None
    
    final_prompt = prompt
    # Nếu có mã màu logo, ép AI vẽ theo tông màu đó
    if color_hex:
        final_prompt = f"background theme color {color_hex}, {prompt}"
        
    print(f"Prompt gửi đi: {final_prompt}") # Debug log
    
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        return client.text_to_image(final_prompt)
    except: return None

def apply_effect(clip, effect_name, w, h):
    # Các hàm hiệu ứng giữ nguyên như V3
    if effect_name == "Zoom nhẹ (Zoom In)":
        return clip.resize(lambda t: 1 + 0.05 * t).set_position('center')
    elif effect_name == "Lắc lư (Shake)":
        return clip.rotate(lambda t: 5 * math.sin(2 * math.pi * t)).set_position('center')
    elif effect_name == "Trượt qua lại (Slide)":
        center_x = w / 2 - clip.w / 2
        center_y = h / 2 - clip.h / 2
        return clip.set_position(lambda t: (center_x + 40 * math.sin(t*2), center_y))
    elif effect_name == "Nhịp đập (Pulse)":
        return clip.resize(lambda t: 1 + 0.03 * math.sin(t*3)).set_position('center')
    else:
        return clip.set_position('center')

def create_video(sim_img, mascot_img, logo_img, bg_img, audio_path, effect, ratio, duration):
    # 1. Thiết lập kích thước
    w, h = (1080, 1920) if "9:16" in ratio else (1920, 1080)
    
    # 2. Xử lý Audio
    audio_clip = AudioFileClip(audio_path)
    final_duration = min(audio_clip.duration, duration)
    if audio_clip.duration > final_duration:
        audio_clip = audio_clip.subclip(0, final_duration)
    
    # 3. Tạo nền
    if bg_img:
        bg_resized = bg_img.resize((w, h))
        bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
    else:
        bg_clip = ColorClip(size=(w, h), color=(10,10,10)).set_duration(final_duration)
        
    layers = [bg_clip]
    
    # 4. Mascot (Lớp dưới)
    if mascot_img:
        mascot_nobg = remove(mascot_img)
        m_w = int(w * 0.35)
        m_h = int(mascot_nobg.height * (m_w / mascot_nobg.width))
        mascot_resized = mascot_nobg.resize((m_w, m_h))
        mascot_clip = ImageClip(np.array(mascot_resized)).set_duration(final_duration)
        pos = ('center', 'bottom') if "9:16" in ratio else ('right', 'bottom')
        mascot_clip = mascot_clip.set_position(pos)
        layers.append(mascot_clip)

    # 5. SIM (Nhân vật chính - Giữ nguyên ảnh gốc)
    s_w = int(w * 0.55) # Sim chiếm 55%
    s_h = int(sim_img.height * (s_w / sim_img.width))
    sim_resized = sim_img.resize((s_w, s_h))
    sim_clip = ImageClip(np.array(sim_resized)).set_duration(final_duration)
    sim_anim = apply_effect(sim_clip, effect, w, h)
    layers.append(sim_anim)
    
    # 6. LOGO (Góc trái trên - Giữ nguyên ảnh gốc)
    if logo_img:
        l_w = int(w * 0.15) # Logo chiếm 15% chiều rộng video
        l_h = int(logo_img.height * (l_w / logo_img.width))
        logo_resized = logo_img.resize((l_w, l_h))
        
        logo_clip = ImageClip(np.array(logo_resized)).set_duration(final_duration)
        # Vị trí: Cách lề trái 30px, lề trên 30px
        logo_clip = logo_clip.set_position((30, 30))
        layers.append(logo_clip)
    
    # 7. Xuất file
    final = CompositeVideoClip(layers, size=(w,h)).set_audio(audio_clip)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
        
    return out_path

# --- UI CHÍNH ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Hình ảnh & Thương hiệu")
    
    # SIM (Bắt buộc)
    sim_file = st.file_uploader("🖼️ Tải ảnh SIM (PNG đã tách nền):", type=['png'])
    if sim_file:
        st.caption("✅ Đã nhận ảnh SIM")
        
    # Mascot (Tùy chọn)
    mascot_file = st.file_uploader("🦖 Tải ảnh Mascot (Tùy chọn):", type=['png', 'jpg'])
    
    # LOGO (Mới)
    logo_file = st.file_uploader("©️ Tải Logo (Sẽ đặt góc trái trên):", type=['png', 'jpg'])
    logo_color_hint = None
    if logo_file:
        # Xem trước logo và lấy màu
        logo_pil_preview = Image.open(logo_file)
        st.image(logo_pil_preview, width=100, caption="Logo")
        logo_color_hint = get_dominant_color_hex(logo_pil_preview)
        if logo_color_hint:
            st.caption(f"🎨 Phát hiện tông màu Logo: {logo_color_hint}. AI sẽ vẽ nền theo màu này.")

with col2:
    st.subheader("2. Âm thanh & Nội dung")
    
    # Lựa chọn nguồn âm thanh (Radio Button)
    voice_option = st.radio("Chọn nguồn giọng đọc:", 
                            ["📝 AI Đọc (Nhập kịch bản)", "🎙️ Tải file ghi âm (MP3/WAV)"])
    
    final_audio_path = None
    script_content = ""
    
    if voice_option == "📝 AI Đọc (Nhập kịch bản)":
        script_content = st.text_area("Nhập nội dung quảng cáo:", height=150, 
                                      placeholder="Ví dụ: Chào các bạn, sim data này siêu rẻ...")
        if script_content:
             # Logic xử lý TTS sau khi bấm nút start để tiết kiệm
             pass
    else:
        uploaded_audio = st.file_uploader("Tải file giọng đọc lên:", type=['mp3', 'wav'])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                fp.write(uploaded_audio.getvalue())
                final_audio_path = fp.name
                
    st.markdown("---")
    bg_prompt = st.text_input("Mô tả bối cảnh nền (Tiếng Anh):", 
                              value="abstract technology background, bokeh lights, 8k, 3d render")

# Đặt tên video
st.markdown("---")
video_name_input = st.text_input("3. Đặt tên file video:", "video_quang_cao_dat_media")

# --- NÚT XỬ LÝ TRUNG TÂM ---
if st.button("🚀 XUẤT BẢN VIDEO NGAY", type="primary"):
    
    # Kiểm tra lỗi đầu vào
    error_msg = ""
    if not hf_token: error_msg = "⚠️ Chưa nhập Hugging Face Token!"
    elif not sim_file: error_msg = "⚠️ Chưa tải ảnh SIM!"
    elif voice_option == "📝 AI Đọc (Nhập kịch bản)" and not script_content.strip():
        error_msg = "⚠️ Bạn chọn AI đọc nhưng chưa nhập kịch bản!"
    elif voice_option == "🎙️ Tải file ghi âm (MP3/WAV)" and not final_audio_path:
        error_msg = "⚠️ Bạn chọn tải file ghi âm nhưng chưa tải file nào lên!"
        
    if error_msg:
        st.error(error_msg)
    else:
        # Bắt đầu xử lý
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Xử lý Audio (Nếu là AI đọc thì giờ mới tạo file)
            if voice_option == "📝 AI Đọc (Nhập kịch bản)":
                status_text.text("🔊 Đang tạo giọng đọc AI...")
                tts = gTTS(script_content, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    final_audio_path = fp.name
            
            progress_bar.progress(20)
            
            # 2. Load ảnh
            sim_pil = Image.open(sim_file).convert("RGBA")
            mascot_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            logo_pil = Image.open(logo_file).convert("RGBA") if logo_file else None
            
            # 3. Tạo nền AI (Có tính đến màu logo)
            status_text.text("🎨 AI đang vẽ bối cảnh theo thương hiệu...")
            # Lấy màu logo nếu chưa có
            if logo_pil and not logo_color_hint:
                logo_color_hint = get_dominant_color_hex(logo_pil)
                
            bg_img = generate_ai_background(bg_prompt, hf_token, logo_color_hint)
            progress_bar.progress(50)
            
            # 4. Render Video
            status_text.text("🎬 Đang dựng video và ghép hiệu ứng...")
            out_video = create_video(
                sim_pil, mascot_pil, logo_pil, bg_img, 
                final_audio_path, effect_type, 
                video_ratio, video_duration
            )
            
            progress_bar.progress(100)
            status_text.success("✅ Hoàn tất! Video đã sẵn sàng.")
            
            # Hiển thị
            st.video(out_video)
            
            # Nút tải
            with open(out_video, "rb") as f:
                st.download_button(
                    label="⬇️ TẢI VIDEO VỀ MÁY",
                    data=f,
                    file_name=f"{video_name_input}.mp4",
                    mime="video/mp4"
                )
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra trong quá trình xử lý: {e}")
