import streamlit as st
import os
import numpy as np
# --- ĐOẠN MÃ VÁ LỖI QUAN TRỌNG (FIX BUG PILLOW) ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# --------------------------------------------------
from PIL import Image
from rembg import remove
from moviepy.editor import *
from gtts import gTTS
from huggingface_hub import InferenceClient
import tempfile

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="DAT Media AI Studio", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; font-weight: bold; }
    .stTextInput>div>div>input { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 DAT Media - Tạo Video Quảng Cáo Tự Động")
st.markdown("---")

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cài đặt chung")
    hf_token = st.text_input("🔑 Nhập Hugging Face Token:", type="password", help="Nhập token bắt đầu bằng hf_... để dùng tính năng vẽ nền AI")
    
    st.divider()
    
    video_ratio = st.radio("Tỷ lệ khung hình:", ("16:9 (Ngang - Youtube)", "9:16 (Dọc - Tiktok/Reels)"))
    max_duration = st.slider("Thời lượng tối đa (giây):", 10, 60, 30)
    
    st.info("💡 **Mẹo:** Ảnh SIM nên chụp thẳng góc, đủ sáng để AI tách nền đẹp nhất.")

# --- HÀM HỖ TRỢ (CORE FUNCTIONS) ---

def remove_background(image):
    """Tách nền khỏi chủ thể"""
    return remove(image)

def generate_ai_background(prompt, token):
    """Vẽ nền bằng AI (Stable Diffusion XL)"""
    if not token:
        return None
    try:
        client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=token)
        image = client.text_to_image(prompt)
        return image
    except Exception as e:
        st.error(f"Lỗi tạo ảnh AI: {str(e)}")
        return None

def create_final_video(sim_img, mascot_img, bg_img, text, ratio, duration_limit):
    # 1. Thiết lập kích thước
    if ratio == "16:9 (Ngang - Youtube)":
        w, h = 1920, 1080
    else:
        w, h = 1080, 1920
        
    # 2. Xử lý Audio (Text to Speech)
    tts = gTTS(text=text, lang='vi')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name
        
    audio_clip = AudioFileClip(audio_path)
    
    # Giới hạn thời lượng
    final_duration = min(audio_clip.duration, duration_limit)
    if audio_clip.duration > final_duration:
        audio_clip = audio_clip.subclip(0, final_duration)
    
    # 3. Tạo Clip Nền (Background)
    bg_resized = bg_img.resize((w, h))
    bg_clip = ImageClip(np.array(bg_resized)).set_duration(final_duration)
    
    clips_to_overlay = [bg_clip]
    
    # 4. Xử lý SIM (Nhân vật chính)
    sim_nobg = remove_background(sim_img)
    sim_w = int(w * 0.45) 
    sim_h = int(sim_nobg.height * (sim_w / sim_nobg.width))
    sim_nobg = sim_nobg.resize((sim_w, sim_h))
    
    sim_clip = ImageClip(np.array(sim_nobg)).set_duration(final_duration)
    sim_clip = sim_clip.set_position(('center', 'center'))
    sim_clip = sim_clip.resize(lambda t: 1 + 0.02 * t) # Hiệu ứng Zoom
    
    clips_to_overlay.append(sim_clip)
    
    # 5. Xử lý Mascot (Nếu có)
    if mascot_img:
        mascot_nobg = remove_background(mascot_img)
        mascot_w = int(w * 0.3)
        mascot_h = int(mascot_nobg.height * (mascot_w / mascot_nobg.width))
        mascot_nobg = mascot_nobg.resize((mascot_w, mascot_h))
        
        mascot_clip = ImageClip(np.array(mascot_nobg)).set_duration(final_duration)
        pos = ('right', 'bottom') if ratio == "16:9 (Ngang - Youtube)" else ('center', 'bottom')
        mascot_clip = mascot_clip.set_position(pos)
        
        clips_to_overlay.append(mascot_clip)

    # 6. Xuất Video
    final_video = CompositeVideoClip(clips_to_overlay, size=(w,h))
    final_video = final_video.set_audio(audio_clip)
    
    # Sử dụng tempfile để tránh lỗi quyền ghi file trên Cloud
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        output_filename = tmp_video.name
        final_video.write_videofile(output_filename, fps=24, codec='libx264', audio_codec='aac')
    
    return output_filename

# --- GIAO DIỆN CHÍNH (MAIN UI) ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Tài nguyên Hình ảnh")
    sim_file = st.file_uploader("Tải ảnh SIM/Sản phẩm (Bắt buộc)", type=['png', 'jpg', 'jpeg'])
    if sim_file:
        st.image(sim_file, width=200)

    mascot_file = st.file_uploader("Tải ảnh Linh vật (Tùy chọn)", type=['png', 'jpg', 'jpeg'])
    if mascot_file:
        st.image(mascot_file, width=150)

with col2:
    st.subheader("2. Bối cảnh & Nội dung")
    bg_prompt = st.text_area("Mô tả bối cảnh để AI vẽ (Tiếng Anh tốt hơn):", 
                             value="futuristic technology background, neon lights, 8k resolution, blue and purple theme",
                             height=100)
    
    script_text = st.text_area("Kịch bản lời thoại (Tiếng Việt):", 
                               value="Chào bạn, đây là SIM Data 4G tốc độ cao từ DAT Media. Lướt web thả ga, không lo về giá!",
                               height=100)

# --- KHU VỰC XỬ LÝ ---
st.markdown("---")
if st.button("🚀 BẮT ĐẦU TẠO VIDEO (START)"):
    if not hf_token:
        st.error("⚠️ Vui lòng nhập Hugging Face Token ở thanh bên trái (Sidebar)!")
    elif not sim_file:
        st.error("⚠️ Vui lòng tải ảnh SIM lên!")
    else:
        step_progress = st.progress(0)
        status_text = st.empty()
        
        try:
            # B1: Load ảnh
            status_text.text("⏳ Đang xử lý hình ảnh đầu vào...")
            sim_img_pil = Image.open(sim_file).convert("RGBA")
            mascot_img_pil = Image.open(mascot_file).convert("RGBA") if mascot_file else None
            step_progress.progress(20)
            
            # B2: Tạo nền AI
            status_text.text("🎨 AI đang vẽ bối cảnh (Mất khoảng 10-20s)...")
            generated_bg = generate_ai_background(bg_prompt, hf_token)
            
            if generated_bg is None:
                st.error("Không tạo được nền. Kiểm tra lại Token!")
            else:
                step_progress.progress(50)
                st.image(generated_bg, caption="Bối cảnh do AI vừa vẽ", width=400)
                
                # B3: Render Video
                status_text.text("🎬 Đang dựng video và lồng tiếng...")
                video_path = create_final_video(
                    sim_img_pil, 
                    mascot_img_pil, 
                    generated_bg, 
                    script_text, 
                    video_ratio, 
                    max_duration
                )
                step_progress.progress(100)
                status_text.text("✅ Hoàn tất!")
                
                # Hiển thị và Tải về
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.video(video_path)
                with col_res2:
                    st.success("Video của bạn đã sẵn sàng!")
                    with open(video_path, "rb") as file:
                        st.download_button(
                            label="⬇️ TẢI VIDEO VỀ MÁY",
                            data=file,
                            file_name="DAT_Media_Video.mp4",
                            mime="video/mp4"
                        )
                        
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
