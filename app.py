import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display
import telegram
from telegram.constants import ParseMode
import os
import asyncio
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import time
import sys # <--- استيراد مكتبة sys للوصول لمعلومات النظام

# ================================ الإعدادات العامة =================================
# (لا تغييرات هنا)
FONT_FILE = "Amiri-Bold.ttf"
DEFAULT_LOGO_FILE = "logo.png"
TEXT_COLOR = "#FFFFFF"
SHADOW_COLOR = "#000000"
TEXT_PLATE_COLOR = (0, 0, 0, 160)
BACKGROUND_MUSIC_VOLUME = 0.15
FPS = 30
NEWS_TEMPLATES = {
    "1": {"name": "دليلك في سوريا", "hashtag": "#عاجل #سوريا #سوريا_عاجل #syria", "color": (211, 47, 47)},
    "2": {"name": "دليلك في الأخبار", "hashtag": "#عاجل #أخبار #دليلك", "color": (200, 30, 30)},
    "3": {"name": "عاجل||نتائج", "hashtag": "#عاجل #نتائج #التعليم_الأساسي #التاسع", "color": (200, 30, 30)},
    "4": {"name": "دليلك في الرياضة", "hashtag": "#أخبار #رياضة", "color": (0, 128, 212)}
}
VIDEO_DIMENSIONS = {
    "Instagram Post (4:5)": (1080, 1350),
    "Instagram Story/Reel (9:16)": (1080, 1920),
    "Square (1:1)": (1080, 1080),
    "YouTube Standard (16:9)": (1920, 1080)
}
DETAILS_TEXT = "الـتـفـاصـيـل:"
FOOTER_TEXT = "تابعنا عبر موقع دليلك نيوز الإخباري"
# =================================================================================

# ================================ دوال مساعدة (Helper Functions) ==================
# (لا تغييرات هنا)
def add_kashida(text):
    non_connecting_chars = {'ا', 'أ', 'إ', 'آ', 'د', 'ذ', 'ر', 'ز', 'و', 'ؤ', 'ة'}
    arabic_range = ('\u0600', '\u06FF'); result = []
    text_len = len(text)
    for i, char in enumerate(text):
        result.append(char)
        if i < text_len - 1:
            next_char = text[i+1]
            is_char_arabic = arabic_range[0] <= char <= arabic_range[1]
            is_next_char_arabic = arabic_range[0] <= next_char <= arabic_range[1]
            if (is_char_arabic and is_next_char_arabic and char not in non_connecting_chars and next_char != ' '):
                result.append('ـ')
    return "".join(result)

def process_text_for_image(text): return get_display(arabic_reshaper.reshape(text))

def wrap_text_to_pages(text, font, max_width, max_lines_per_page):
    if not text: return [[]]
    lines, words, current_line = [], text.split(), ''
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if font.getbbox(process_text_for_image(test_line))[2] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line); current_line = word
    lines.append(current_line)
    return [lines[i:i + max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)]

def draw_text_with_shadow(draw, position, text, font, fill_color, shadow_color):
    x, y = position; processed_text = process_text_for_image(text); shadow_offset = 3
    draw.text((x + shadow_offset, y + shadow_offset), processed_text, font=font, fill=shadow_color, stroke_width=2)
    draw.text((x, y), processed_text, font=font, fill=fill_color)

def fit_image_to_box(img, box_width, box_height):
    img_ratio = img.width / img.height
    box_ratio = box_width / box_height
    if img_ratio > box_ratio:
        new_height = box_height; new_width = int(new_height * img_ratio)
    else:
        new_width = box_width; new_height = int(new_width / img_ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - box_width) / 2; top = (new_height - box_height) / 2
    return img.crop((left, top, left + box_width, top + box_height))
    
def render_design(design_type, draw, W, H, template, lines_to_draw, news_font, logo_img):
    if design_type == 'classic':
        header_height = int(H * 0.1)
        dark_color, light_color = template['color'], tuple(min(c+30, 255) for c in template['color'])
        for i in range(header_height):
            ratio = i / header_height; r,g,b = [int(dark_color[j]*(1-ratio) + light_color[j]*ratio) for j in range(3)]
            draw.line([(0, i), (W, i)], fill=(r,g,b))
        draw.rectangle([(0,0), (W, header_height//3)], fill=(255,255,255,50))
        header_font = ImageFont.truetype(FONT_FILE, int(W / 14.5))
        header_text_proc = process_text_for_image(template['name'])
        draw_text_with_shadow(draw, ((W - header_font.getbbox(header_text_proc)[2]) / 2, (header_height - header_font.getbbox(header_text_proc)[3]) / 2 - 10), template['name'], header_font, TEXT_COLOR, SHADOW_COLOR)
    elif design_type == 'cinematic':
        tag_font = ImageFont.truetype(FONT_FILE, int(W / 24)); tag_text = process_text_for_image(template['name'])
        tag_bbox = tag_font.getbbox(tag_text); tag_width = tag_bbox[2] - tag_bbox[0] + 60; tag_height = tag_bbox[3] - tag_bbox[1] + 30
        tag_x, tag_y = W - tag_width - 40, 40
        draw.rounded_rectangle([tag_x, tag_y, tag_x + tag_width, tag_y + tag_height], radius=tag_height/2, fill=template['color'])
        draw.text((tag_x + tag_width/2, tag_y + tag_height/2), tag_text, font=tag_font, fill=TEXT_COLOR, anchor="mm")
    
    if lines_to_draw:
        line_heights = [news_font.getbbox(process_text_for_image(line))[3] + 20 for line in lines_to_draw]
        plate_height = sum(line_heights) + 60; plate_y0 = (H - plate_height) / 2
        draw.rectangle([(0, plate_y0), (W, plate_y0 + plate_height)], fill=TEXT_PLATE_COLOR)
        text_y_start = plate_y0 + 30
        for line in lines_to_draw:
            line_width = news_font.getbbox(process_text_for_image(line))[2]
            draw_text_with_shadow(draw, ((W - line_width) / 2, text_y_start), line, news_font, TEXT_COLOR, SHADOW_COLOR)
            text_y_start += news_font.getbbox(process_text_for_image(line))[3] + 20
# =================================================================================

# ======================== الدوال الأساسية لإنشاء الفيديو ==========================
# (لا يوجد تغييرات هنا)
def create_video_frames(params, progress_bar):
    W, H = params['dimensions']
    news_title = params['text']
    template = params['template']
    background_image_path = params['image_path']
    design_type = params['design_type']
    logo_file = params['logo_path']
    status_placeholder = st.empty()
    try:
        font_size_base = int(W / 12)
        news_font = ImageFont.truetype(FONT_FILE, font_size_base if len(news_title) < 50 else font_size_base - 20)
        if background_image_path and os.path.exists(background_image_path):
            base_image_raw = Image.open(background_image_path).convert("RGB")
            base_image = fit_image_to_box(base_image_raw, W, H)
        else:
            default_bg_logo = Image.open(logo_file).convert("RGB")
            base_image = default_bg_logo.resize((W,H)).filter(ImageFilter.GaussianBlur(15))
        logo_img = Image.open(logo_file).convert("RGBA") if logo_file and os.path.exists(logo_file) else None
    except Exception as e: 
        st.error(f"خطأ في تحميل الملفات الأساسية (الخط، اللوجو، الصورة): {e}")
        return None, None
    text_pages = wrap_text_to_pages(news_title, news_font, max_width=W-120, max_lines_per_page=params['max_lines'])
    num_pages = len(text_pages)
    status_placeholder.info("⏳ جاري إنشاء الصورة المصغرة (Thumbnail)...")
    thumb_image = base_image.copy()
    render_design(design_type, ImageDraw.Draw(thumb_image, 'RGBA'), W, H, template, text_pages[0], news_font, logo_img)
    thumb_path = f"temp_thumb_{int(time.time())}.jpg"
    thumb_image.convert('RGB').save(thumb_path, quality=85)
    silent_video_path = f"temp_silent_{int(time.time())}.mp4"
    video_writer = cv2.VideoWriter(silent_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    total_main_frames = int(params['seconds_per_page'] * FPS) * num_pages
    total_video_frames = total_main_frames + int(params['outro_duration'] * FPS)
    global_frame_index = 0
    for page_index, original_page_lines in enumerate(text_pages):
        status_placeholder.info(f"⏳ جاري معالجة الصفحة {page_index + 1}/{num_pages}...")
        page_text = " ".join(original_page_lines) + (" ..." if num_pages > 1 and page_index < num_pages - 1 else "")
        words_on_page = page_text.split()
        num_words_on_page = len(words_on_page)
        for i in range(int(params['seconds_per_page'] * FPS)):
            progress_in_video = global_frame_index / total_video_frames
            zoom = 1 + progress_in_video * (params['ken_burns_zoom'] - 1)
            zoomed_w, zoomed_h = int(W * zoom), int(H * zoom)
            zoomed_bg = base_image.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)
            x_offset = (zoomed_w - W) / 2; y_offset = (zoomed_h - H) / 2
            frame_bg = zoomed_bg.crop((x_offset, y_offset, x_offset + W, y_offset + H))
            draw = ImageDraw.Draw(frame_bg, 'RGBA')
            seconds_on_page = i / FPS
            words_to_show_count = min(num_words_on_page, int(seconds_on_page * params['words_per_second']) + 1)
            lines_to_draw_now = wrap_text_to_pages(" ".join(words_on_page[:words_to_show_count]), news_font, W-120, params['max_lines'])[0]
            render_design(design_type, draw, W, H, template, lines_to_draw_now, news_font, logo_img)
            video_writer.write(cv2.cvtColor(np.array(frame_bg), cv2.COLOR_RGB2BGR))
            global_frame_index += 1
            progress_bar.progress(global_frame_index / total_video_frames)
    status_placeholder.info("⏳ جاري إضافة الخاتمة...")
    outro_frames = int(params['outro_duration'] * FPS)
    outro_font = ImageFont.truetype(FONT_FILE, int(W / 18))
    for i in range(outro_frames):
        image = Image.new('RGB', (W, H), (10, 10, 10)); draw = ImageDraw.Draw(image, 'RGBA')
        progress = i / outro_frames
        max_logo_size = int(min(W, H) / 2.5)
        current_size = int(max_logo_size * (progress ** 2))
        outro_processed = process_text_for_image(FOOTER_TEXT)
        text_width = outro_font.getbbox(outro_processed)[2]
        text_y_pos = H//2 - (current_size//2) - 50 if logo_img else H // 2
        draw_text_with_shadow(draw, ((W - text_width) / 2, text_y_pos), FOOTER_TEXT, outro_font, TEXT_COLOR, SHADOW_COLOR)
        if logo_img and current_size > 0:
            resized_logo = logo_img.resize((current_size, current_size), Image.Resampling.LANCZOS)
            logo_pos_x = (W - current_size) // 2
            logo_pos_y = H//2 - (current_size//2) + 20
            image.paste(resized_logo, (logo_pos_x, logo_pos_y), resized_logo)
        video_writer.write(cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        global_frame_index += 1
        progress_bar.progress(min(1.0, global_frame_index / total_video_frames))
    video_writer.release()
    status_placeholder.empty()
    return silent_video_path, thumb_path

# ====> التعديل التشخيصي <====
def combine_media(params, silent_video_path):
    try:
        # 1. نقوم باستيراد المكتبة
        import ffmpeg as ffmpeg_lib
        
        # 2. نقوم بطباعة معلومات تشخيصية عنها قبل استخدامها
        st.warning("--- DEBUGGING FFMPEG MODULE ---")
        # طباعة مسار الملف للمكتبة
        if hasattr(ffmpeg_lib, '__file__'):
            st.info(f"ffmpeg_lib.__file__: {ffmpeg_lib.__file__}")
        else:
            st.error("ffmpeg_lib has no __file__ attribute.")
            
        # طباعة محتويات المكتبة
        st.info("Contents of ffmpeg_lib (dir):")
        st.json(dir(ffmpeg_lib))
        st.warning("--- END DEBUGGING ---")

    except ImportError as e:
        st.error(f"CRITICAL: Failed to import ffmpeg. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during the debug phase: {e}")
        return None

    # 3. الآن نحاول تشغيل الكود الأصلي داخل try/except
    try:
        status_placeholder = st.empty()
        status_placeholder.info("⏳ جاري دمج الصوتيات ومقاطع الفيديو الإضافية...")
        
        main_video = ffmpeg_lib.input(silent_video_path)
        video_parts = []
        audio_parts = []
        
        if params['intro_path']:
            intro_clip = ffmpeg_lib.input(params['intro_path'])
            video_parts.extend([intro_clip.video])
            if 'audio' in [s['codec_type'] for s in ffmpeg_lib.probe(params['intro_path'])['streams']]:
                audio_parts.extend([intro_clip.audio])

        video_parts.append(main_video.video)

        voiceover_stream = None
        if params['voiceover_path']:
            voiceover_stream = ffmpeg_lib.input(params['voiceover_path']).audio

        music_stream = None
        if params['music_path']:
            music_stream = ffmpeg_lib.input(params['music_path'], stream_loop=-1).filter('volume', params['music_volume'])

        if voiceover_stream and music_stream:
            mixed_audio = ffmpeg_lib.filter([voiceover_stream, music_stream], 'amix', duration='first', dropout_transition=0)
            audio_parts.append(mixed_audio)
        elif voiceover_stream:
            audio_parts.append(voiceover_stream)
        elif music_stream:
            audio_parts.append(music_stream)

        if params['outro_path']:
            outro_clip = ffmpeg_lib.input(params['outro_path'])
            video_parts.append(outro_clip.video)
            if 'audio' in [s['codec_type'] for s in ffmpeg_lib.probe(params['outro_path'])['streams']]:
                audio_parts.append(outro_clip.audio)

        final_video = ffmpeg_lib.concat(*video_parts, v=1, a=0)
        
        output_video_name = f"final_video_{int(time.time())}.mp4"
        
        if audio_parts:
            final_audio = ffmpeg_lib.concat(*audio_parts, v=0, a=1)
            stream = ffmpeg_lib.output(final_video, final_audio, output_video_name, vcodec='libx264', acodec='aac', pix_fmt='yuv420p', loglevel="quiet")
        else:
            stream = ffmpeg_lib.output(final_video, output_video_name, vcodec='libx264', pix_fmt='yuv420p', loglevel="quiet")
        
        stream.overwrite_output().run()
        
        status_placeholder.empty()
        return output_video_name

    except ffmpeg_lib.Error as e:
        st.error(f"!! خطأ فادح أثناء دمج الفيديو بالصوت (ffmpeg):")
        st.code(e.stderr.decode() if e.stderr else 'Unknown Error')
        return None
    except AttributeError:
        # إذا حدث خطأ AttributeError المحدد، نعرض المعلومات التي جمعناها
        st.error("FATAL: The `AttributeError: ... has no attribute 'input'` occurred as predicted.")
        st.error("Please copy the DEBUG information above and share it for analysis.")
        return None
    finally:
        if os.path.exists(silent_video_path): os.remove(silent_video_path)
        if params.get('voiceover_path') and "temp_tts" in params['voiceover_path']: os.remove(params['voiceover_path'])


# ================================ دوال الواجهة والتطبيق ==========================
# (لا يوجد تغييرات هنا)
def login_page():
    st.title("تسجيل الدخول")
    st.write("الرجاء إدخال اسم المستخدم وكلمة المرور للوصول إلى الأداة.")
    with st.form("login_form"):
        username = st.text_input("اسم المستخدم")
        password = st.text_input("كلمة المرور", type="password")
        submitted = st.form_submit_button("دخول")
        if submitted:
            if username in st.secrets.users and st.secrets.users[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة.")

def scrape_article_page(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('h1', class_='entry-title') or soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "لم يتم العثور على عنوان"
        og_image_tag = soup.find('meta', property='og_image')
        image_url = og_image_tag['content'] if og_image_tag else None
        return {'title': title, 'image_url': image_url}
    except requests.RequestException as e:
        st.warning(f"فشل في استخلاص البيانات من الرابط: {e}")
        return None

def download_image(url, filename="temp_background.jpg"):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except requests.RequestException:
        return None

async def send_to_telegram(video_path, thumb_path, caption, hashtag):
    try:
        bot = telegram.Bot(token=st.secrets.TELEGRAM_BOT_TOKEN)
        full_caption = f"{caption}\n\n<b>{hashtag}</b>"
        with open(video_path, 'rb') as video_file, open(thumb_path, 'rb') as thumb_file:
            await bot.send_video(
                chat_id=st.secrets.TELEGRAM_CHANNEL_ID,
                video=video_file,
                thumbnail=thumb_file,
                caption=full_caption,
                parse_mode=ParseMode.HTML,
                read_timeout=180,
                write_timeout=180,
                supports_streaming=True
            )
        st.success("✅ تم نشر الفيديو بنجاح على تليجرام!")
    except Exception as e:
        st.error(f"حدث خطأ أثناء الإرسال إلى تليجرام: {e}")

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def main_app():
    st.title("🎬 أداة إنشاء الفيديو الإخباري")
    st.markdown(f"مرحباً بك **{st.session_state['username']}**! استخدم هذه الواجهة لإنشاء فيديوهات إخبارية بسهولة.")
    
    if not os.path.exists(FONT_FILE):
        st.error(f"خطأ فادح: ملف الخط '{FONT_FILE}' غير موجود. يرجى وضعه في نفس مجلد التطبيق.")
        return
    if not os.path.exists(DEFAULT_LOGO_FILE):
        st.warning(f"تنبيه: ملف اللوجو الافتراضي '{DEFAULT_LOGO_FILE}' غير موجود. يمكنك رفع لوجو مخصص.")
        
    st.header("1. إدخال المحتوى")
    input_method = st.radio("اختر طريقة الإدخال:", ("إدخال نص يدوي", "سحب البيانات من رابط"))
    news_text = ""
    news_image_path = None
    news_url = ""
    if input_method == "سحب البيانات من رابط":
        news_url = st.text_input("أدخل رابط المقال هنا:")
        if st.button("🔍 سحب البيانات"):
            if news_url:
                with st.spinner("جاري تحليل الرابط..."):
                    article_data = scrape_article_page(news_url)
                    if article_data:
                        st.session_state['news_text'] = article_data['title']
                        if article_data['image_url']:
                           st.session_state['news_image_path'] = download_image(article_data['image_url'])
                           st.success("تم سحب البيانات بنجاح!")
                    else:
                        st.error("لم يتم العثور على بيانات في الرابط.")
            else:
                st.warning("الرجاء إدخال رابط.")
    news_text = st.text_area("نص الخبر:", value=st.session_state.get('news_text', ''), height=150)
    st.write("صورة الخلفية:")
    if 'news_image_path' in st.session_state and st.session_state['news_image_path']:
        st.image(st.session_state['news_image_path'], caption="الصورة المسحوبة من الرابط", width=200)
    uploaded_background = st.file_uploader("أو قم برفع صورة خلفية مخصصة (اختياري)", type=['jpg', 'jpeg', 'png'])
    if uploaded_background:
        news_image_path = save_uploaded_file(uploaded_background)
    elif 'news_image_path' in st.session_state:
        news_image_path = st.session_state.get('news_image_path')
    st.header("2. تخصيص الفيديو")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("التصميم الأساسي")
        template_options = {v['name']: k for k, v in NEWS_TEMPLATES.items()}
        selected_template_name = st.selectbox("اختر قالب الخبر:", list(template_options.keys()))
        selected_template_key = template_options[selected_template_name]
        selected_template = NEWS_TEMPLATES[selected_template_key]
        design_type = st.selectbox("اختر نمط التصميم:", ("classic", "cinematic"), format_func=lambda x: "كلاسيكي" if x == 'classic' else "سينمائي")
        dimension_name = st.selectbox("اختر أبعاد الفيديو:", list(VIDEO_DIMENSIONS.keys()))
        W, H = VIDEO_DIMENSIONS[dimension_name]
    with col2:
        st.subheader("إعدادات الأنيميشن والتوقيت")
        seconds_per_page = st.slider("مدة عرض الصفحة (ثانية):", 1, 20, 8)
        words_per_second = st.slider("سرعة ظهور الكلمات (كلمة/ثانية):", 1, 10, 4)
        max_lines = st.slider("أقصى عدد للأسطر في الصفحة:", 1, 6, 3)
        outro_duration = st.slider("مدة الخاتمة (ثانية):", 1.0, 10.0, 6.5)
        ken_burns_zoom = st.slider("معامل تقريب Ken Burns:", 1.0, 1.2, 1.05)
    st.header("3. تخصيص الوسائط (اختياري)")
    media_col1, media_col2 = st.columns(2)
    with media_col1:
        st.subheader("الصوتيات")
        use_tts = st.checkbox("🎤 إنشاء تعليق صوتي من نص الخبر (TTS)")
        voiceover_path = None
        uploaded_music = st.file_uploader("🎵 رفع موسيقى خلفية", type=['mp3', 'wav', 'aac'])
        music_path = save_uploaded_file(uploaded_music)
        music_volume = st.slider("🔊 مستوى صوت الموسيقى:", 0.0, 1.0, 0.15, disabled=(music_path is None))
    with media_col2:
        st.subheader("الملفات الإضافية")
        uploaded_logo = st.file_uploader("🖼️ رفع لوجو مخصص (سيستخدم بدلاً من الافتراضي)", type=['png'])
        logo_path = save_uploaded_file(uploaded_logo) or (DEFAULT_LOGO_FILE if os.path.exists(DEFAULT_LOGO_FILE) else None)
        uploaded_intro = st.file_uploader("🎞️ رفع فيديو مقدمة (Intro)", type=['mp4'])
        intro_path = save_uploaded_file(uploaded_intro)
        uploaded_outro = st.file_uploader("🎞️ رفع فيديو خاتمة (Outro)", type=['mp4'])
        outro_path = save_uploaded_file(uploaded_outro)
    st.header("4. إنشاء ونشر")
    if st.button("🚀 ابدأ إنشاء الفيديو", type="primary"):
        if not news_text.strip():
            st.error("خطأ: نص الخبر فارغ! يرجى إدخال نص لإنشاء الفيديو.")
        elif not logo_path:
            st.error("خطأ: لم يتم العثور على ملف اللوجو. يرجى التأكد من وجود `logo.png` أو رفع ملف مخصص.")
        else:
            with st.spinner("التحضير لإنشاء الفيديو..."):
                if use_tts:
                    try:
                        tts_status = st.info("⏳ جاري تحويل النص إلى كلام...")
                        tts = gTTS(text=news_text, lang='ar', slow=False)
                        voiceover_path = f"temp_tts_{int(time.time())}.mp3"
                        tts.save(voiceover_path)
                        tts_status.success("✅ تم إنشاء التعليق الصوتي بنجاح.")
                    except Exception as e:
                        st.warning(f"فشل إنشاء التعليق الصوتي: {e}. سيتم إنشاء الفيديو بدون صوت.")
                        voiceover_path = None
                params = {
                    'text': news_text, 'image_path': news_image_path, 'design_type': design_type,
                    'template': selected_template, 'dimensions': (W, H), 'seconds_per_page': seconds_per_page,
                    'words_per_second': words_per_second, 'max_lines': max_lines, 'outro_duration': outro_duration,
                    'ken_burns_zoom': ken_burns_zoom, 'logo_path': logo_path, 'music_path': music_path,
                    'music_volume': music_volume, 'intro_path': intro_path, 'outro_path': outro_path,
                    'voiceover_path': voiceover_path
                }
                progress_bar = st.progress(0, "بدء عملية إنشاء الإطارات...")
                silent_video_path, thumb_path = create_video_frames(params, progress_bar)
                if silent_video_path:
                    final_video_path = combine_media(params, silent_video_path)
                    if final_video_path:
                        st.success("🎉 اكتمل إنشاء الفيديو بنجاح!")
                        st.video(final_video_path)
                        caption_parts = [news_text]
                        if news_url:
                            caption_parts.extend(["", f"<b>{DETAILS_TEXT}</b> {news_url}"])
                        final_caption = "\n".join(caption_parts)
                        if st.checkbox("نشر الفيديو على تليجرام؟", value=True):
                            if st.button("📤 إرسال إلى تليجرام"):
                                with st.spinner("جاري الإرسال..."):
                                    asyncio.run(send_to_telegram(final_video_path, thumb_path, final_caption, selected_template['hashtag']))

# ============================ نقطة بداية التطبيق ==============================

st.set_page_config(page_title="أداة إنشاء الفيديو الإخباري", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    with st.sidebar:
        st.write(f"المستخدم: **{st.session_state.get('username', '')}**")
        if st.button("تسجيل الخروج"):
            st.session_state["logged_in"] = False
            st.rerun()
    main_app()
else:
    login_page()