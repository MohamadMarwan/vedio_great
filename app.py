# ==============================================================================
#     أداة إنشاء ونشر الفيديو الإخباري الاحترافي (إصدار 10.0 - الهوية العربية)
#     - إضافة تحويل الأرقام إلى الصيغة العربية (١, ٢, ٣).
#     - تعريب واجهة Streamlit عبر CSS.
#     - عكس تصميم القالب "الديناميكي" ليناسب القراءة من اليمين لليسار (RTL).
#     - تغيير الخط الافتراضي إلى خط "Tajawal" العصري.
# ==============================================================================
import os
import random
import cv2
import numpy as np
import ffmpeg
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import time
import streamlit as st
from urllib.parse import urlparse
from gtts import gTTS

# ==============================================================================
#                                 إعدادات الواجهة
# ==============================================================================
st.set_page_config(page_title="أداة إنشاء الفيديو الإخباري", layout="wide", initial_sidebar_state="expanded")

# --- إضافة: حقن CSS لتعريب أجزاء من الواجهة وإخفاء النصوص الإنجليزية ---
st.markdown("""
<style>
    /* إخفاء النص الإنجليزي الافتراضي لشريط التقدم */
    .stProgress > div > div > div > div {
        color: transparent;
    }
    /* إخفاء النص الإنجليزي الافتراضي للمؤشر الدوار */
    .stSpinner > div > div {
        color: transparent;
    }
    /* محاذاة كل شيء لليمين ليعطي إحساساً عربياً أكثر */
    body {
        direction: rtl;
    }
    /* إعادة محاذاة بعض العناصر التي قد تتأثر سلباً */
    .stButton>button {
        direction: ltr; /* للحفاظ على الأيقونة والنص بترتيب صحيح */
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 أداة إنشاء ونشر الفيديو الإخباري الاحترافي")
st.markdown("v10.0 - الهوية العربية الكاملة | تصميم RTL | خطوط عصرية")

# إنشاء مجلدات ضرورية
if not os.path.exists("uploads"): os.makedirs("uploads")
if not os.path.exists("temp_media"): os.makedirs("temp_media")

# ==============================================================================
#                             الدوال المساعدة
# ==============================================================================

# --- إضافة: دالة لتحويل الأرقام الغربية إلى عربية ---
def convert_numbers_to_arabic(text):
    """يحول الأرقام الغربية في النص إلى أرقام هندية/عربية."""
    mapping = str.maketrans("1234567890", "١٢٣٤٥٦٧٨٩٠")
    return text.translate(mapping)

# --- تعديل: دمج تحويل الأرقام في دالة معالجة النص ---
def process_text(text):
    processed = convert_numbers_to_arabic(str(text))
    reshaped = arabic_reshaper.reshape(processed)
    return get_display(reshaped)

def ease_in_out_quad(t): return 2*t*t if t<0.5 else 1-pow(-2*t+2,2)/2
def draw_text(draw, pos, text, font, fill, shadow_color, offset=(2,2)):
    proc_text=process_text(text)
    draw.text((pos[0]+offset[0],pos[1]+offset[1]),proc_text,font=font,fill=shadow_color)
    draw.text(pos,proc_text,font=font,fill=fill)

def fit_image_to_frame(img, target_w, target_h, frame_idx, total_frames):
    img_w, img_h = img.size; target_aspect = target_w / target_h; img_aspect = img_w / img_h
    if img_aspect > target_aspect: new_h = target_h; new_w = int(new_h * img_aspect)
    else: new_w = target_w; new_h = int(new_w / img_aspect)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    zoom_factor=1.20; progress=frame_idx/total_frames; current_zoom=1+(zoom_factor-1)*ease_in_out_quad(progress)
    zoomed_w,zoomed_h=int(target_w*current_zoom),int(target_h*current_zoom)
    zoom_img = img.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)
    x_offset=(zoomed_w-target_w)/2; y_offset=(zoomed_h-target_h)*progress
    return zoom_img.crop((x_offset,y_offset,x_offset+target_w,y_offset+target_h))

def wrap_text(text, font, max_width):
    lines,words=[],text.split(); current_line=""
    for word in words:
        if font.getbbox(process_text(f"{current_line} {word}"))[2]<=max_width: current_line=f"{current_line} {word}".strip()
        else: lines.append(current_line); current_line=word
    lines.append(current_line); return [l for l in lines if l]

def save_uploaded_file(uploaded_file, folder="uploads"):
    if uploaded_file is not None:
        path = os.path.join(folder, uploaded_file.name);
        with open(path, "wb") as f: f.write(uploaded_file.getbuffer());
        return path
    return None

@st.cache_data(ttl=3600)
def generate_tts_audio(text, lang='ar'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        path = f"temp_media/tts_audio_{random.randint(1000, 9999)}.mp3"
        tts.save(path); return path
    except Exception as e:
        st.error(f"!! فشل في إنشاء التعليق الصوتي: {e}"); return None

# ==============================================================================
#                        دوال سحب البيانات والنشر
# ==============================================================================
@st.cache_data(ttl=600)
def scrape_article_data(url):
    st.info(f"🔍 جاري تحليل الرابط: {url}")
    try:
        headers={'User-Agent':'Mozilla/5.0'}; res=requests.get(url,headers=headers,timeout=15); res.raise_for_status()
        soup=BeautifulSoup(res.content,'html.parser')
        title_tag=soup.find('h1') or soup.find('meta',property='og:title')
        title=title_tag.get_text(strip=True) if hasattr(title_tag,'get_text') else title_tag.get('content','')
        content_div=soup.find('div',class_='entry-content') or soup.find('article')
        content=" ".join([p.get_text(strip=True) for p in (content_div or soup).find_all('p')])
        image_urls=set()
        og_image=soup.find('meta',property='og:image')
        if og_image: image_urls.add(og_image['content'])
        for img_tag in (content_div or soup).find_all('img',limit=5):
            src=img_tag.get('src') or img_tag.get('data-src')
            if src and src.startswith('http'): image_urls.add(src)
        st.success(f"✅ تم العثور على العنوان و {len(image_urls)} رابط صورة.")
        return {'title':title,'content':content,'image_urls':list(image_urls)}
    except Exception as e: st.error(f"!! خطأ في تحليل الرابط: {e}"); return None

def download_images(urls):
    paths=[]
    for i,url in enumerate(urls[:4]):
        try:
            res=requests.get(url,stream=True,timeout=15); res.raise_for_status()
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or f"temp_img_{random.randint(1000,9999)}.jpg"
            path=os.path.join("temp_media", filename)
            with open(path,'wb') as f: f.write(res.content)
            paths.append(path)
        except Exception as e: st.warning(f"  !! فشل تنزيل الصورة: {url[:50]}...")
    return paths

def send_video_to_telegram(video_path, thumb_path, caption, token, channel_id):
    st.info("--> جاري نشر الفيديو إلى تليجرام...")
    try:
        url = f"https://api.telegram.org/bot{token}/sendVideo"
        with open(video_path, 'rb') as video_file, open(thumb_path, 'rb') as thumb_file:
            payload={'chat_id':channel_id,'caption':caption,'parse_mode':'HTML','supports_streaming':True}
            files={'video':video_file,'thumb':thumb_file}
            response = requests.post(url, data=payload, files=files, timeout=1800)
            if response.status_code == 200:
                st.balloons(); st.success("✅ تم النشر بنجاح!")
                return True
            else:
                st.error(f"!! فشل النشر: {response.status_code} - {response.text}")
                return False
    except requests.exceptions.RequestException as e:
        st.error(f"!! خطأ فادح أثناء الاتصال بتليجرام: {e}"); return False

# ==============================================================================
#             محرك رسم التصاميم الاحترافية (جميع دوال render)
# ==============================================================================
def draw_text_word_by_word(draw, box_coords, lines, words_to_show, font, fill, shadow):
    x, y, w, h = box_coords; line_height = font.getbbox("ا")[3] + 20
    total_text_height = len(lines) * line_height; current_y = y + (h - total_text_height) / 2
    words_shown = 0
    for line in lines:
        words_in_line = line.split(); words_to_draw_in_line = []
        for word in words_in_line:
            if words_shown < words_to_show: words_to_draw_in_line.append(word); words_shown += 1
            else: break
        if words_to_draw_in_line:
            partial_line = " ".join(words_to_draw_in_line); processed_partial_line = process_text(partial_line)
            line_width = font.getbbox(processed_partial_line)[2]
            draw_text(draw, (x + w - line_width, current_y), partial_line, font, fill, shadow)
        current_y += line_height
        if words_shown >= words_to_show: break

# --- تعديل: عكس تصميم القالب الديناميكي ليناسب RTL ---
def render_dynamic_split_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = Image.new('RGB', (W, H), (15, 15, 15))
    if H > W: # الوضع العمودي (يبقى كما هو)
        img_h, text_h = H // 2, H // 2; img_resized = image.resize((W, img_h), Image.Resampling.LANCZOS); frame.paste(img_resized, (0, 0)); text_box = (20, img_h + 20, W - 40, text_h - 40)
    else: # الوضع الأفقي (تم عكسه ليناسب RTL)
        img_w = W // 2
        img_resized = image.resize((img_w, H), Image.Resampling.LANCZOS)
        frame.paste(img_resized, (W - img_w, 0)) # الصق الصورة على اليمين

        grad = Image.new('RGBA', (img_w, H), (0, 0, 0, 0)); g_draw = ImageDraw.Draw(grad)
        for j in range(img_w // 2): # ارسم التدرج من اليسار لليمين
            g_draw.line([(j, 0), (j, H)], fill=(0, 0, 0, int(255 * (j / (img_w // 2)))), width=1)
        frame.paste(grad, (W - img_w, 0), grad) # الصق التدرج فوق حافة الصورة اليسرى

        text_box = (50, 0, (W // 2) - 100, H) # ضع صندوق النص على اليسار

    draw = ImageDraw.Draw(frame); text_font = ImageFont.truetype(FONT_FILE, settings['font_size'])
    total_words = len(" ".join(text_lines).split()); words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    draw_text_word_by_word(draw, text_box, text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_cinematic_overlay_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame(image, W, H, frame_idx, total_frames)
    draw = ImageDraw.Draw(frame, 'RGBA'); text_font = ImageFont.truetype(FONT_FILE, settings['font_size'])
    line_height = text_font.getbbox("ا")[3] + 20; plate_height = min((len(text_lines) * line_height) + 60, H // 2)
    draw.rectangle([(0, H - plate_height), (W, H)], fill=(0, 0, 0, 180)); total_words = len(" ".join(text_lines).split())
    words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    draw_text_word_by_word(draw, (40, H - plate_height, W - 80, plate_height), text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_modern_grid_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame(image, W, H, frame_idx, total_frames).point(lambda p: p * 0.5)
    draw = ImageDraw.Draw(frame, 'RGBA'); padding = 80 if W > H else 40
    draw.rectangle([(padding, padding), (W - padding, H - padding)], outline=settings['cat']['color'], width=5)
    draw.rectangle([(padding + 10, padding + 10), (W - padding - 10, H - padding - 10)], fill=(0, 0, 0, 190))
    text_font = ImageFont.truetype(FONT_FILE, settings['font_size']); total_words = len(" ".join(text_lines).split())
    words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    box = (padding + 40, padding + 40, W - 2 * (padding + 40), H - 2 * (padding + 40))
    draw_text_word_by_word(draw, box, text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_news_ticker_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame(image, W, H, frame_idx, total_frames)
    draw = ImageDraw.Draw(frame, 'RGBA'); font_size = int(H * 0.05)
    ticker_font = ImageFont.truetype(FONT_FILE, font_size); bar_height = int(font_size * 1.8)
    draw.rectangle([(0, H - bar_height), (W, H)], fill=(0, 0, 0, 190)); cat_bar_width = W // 4
    draw.rectangle([(W - cat_bar_width, H - bar_height), (W, H)], fill=settings['cat']['color'])
    cat_font_size = int(font_size * 0.8); cat_font = ImageFont.truetype(FONT_FILE, cat_font_size)
    cat_text = process_text(settings['cat']['name']); cat_w, cat_h = cat_font.getbbox(cat_text)[2], cat_font.getbbox(cat_text)[3]
    draw_text(draw, (W - (cat_bar_width + cat_w) // 2, H - bar_height + (bar_height - cat_h) // 2 - 5), settings['cat']['name'], cat_font, '#FFFFFF', (0, 0, 0, 128))
    full_text = " ".join(text_lines) + "   ***   "; full_text_processed = process_text(full_text)
    text_width = ticker_font.getbbox(full_text_processed)[2]; progress = frame_idx / total_frames
    total_scroll_dist = (W * 0.7) + text_width; start_pos = W; current_x = start_pos - (total_scroll_dist * progress)
    draw_text(draw, (current_x, H - bar_height + (bar_height - font_size) // 2 - 10), full_text, ticker_font, settings['text_color'], settings['shadow_color'])
    draw_text(draw, (current_x + text_width, H - bar_height + (bar_height - font_size) // 2 - 10), full_text, ticker_font, settings['text_color'], settings['shadow_color']); return frame

def render_title_scene(writer, duration, text, image_path, settings):
    W, H = settings['dimensions']; FPS = 30; frames = int(duration * FPS); img = Image.open(image_path).convert("RGB")
    title_font = ImageFont.truetype(FONT_FILE, int(W / 12 if H > W else W / 18)); cat_font = ImageFont.truetype(FONT_FILE, int(W / 20 if H > W else W / 35))
    cat = settings['cat']
    for i in range(frames):
        frame = fit_image_to_frame(img, W, H, i, frames); draw = ImageDraw.Draw(frame, 'RGBA')
        draw.rectangle([(0, H * 0.6), (W, H)], fill=(0, 0, 0, 180)); cat_bbox = cat_font.getbbox(process_text(cat['name']))
        draw_text(draw, (W - cat_bbox[2] - 40, H * 0.65), cat['name'], cat_font, cat['color'], (0, 0, 0, 150))
        wrapped_lines = wrap_text(text, title_font, W - 80); y = H * 0.72
        for line in wrapped_lines:
            bbox = title_font.getbbox(process_text(line))
            draw_text(draw, (W - bbox[2] - 40, y), line, title_font, settings['text_color'], settings['shadow_color']); y += bbox[3] * 1.3
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

def render_source_outro_scene(writer, duration, logo_path, settings):
    W, H = settings['dimensions']; FPS = 30; frames = int(duration * FPS)
    font_big = ImageFont.truetype(FONT_FILE, int(W / 20 if H > W else W / 28)); font_small = ImageFont.truetype(FONT_FILE, int(W / 30 if H > W else W / 45))
    logo = Image.open(logo_path).convert("RGBA") if logo_path and os.path.exists(logo_path) else None
    text1 = "للتفاصيل الكاملة ومتابعة آخر الأخبار"; text2 = "قــم بــزيــارة مــوقــعــنــا"
    for i in range(frames):
        progress = i / frames; frame = Image.new('RGB', (W, H), (10, 10, 10)); draw = ImageDraw.Draw(frame, 'RGBA')
        if logo:
            size = int((settings['logo_size']) * ease_in_out_quad(progress))
            if size > 0: l = logo.resize((size, size), Image.Resampling.LANCZOS); frame.paste(l, ((W - size) // 2, H // 2 - size - 20), l)
        if progress > 0.2:
            text_progress = (progress - 0.2) / 0.8; alpha = int(255 * text_progress)
            def hex_to_rgba(hex_color, alpha_val):
                h = hex_color.lstrip('#')
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha_val,)
            
            text_color_rgba = hex_to_rgba(settings['text_color'], alpha)
            shadow_color_rgba = hex_to_rgba(settings['shadow_color'], int(alpha * 0.8))

            y_pos = H // 2 + 50; bbox1 = font_big.getbbox(process_text(text1))
            draw_text(draw, ((W - bbox1[2]) / 2, y_pos), text1, font_big, text_color_rgba, shadow_color_rgba)
            bbox2 = font_small.getbbox(process_text(text2))
            draw_text(draw, ((W - bbox2[2]) / 2, y_pos + 100), text2, font_small, text_color_rgba, shadow_color_rgba)
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

# ==============================================================================
#                      الوظيفة الرئيسية لإنشاء الفيديو
# ==============================================================================
def create_story_video(article_data, image_paths, settings):
    W, H = settings['dimensions']; FPS = 30
    if not image_paths: st.error("!! خطأ فادح: لا توجد صور لإنشاء الفيديو."); return None, None
    progress_bar = st.progress(0, text="بدء عملية إنشاء الفيديو...")
    render_function = {"ديناميكي": render_dynamic_split_scene, "سينمائي": render_cinematic_overlay_scene, "عصري": render_modern_grid_scene, "شريط إخباري": render_news_ticker_scene}[settings['design_choice']]
    scenes=[]; current_duration = settings['intro_duration'] + settings['outro_duration']
    content_sentences=[s.strip() for s in article_data.get('content','').split('.') if len(s.strip())>20]
    available_images=image_paths[1:] if len(image_paths)>1 else list(image_paths); current_text_chunk=""
    for sentence in content_sentences:
        current_text_chunk += sentence + ". "; words_in_chunk = len(current_text_chunk.split()); estimated_scene_duration = max(settings['min_scene_duration'], words_in_chunk / 2.5)
        if current_duration + estimated_scene_duration > settings['max_video_duration']: st.warning(f"⚠️ تم الوصول للحد الأقصى للمدة."); break
        if words_in_chunk > 30 and available_images:
            img_scene = available_images.pop(0); scenes.append({'duration': estimated_scene_duration, 'text': current_text_chunk, 'image': img_scene})
            current_duration += estimated_scene_duration; current_text_chunk = ""
            if not available_images: available_images = list(image_paths)
    if not scenes and content_sentences:
        text = " ".join(content_sentences); duration = max(settings['min_scene_duration'], len(text.split()) / 2.5)
        scenes.append({'duration': duration, 'text': text, 'image': image_paths[0]})
    temp_videos = []
    if settings.get('intro_video'):
        progress_bar.progress(5, text="إعداد مقدمة الفيديو..."); resized_intro = f"temp_media/resized_intro.mp4"
        (ffmpeg.input(settings['intro_video']).filter('scale', W, H).output(resized_intro, r=FPS).overwrite_output().run(quiet=True)); temp_videos.append(resized_intro)
    silent_content_path = f"temp_media/silent_content_{random.randint(1000,9999)}.mp4"
    writer = cv2.VideoWriter(silent_content_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    progress_bar.progress(10, text="🎬 تصيير مشهد العنوان..."); render_title_scene(writer, settings['intro_duration'], article_data['title'], image_paths[0], settings)
    sfx_times=[settings['intro_duration']]
    for i, scene in enumerate(scenes):
        progress_bar.progress(20 + int(60 * (i / len(scenes))), text=f"-> تصيير مشهد نصي ({i+1}/{len(scenes)})...")
        frames_scene=int(scene['duration']*FPS); image=Image.open(scene['image']).convert("RGB"); text_font=ImageFont.truetype(FONT_FILE, settings['font_size'])
        max_w = (W//2 - 120) if settings['design_choice'] == "ديناميكي" and W > H else (W - 160)
        text_lines=wrap_text(scene['text'],text_font,max_w)
        for j in range(frames_scene): frame=render_function(j,frames_scene,text_lines,image,settings); writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        sfx_times.append(sfx_times[-1]+scene['duration'])
    progress_bar.progress(80, text="🎬 تصيير مشهد الخاتمة..."); render_source_outro_scene(writer, settings['outro_duration'], settings['logo_file'], settings); writer.release(); temp_videos.append(silent_content_path)
    if settings.get('outro_video'):
        progress_bar.progress(85, text="إعداد خاتمة الفيديو..."); resized_outro = f"temp_media/resized_outro.mp4"
        (ffmpeg.input(settings['outro_video']).filter('scale', W, H).output(resized_outro, r=FPS).overwrite_output().run(quiet=True)); temp_videos.append(resized_outro)
    progress_bar.progress(90, text="🔄 دمج مقاطع الفيديو..."); final_silent_video_path = f"temp_media/final_silent_{random.randint(1000,9999)}.mp4"
    concat_list_path = os.path.join("temp_media", "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for v in temp_videos: f.write(f"file '{os.path.basename(v)}'\n")
    # تم تصحيح المسار ليكون مطلقًا لتجنب أي مشاكل
    (ffmpeg.input(concat_list_path, format='concat', safe=0, r=FPS).output(final_silent_video_path, c='copy').overwrite_output().run(quiet=True))
    progress_bar.progress(95, text="🔊 دمج الصوتيات...");
    try:
        total_duration = float(ffmpeg.probe(final_silent_video_path)['format']['duration']); vid_stream=ffmpeg.input(final_silent_video_path); audio_inputs=[]
        tts_path = settings.get('tts_audio_path')
        if tts_path: audio_inputs.append(ffmpeg.input(tts_path, t=total_duration).filter('volume', settings['tts_volume']))
        music_files = settings.get('music_files', [])
        if music_files:
            music_vol = settings['music_volume'] * 0.4 if tts_path else settings['music_volume']
            music_stream = ffmpeg.input(random.choice(music_files),stream_loop=-1,t=total_duration).filter('volume',music_vol).filter('afade',type='out',start_time=total_duration-3,duration=3); audio_inputs.append(music_stream)
        sfx_file = settings.get('sfx_file')
        if sfx_file and sfx_times:
            sfx_streams = ffmpeg.input(sfx_file).filter('volume', settings['sfx_volume']).asplit(len(sfx_times))
            for i, time_s in enumerate(sfx_times): audio_inputs.append(sfx_streams[i].filter('adelay',f'{int(time_s*1000)}ms|{int(time_s*1000)}ms'))
        output_video_name = "final_news_story.mp4"
        if audio_inputs:
            mixed_audio=ffmpeg.filter(audio_inputs,'amix',duration='longest',inputs=len(audio_inputs))
            ffmpeg.output(vid_stream, mixed_audio, output_video_name, vcodec='libx264', acodec='aac', pix_fmt='yuv420p', preset='fast', crf=28, audio_bitrate='96k').overwrite_output().run(quiet=True)
        else: ffmpeg.output(vid_stream, output_video_name, vcodec='copy').run(quiet=True)
    except ffmpeg.Error as e: st.error(f"!! خطأ FFMPEG: {e.stderr.decode()}"); return None,None
    progress_bar.progress(98, text="🖼️ إنشاء الصورة المصغرة..."); thumbnail_name = "thumbnail.jpg"
    thumb=Image.open(image_paths[0]).convert("RGB").resize((W,H)); draw_t=ImageDraw.Draw(thumb,'RGBA')
    draw_t.rectangle([(0,0),(W,H)],fill=(0,0,0,100)); font_t=ImageFont.truetype(FONT_FILE,int(W/10 if H>W else W/15))
    lines=wrap_text(article_data['title'],font_t,W-100); y=H/2-(len(lines)*120)/2
    for line in lines: draw_text(draw_t,((W-font_t.getbbox(process_text(line))[2])/2,y),line,font_t,settings['text_color'],settings['shadow_color']); y+=120
    thumb.save(thumbnail_name,'JPEG',quality=85); progress_bar.progress(100, text="✅ اكتمل الإنشاء!"); return output_video_name, thumbnail_name

# ==============================================================================
#                      واجهة المستخدم الرسومية (Streamlit)
# ==============================================================================
with st.sidebar:
    st.header("🎬 إعدادات الفيديو الأساسية")
    aspect_ratio_option = st.selectbox("اختر أبعاد الفيديو (النسبة):", ("16:9 (يوتيوب، أفقي)", "9:16 (تيك توك، عمودي)"), key="aspect_ratio")
    dimensions = (1920, 1080) if "16:9" in aspect_ratio_option else (1080, 1920)
    st.header("📂 الملفات")
    logo_file_uploaded = st.file_uploader("ارفع شعار القناة (PNG)", type=["png"])
    logo_file_path = save_uploaded_file(logo_file_uploaded) or ("logo.png" if os.path.exists("logo.png") else None)
    
    # --- تعديل: تغيير الخط الافتراضي إلى Tajawal ---
    # ملاحظة: يجب عليك تحميل ملف الخط Tajawal-Bold.ttf ووضعه في نفس المجلد
    font_file_uploaded = st.file_uploader("ارفع ملف الخط (TTF)", type=["ttf"])
    FONT_FILE = save_uploaded_file(font_file_uploaded) or "Tajawal-Bold.ttf"

    intro_video_uploaded = st.file_uploader("ارفع فيديو المقدمة (اختياري)", type=["mp4"])
    intro_video_path = save_uploaded_file(intro_video_uploaded, "temp_media")
    outro_video_uploaded = st.file_uploader("ارفع فيديو الخاتمة (اختياري)", type=["mp4"])
    outro_video_path = save_uploaded_file(outro_video_uploaded, "temp_media")
    st.header("🎵 الصوتيات")
    enable_tts = st.checkbox("📢 تفعيل التعليق الصوتي الآلي (TTS)")
    tts_volume = st.slider("مستوى صوت التعليق الصوتي (TTS)", 0.0, 2.0, 1.0, 0.1, disabled=not enable_tts)
    music_files_uploaded = st.file_uploader("ارفع ملفات الموسيقى الخلفية (MP3)", type=["mp3"], accept_multiple_files=True)
    music_files_paths = [save_uploaded_file(f) for f in music_files_uploaded]
    sfx_file_uploaded = st.file_uploader("ارفع المؤثر الصوتي للانتقالات (MP3)", type=["mp3"])
    sfx_file_path = save_uploaded_file(sfx_file_uploaded)
    st.header("🔒 إعدادات النشر والأمان")
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", ""); TELEGRAM_CHANNEL_ID = st.secrets.get("TELEGRAM_CHANNEL_ID", "")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID: st.warning("لم يتم العثور على معلومات تليجرام في ملف الأمان.")
    else: st.success("تم تحميل إعدادات تليجرام.")

tab1, tab2 = st.tabs(["📝 المحتوى والتخصيص", "🖼️ رفع صور يدوية"])
with tab1:
    st.subheader("1. اختر مصدر المحتوى"); content_choice = st.radio("اختر طريقة الإدخال:", ('إدخال رابط مقال', 'إدخال نص يدوي'), horizontal=True, key="content_choice")
    items_to_process = []
    if content_choice == 'إدخال رابط مقال':
        url = st.text_input("🔗 أدخل رابط المقال هنا:");
        if url: items_to_process.append({'type':'url','value':url})
    else:
        title = st.text_input("✍️  أدخل عنوان الخبر:"); content = st.text_area("📄 أدخل نص الخبر (اختياري):")
        if title: items_to_process.append({'type':'text', 'title': title, 'content': content})
with tab2:
    st.subheader("🖼️ رفع صور مخصصة للمشاهد"); st.info("هذه الصور ستُدمج مع الصور المسحوبة من الرابط (إن وجد).")
    manual_images_uploaded = st.file_uploader("ارفع صورة واحدة أو أكثر", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    manual_image_paths = [save_uploaded_file(img, "temp_media") for img in manual_images_uploaded]
    if manual_image_paths: st.image(manual_image_paths, width=150)

st.divider(); st.subheader("2. تخصيص تصميم الفيديو")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("🎨 **التصميم والألوان**")
        design_choice = st.selectbox("اختر التصميم البصري:", ("ديناميكي", "سينمائي", "عصري", "شريط إخباري"))
        NEWS_CATEGORIES={"1":{"name":"عاجل","color":"#D32F2F"}, "2":{"name":"أخبار","color":"#0080D4"}, "3":{"name":"رياضة","color":"#4CAF50"}}
        cat_name = st.selectbox("اختر فئة الخبر:", [v['name'] for v in NEWS_CATEGORIES.values()])
        cat_choice_key = [k for k, v in NEWS_CATEGORIES.items() if v['name'] == cat_name][0]
        cat_color = st.color_picker("اختر لون الفئة:", NEWS_CATEGORIES[cat_choice_key]['color']); final_cat = {'name': cat_name, 'color': cat_color}
        text_color = st.color_picker('لون النص الأساسي', '#FFFFFF'); shadow_color = st.color_picker('لون ظل النص', '#000000')
with col2:
     with st.container(border=True):
        st.markdown("⏱️ **المدة والأبعاد**")
        max_video_duration = st.slider("المدة القصوى للفيديو (ثانية)", 20, 180, 60); min_scene_duration = st.slider("الحد الأدنى لمدة المشهد (ثانية)", 2.0, 10.0, 4.5, 0.5)
        intro_duration = st.slider("مدة مشهد العنوان (ثانية)", 2.0, 10.0, 4.0, 0.5); outro_duration = st.slider("مدة مشهد الخاتمة (ثانية)", 3.0, 15.0, 7.0, 0.5)
col3, col4 = st.columns(2)
with col3:
    with st.container(border=True):
        st.markdown("⚙️ **إعدادات متقدمة**")
        font_size = st.slider("حجم الخط في المشاهد", 30, 120, int(dimensions[0]/28))
        logo_size_outro = st.slider("حجم الشعار في الخاتمة (بكسل)", 100, 800, int(dimensions[0]/4.5))
with col4:
    with st.container(border=True):
        st.markdown("🔊 **مستويات الصوت**")
        music_volume = st.slider("مستوى صوت الموسيقى", 0.0, 1.0, 0.1, 0.05); sfx_volume = st.slider("مستوى صوت المؤثرات", 0.0, 1.0, 0.4, 0.05)

st.divider()
if st.button("🚀 **ابدأ إنشاء الفيديو والنشر**", type="primary", use_container_width=True):
    if not items_to_process:
        st.error("الرجاء إدخال محتوى أولاً (رابط أو نص يدوي).")
    elif not os.path.exists(FONT_FILE):
        st.error(f"ملف الخط '{FONT_FILE}' غير موجود! يرجى تحميله ووضعه بجانب ملف التشغيل.")
    else:
        tts_audio_path = None; item = items_to_process[0]; article_data, image_paths, source_url = None, [], None
        with st.spinner('⏳ جاري تحضير البيانات...'):
            if item['type'] == 'url':
                source_url = item['value']; scraped_data = scrape_article_data(source_url)
                if scraped_data: article_data = scraped_data; image_paths.extend(download_images(article_data.get('image_urls', [])))
            elif item['type'] == 'text': article_data = {'title': item['title'], 'content': item['content']}
            image_paths.extend(manual_image_paths)
            image_paths = sorted(set(image_paths), key=image_paths.index)
            if not image_paths:
                if logo_file_path and os.path.exists(logo_file_path):
                    st.warning("لم يتم العثور على صور، سيتم استخدام الشعار كصورة افتراضية."); image_paths = [logo_file_path]
        if enable_tts and article_data:
            with st.spinner("⏳ جاري إنشاء التعليق الصوتي..."):
                full_text_for_tts = article_data['title'] + ". " + article_data.get('content', '')
                tts_audio_path = generate_tts_audio(full_text_for_tts)
                if tts_audio_path: st.success("✅ تم إنشاء التعليق الصوتي.")
        settings = {
            'dimensions': dimensions, 'tts_audio_path': tts_audio_path, 'tts_volume': tts_volume, 'logo_file': logo_file_path,
            'intro_video': intro_video_path, 'outro_video': outro_video_path, 'music_files': music_files_paths, 'sfx_file': sfx_file_path,
            'design_choice': design_choice, 'cat': final_cat, 'text_color': text_color, 'shadow_color': shadow_color,
            'max_video_duration': max_video_duration, 'min_scene_duration': min_scene_duration, 'intro_duration': intro_duration, 'outro_duration': outro_duration,
            'font_size': font_size, 'logo_size': logo_size_outro, 'music_volume': music_volume, 'sfx_volume': sfx_volume,
        }
        if article_data and image_paths:
            video_file, thumb_file = create_story_video(article_data, image_paths, settings)
            if video_file and thumb_file:
                st.success(f"✅ نجاح! تم إنشاء الفيديو '{video_file}'."); st.video(video_file); st.image(thumb_file)
                caption=[f"<b>{article_data['title']}</b>",""]
                if source_url: caption.append(f"🔗 <b>المصدر:</b> {source_url}")
                send_video_to_telegram(video_file, thumb_file, "\n".join(caption), st.secrets["TELEGRAM_BOT_TOKEN"], st.secrets["TELEGRAM_CHANNEL_ID"])
                time.sleep(2)
                for f in os.listdir("temp_media"):
                    try: os.remove(os.path.join("temp_media", f))
                    except OSError as e: st.warning(f"لم يتمكن من حذف الملف المؤقت: {f} - {e}")
            else: st.error("❌ فشلت عملية إنشاء الفيديو لهذا العنصر.")
        else: st.error("!! فشل في تحضير البيانات أو الصور للفيديو.")