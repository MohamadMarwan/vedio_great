# ==============================================================================
#     أداة إنشاء ونشر الفيديو الإخباري الاحترافي (الإصدار 11.0)
#     - إضافة ميزة "مجموعة العلامة التجارية" (Brand Kit)
#     - إضافة "المعالجة المجمعة" (Batch Processing)
#     - إضافة اختيار لهجات التعليق الصوتي (TTS)
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
import json
import shutil

# ==============================================================================
#                                 إعدادات الواجهة
# ==============================================================================
st.set_page_config(page_title="منصة إنتاج الفيديو الإخباري", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 منصة إنتاج الفيديو الإخباري الآلية")
st.markdown("v11.0 - المعالجة المجمعة | مجموعات العلامة التجارية | لهجات متعددة")

# إنشاء مجلدات ضرورية
if not os.path.exists("uploads"): os.makedirs("uploads")
if not os.path.exists("temp_media"): os.makedirs("temp_media")
if not os.path.exists("brand_kits"): os.makedirs("brand_kits")

# ==============================================================================
#                             الدوال المساعدة
# ==============================================================================
def ease_in_out_quad(t): return 2*t*t if t<0.5 else 1-pow(-2*t+2,2)/2
def process_text(text): return get_display(arabic_reshaper.reshape(text))
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
def generate_tts_audio(text, lang='ar', tld='com'):
    try:
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        path = f"temp_media/tts_audio_{random.randint(1000, 9999)}.mp3"
        tts.save(path); return path
    except Exception as e:
        st.error(f"!! فشل في إنشاء التعليق الصوتي: {e}"); return None

# ==============================================================================
#                        دوال سحب البيانات والنشر
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def scrape_article_data(url):
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
        return {'title':title,'content':content,'image_urls':list(image_urls)}
    except Exception:
        return None

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
        except Exception:
            pass # Fail silently in batch mode
    return paths

def send_video_to_telegram(video_path, thumb_path, caption, token, channel_id):
    try:
        url = f"https://api.telegram.org/bot{token}/sendVideo"
        with open(video_path, 'rb') as video_file, open(thumb_path, 'rb') as thumb_file:
            payload={'chat_id':channel_id,'caption':caption,'parse_mode':'HTML','supports_streaming':True}
            files={'video':video_file,'thumb':thumb_file}
            response = requests.post(url, data=payload, files=files, timeout=1800)
            if response.status_code == 200:
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

def render_dynamic_split_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = Image.new('RGB', (W, H), (15, 15, 15))
    if H > W: img_h, text_h = H // 2, H // 2; img_resized = image.resize((W, img_h), Image.Resampling.LANCZOS); frame.paste(img_resized, (0, 0)); text_box = (20, img_h + 20, W - 40, text_h - 40)
    else:
        img_w = W // 2; img_resized = image.resize((img_w, H), Image.Resampling.LANCZOS); frame.paste(img_resized, (0, 0))
        grad = Image.new('RGBA', (img_w, H), (0, 0, 0, 0)); g_draw = ImageDraw.Draw(grad)
        for j in range(img_w // 2): g_draw.line([(img_w - j, 0), (img_w - j, H)], fill=(0, 0, 0, int(255 * (j / (img_w // 2)))), width=1)
        frame.paste(grad, (0, 0), grad); text_box = (img_w + 50, 0, (W // 2) - 100, H)
    draw = ImageDraw.Draw(frame); text_font = ImageFont.truetype(settings['font_file'], settings['font_size'])
    total_words = len(" ".join(text_lines).split()); words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    draw_text_word_by_word(draw, text_box, text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_cinematic_overlay_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame(image, W, H, frame_idx, total_frames)
    draw = ImageDraw.Draw(frame, 'RGBA'); text_font = ImageFont.truetype(settings['font_file'], settings['font_size'])
    line_height = text_font.getbbox("ا")[3] + 20; plate_height = min((len(text_lines) * line_height) + 60, H // 2)
    draw.rectangle([(0, H - plate_height), (W, H)], fill=(0, 0, 0, 180)); total_words = len(" ".join(text_lines).split())
    words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    draw_text_word_by_word(draw, (40, H - plate_height, W - 80, plate_height), text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_modern_grid_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame(image, W, H, frame_idx, total_frames).point(lambda p: p * 0.5)
    draw = ImageDraw.Draw(frame, 'RGBA'); padding = 80 if W > H else 40
    draw.rectangle([(padding, padding), (W - padding, H - padding)], outline=settings['cat']['color'], width=5)
    draw.rectangle([(padding + 10, padding + 10), (W - padding - 10, H - padding - 10)], fill=(0, 0, 0, 190))
    text_font = ImageFont.truetype(settings['font_file'], settings['font_size']); total_words = len(" ".join(text_lines).split())
    words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    box = (padding + 40, padding + 40, W - 2 * (padding + 40), H - 2 * (padding + 40))
    draw_text_word_by_word(draw, box, text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_news_ticker_scene(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame(image, W, H, frame_idx, total_frames)
    draw = ImageDraw.Draw(frame, 'RGBA'); font_size = int(H * 0.05)
    ticker_font = ImageFont.truetype(settings['font_file'], font_size); bar_height = int(font_size * 1.8)
    draw.rectangle([(0, H - bar_height), (W, H)], fill=(0, 0, 0, 190)); cat_bar_width = W // 4
    draw.rectangle([(W - cat_bar_width, H - bar_height), (W, H)], fill=settings['cat']['color'])
    cat_font_size = int(font_size * 0.8); cat_font = ImageFont.truetype(settings['font_file'], cat_font_size)
    cat_text = process_text(settings['cat']['name']); cat_w, cat_h = cat_font.getbbox(cat_text)[2], cat_font.getbbox(cat_text)[3]
    draw_text(draw, (W - (cat_bar_width + cat_w) // 2, H - bar_height + (bar_height - cat_h) // 2 - 5), settings['cat']['name'], cat_font, '#FFFFFF', (0, 0, 0, 128))
    full_text = " ".join(text_lines) + "   ***   "; full_text_processed = process_text(full_text)
    text_width = ticker_font.getbbox(full_text_processed)[2]; progress = frame_idx / total_frames
    total_scroll_dist = (W * 0.7) + text_width; start_pos = W; current_x = start_pos - (total_scroll_dist * progress)
    draw_text(draw, (current_x, H - bar_height + (bar_height - font_size) // 2 - 10), full_text, ticker_font, settings['text_color'], settings['shadow_color'])
    draw_text(draw, (current_x + text_width, H - bar_height + (bar_height - font_size) // 2 - 10), full_text, ticker_font, settings['text_color'], settings['shadow_color']); return frame

def render_title_scene(writer, duration, text, image_path, settings):
    W, H = settings['dimensions']; FPS = 30; frames = int(duration * FPS); img = Image.open(image_path).convert("RGB")
    title_font = ImageFont.truetype(settings['font_file'], int(W / 12 if H > W else W / 18)); cat_font = ImageFont.truetype(settings['font_file'], int(W / 20 if H > W else W / 35))
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
    font_big = ImageFont.truetype(settings['font_file'], int(W / 20 if H > W else W / 28)); font_small = ImageFont.truetype(settings['font_file'], int(W / 30 if H > W else W / 45))
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
                h = hex_color.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha_val,)
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
def create_story_video(article_data, image_paths, settings, status_placeholder):
    W, H = settings['dimensions']; FPS = 30
    if not image_paths:
        status_placeholder.error("!! خطأ فادح: لا توجد صور لإنشاء الفيديو.")
        return None, None

    render_function = {"ديناميكي": render_dynamic_split_scene, "سينمائي": render_cinematic_overlay_scene, "عصري": render_modern_grid_scene, "شريط إخباري": render_news_ticker_scene}[settings['design_choice']]
    
    outro_duration_final = settings['outro_duration'] if settings['enable_outro'] else 0.0
    
    scenes=[]; current_duration = settings['intro_duration'] + outro_duration_final
    content_sentences=[s.strip() for s in article_data.get('content','').split('.') if len(s.strip())>20]
    available_images=image_paths[1:] if len(image_paths)>1 else list(image_paths); current_text_chunk=""
    
    for sentence in content_sentences:
        current_text_chunk += sentence + ". "; words_in_chunk = len(current_text_chunk.split())
        base_duration = max(settings['min_scene_duration'], words_in_chunk / 2.5)
        estimated_scene_duration = base_duration * settings['pacing_multiplier']
        if current_duration + estimated_scene_duration > settings['max_video_duration']: break
        if words_in_chunk > 30 and available_images:
            img_scene = available_images.pop(0); scenes.append({'duration': estimated_scene_duration, 'text': current_text_chunk, 'image': img_scene})
            current_duration += estimated_scene_duration; current_text_chunk = ""
            if not available_images: available_images = list(image_paths)
            
    if not scenes and content_sentences:
        text = " ".join(content_sentences)
        base_duration = max(settings['min_scene_duration'], len(text.split()) / 2.5)
        duration = base_duration * settings['pacing_multiplier']
        scenes.append({'duration': duration, 'text': text, 'image': image_paths[0]})

    temp_videos = []
    if settings.get('intro_video'):
        status_placeholder.text("إعداد مقدمة الفيديو..."); resized_intro = f"temp_media/resized_intro.mp4"
        (ffmpeg.input(settings['intro_video']).filter('scale', W, H).output(resized_intro, r=FPS).overwrite_output().run(quiet=True)); temp_videos.append(resized_intro)
    
    silent_content_path = f"temp_media/silent_content_{random.randint(1000,9999)}.mp4"
    writer = cv2.VideoWriter(silent_content_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    
    status_placeholder.text("🎬 تصيير مشهد العنوان..."); render_title_scene(writer, settings['intro_duration'], article_data['title'], image_paths[0], settings)
    sfx_times=[settings['intro_duration']]
    for i, scene in enumerate(scenes):
        status_placeholder.text(f"-> تصيير مشهد نصي ({i+1}/{len(scenes)})...")
        frames_scene=int(scene['duration']*FPS); image=Image.open(scene['image']).convert("RGB"); text_font=ImageFont.truetype(settings['font_file'], settings['font_size'])
        max_w = (W//2 - 120) if settings['design_choice'] == "ديناميكي" and W > H else (W - 160)
        text_lines=wrap_text(scene['text'],text_font,max_w)
        for j in range(frames_scene): frame=render_function(j,frames_scene,text_lines,image,settings); writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        sfx_times.append(sfx_times[-1]+scene['duration'])
    
    if settings['enable_outro']:
        status_placeholder.text("🎬 تصيير مشهد الخاتمة...")
        render_source_outro_scene(writer, settings['outro_duration'], settings['logo_file'], settings)
    
    writer.release(); temp_videos.append(silent_content_path)
    
    if settings.get('outro_video'):
        status_placeholder.text("إعداد خاتمة الفيديو..."); resized_outro = f"temp_media/resized_outro.mp4"
        (ffmpeg.input(settings['outro_video']).filter('scale', W, H).output(resized_outro, r=FPS).overwrite_output().run(quiet=True)); temp_videos.append(resized_outro)
    
    status_placeholder.text("🔄 دمج مقاطع الفيديو...");
    final_silent_video_path = f"temp_media/final_silent_{random.randint(1000,9999)}.mp4"
    concat_list_path = "temp_media/concat_list.txt"
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for v_path in temp_videos:
            absolute_path = os.path.abspath(v_path).replace('\\', '/')
            f.write(f"file '{absolute_path}'\n")
            
    (ffmpeg.input(concat_list_path, format='concat', safe=0)
     .output(final_silent_video_path, c='copy', r=FPS)
     .overwrite_output()
     .run(quiet=True))
    
    status_placeholder.text("🔊 دمج الصوتيات...");
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
    
    status_placeholder.text("🖼️ إنشاء الصورة المصغرة..."); thumbnail_name = "thumbnail.jpg"
    thumb=Image.open(image_paths[0]).convert("RGB").resize((W,H)); draw_t=ImageDraw.Draw(thumb,'RGBA')
    draw_t.rectangle([(0,0),(W,H)],fill=(0,0,0,100)); font_t=ImageFont.truetype(settings['font_file'],int(W/10 if H>W else W/15))
    lines=wrap_text(article_data['title'],font_t,W-100); y=H/2-(len(lines)*120)/2
    for line in lines: draw_text(draw_t,((W-font_t.getbbox(process_text(line))[2])/2,y),line,font_t,settings['text_color'],settings['shadow_color']); y+=120
    thumb.save(thumbnail_name,'JPEG',quality=85);
    return output_video_name, thumbnail_name

# ==============================================================================
# دوال إدارة مجموعة العلامة التجارية (Brand Kit)
# ==============================================================================
def get_brand_kits():
    kits = [d for d in os.listdir("brand_kits") if os.path.isdir(os.path.join("brand_kits", d))]
    return kits

def save_brand_kit(kit_name, settings):
    if not kit_name:
        st.error("الرجاء إدخال اسم لمجموعة العلامة التجارية.")
        return
    
    kit_folder = os.path.join("brand_kits", kit_name)
    os.makedirs(kit_folder, exist_ok=True)
    
    kit_data = {
        "cat_color": settings.get('cat', {}).get('color', '#D32F2F'),
        "text_color": settings.get('text_color', '#FFFFFF'),
        "shadow_color": settings.get('shadow_color', '#000000'),
    }
    
    files_to_copy = {"logo_file": "logo_path", "font_file": "font_path", "intro_video": "intro_path", "outro_video": "outro_path"}
    
    for setting_key, path_key in files_to_copy.items():
        original_path = settings.get(setting_key)
        if original_path and os.path.exists(original_path):
            filename = os.path.basename(original_path)
            new_path = os.path.join(kit_folder, filename)
            shutil.copy(original_path, new_path)
            kit_data[path_key] = new_path
        else:
            kit_data[path_key] = None
            
    json_path = os.path.join(kit_folder, "config.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(kit_data, f, indent=4)
        
    st.success(f"✅ تم حفظ مجموعة العلامة التجارية '{kit_name}' بنجاح!")

def load_brand_kit(kit_name):
    json_path = os.path.join("brand_kits", kit_name, "config.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            kit_data = json.load(f)
            st.session_state['loaded_kit'] = kit_data
            st.success(f"تم تحميل إعدادات '{kit_name}'.")
            # We don't rerun, values will be picked up by widgets' default values.
    else:
        st.error("لم يتم العثور على ملف الإعدادات الخاص بمجموعة العلامة التجارية.")

# ==============================================================================
#                      واجهة المستخدم الرسومية (Streamlit)
# ==============================================================================

if 'loaded_kit' not in st.session_state:
    st.session_state['loaded_kit'] = {}

with st.sidebar:
    st.header("🎬 إعدادات الفيديو الأساسية")
    aspect_ratio_option = st.selectbox("اختر أبعاد الفيديو (النسبة):", ("16:9 (يوتيوب، أفقي)", "9:16 (تيك توك، عمودي)"), key="aspect_ratio")
    dimensions = (1920, 1080) if "16:9" in aspect_ratio_option else (1080, 1920)
    
    st.header("📂 الملفات الافتراضية")
    st.info("ارفع الملفات هنا، أو قم بتحميلها من مجموعة علامة تجارية.")
    
    logo_file_uploaded = st.file_uploader("ارفع شعار القناة (PNG)", type=["png"])
    font_file_uploaded = st.file_uploader("ارفع ملف الخط (TTF)", type=["ttf"])
    intro_video_uploaded = st.file_uploader("ارفع فيديو المقدمة (اختياري)", type=["mp4"])
    outro_video_uploaded = st.file_uploader("ارفع فيديو الخاتمة (اختياري)", type=["mp4"])

    st.header("🎵 الصوتيات")
    TTS_ACCENTS = {"العربية (قياسية)": "com", "العربية (مصر)": "com.eg", "العربية (السعودية)": "com.sa"}
    selected_accent_name = st.selectbox("اختر لهجة التعليق الصوتي:", list(TTS_ACCENTS.keys()))
    tts_tld = TTS_ACCENTS[selected_accent_name]
    
    enable_tts = st.checkbox("📢 تفعيل التعليق الصوتي الآلي (TTS)")
    tts_volume = st.slider("مستوى صوت التعليق الصوتي (TTS)", 0.0, 2.0, 1.0, 0.1, disabled=not enable_tts)
    music_files_uploaded = st.file_uploader("ارفع ملفات الموسيقى الخلفية (MP3)", type=["mp3"], accept_multiple_files=True)
    sfx_file_uploaded = st.file_uploader("ارفع المؤثر الصوتي للانتقالات (MP3)", type=["mp3"])
    
    st.header("🔒 إعدادات النشر")
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", ""); TELEGRAM_CHANNEL_ID = st.secrets.get("TELEGRAM_CHANNEL_ID", "")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID: st.warning("لم يتم العثور على معلومات تليجرام في ملف الأمان.")
    else: st.success("تم تحميل إعدادات تليجرام.")

tab_main, tab_branding, tab_manual_images = st.tabs(["🚀 الإنتاج", "🎨 إدارة العلامة التجارية", "🖼️ رفع صور يدوية"])

with tab_branding:
    st.subheader("🎨 إدارة مجموعات العلامة التجارية (Brand Kits)")
    st.info("احفظ إعداداتك الحالية (الملفات والألوان) للوصول إليها بسرعة في المستقبل.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### تحميل مجموعة محفوظة")
        available_kits = get_brand_kits()
        if not available_kits:
            st.write("لا توجد مجموعات محفوظة بعد.")
        else:
            selected_kit_to_load = st.selectbox("اختر مجموعة لتحميلها:", [""] + available_kits)
            if st.button("📥 تحميل المجموعة"):
                if selected_kit_to_load:
                    load_brand_kit(selected_kit_to_load)
    
    with col2:
        st.markdown("#### حفظ الإعدادات الحالية كمجموعة جديدة")
        new_kit_name = st.text_input("أدخل اسمًا لمجموعتك الجديدة:")
        if st.button("💾 حفظ المجموعة الحالية"):
            temp_logo_path = save_uploaded_file(logo_file_uploaded, "temp_media")
            temp_font_path = save_uploaded_file(font_file_uploaded, "temp_media")
            temp_intro_path = save_uploaded_file(intro_video_uploaded, "temp_media")
            temp_outro_path = save_uploaded_file(outro_video_uploaded, "temp_media")
            
            # Get colors from session state if they exist, otherwise use default
            current_settings_for_kit = {
                "logo_file": temp_logo_path, "font_file": temp_font_path,
                "intro_video": temp_intro_path, "outro_video": temp_outro_path,
                "cat": {"color": st.session_state.get('cat_color_value', '#D32F2F')},
                "text_color": st.session_state.get('text_color_value', '#FFFFFF'),
                "shadow_color": st.session_state.get('shadow_color_value', '#000000')
            }
            save_brand_kit(new_kit_name, current_settings_for_kit)

with tab_manual_images:
    st.subheader("🖼️ رفع صور مخصصة للإنتاج الحالي")
    st.info("هذه الصور ستُستخدم فقط في عملية الإنتاج الحالية، وستُدمج مع الصور المسحوبة من الرابط.")
    manual_images_uploaded = st.file_uploader("ارفع صورة واحدة أو أكثر", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="manual_images_uploader")

with tab_main:
    st.subheader("1. أدخل المحتوى للإنتاج")
    st.info("يمكنك إدخال رابط واحد أو قائمة من الروابط (كل رابط في سطر) للمعالجة المجمعة.")
    urls_input = st.text_area("🔗 أدخل رابط المقال (أو قائمة روابط):", height=150)
    
    st.divider(); st.subheader("2. تخصيص تصميم الفيديو")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("🎨 **التصميم والألوان**")
            design_choice = st.selectbox("اختر التصميم البصري:", ("ديناميكي", "سينمائي", "عصري", "شريط إخباري"))
            NEWS_CATEGORIES={"1":{"name":"عاجل","color":"#D32F2F"}, "2":{"name":"أخبار","color":"#0080D4"}, "3":{"name":"رياضة","color":"#4CAF50"}}
            cat_name = st.selectbox("اختر فئة الخبر:", [v['name'] for v in NEWS_CATEGORIES.values()])
            cat_choice_key = [k for k, v in NEWS_CATEGORIES.items() if v['name'] == cat_name][0]
            cat_color = st.color_picker("لون الفئة:", st.session_state.loaded_kit.get('cat_color', NEWS_CATEGORIES[cat_choice_key]['color']), key='cat_color_value')
            final_cat = {'name': cat_name, 'color': cat_color}
            text_color = st.color_picker('لون النص الأساسي', st.session_state.loaded_kit.get('text_color', '#FFFFFF'), key='text_color_value')
            shadow_color = st.color_picker('لون ظل النص', st.session_state.loaded_kit.get('shadow_color', '#000000'), key='shadow_color_value')
    with col2:
         with st.container(border=True):
            st.markdown("⏱️ **المدة والإيقاع**")
            max_video_duration = st.slider("المدة القصوى للفيديو (ثانية)", 20, 180, 60)
            pacing_multiplier = st.slider("سرعة عرض المشاهد (الإيقاع)", 0.5, 2.0, 1.0, 0.1, help="أقل من 1.0 لعرض أسرع، أكبر من 1.0 لعرض أبطأ")
            min_scene_duration = st.slider("الحد الأدنى لمدة المشهد (ثانية)", 2.0, 10.0, 4.5, 0.5)
            intro_duration = st.slider("مدة مشهد العنوان (ثانية)", 2.0, 10.0, 4.0, 0.5)
            outro_duration = st.slider("مدة مشهد الخاتمة (ثانية)", 3.0, 15.0, 7.0, 0.5)
            
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown("⚙️ **إعدادات متقدمة**")
            enable_outro = st.checkbox("🎬 تفعيل مشهد الخاتمة (اللوغو)", value=True)
            font_size = st.slider("حجم الخط في المشاهد", 30, 120, int(dimensions[0]/28))
            logo_size_outro = st.slider("حجم الشعار في الخاتمة (بكسل)", 100, 800, int(dimensions[0]/4.5))
    with col4:
        with st.container(border=True):
            st.markdown("🔊 **مستويات الصوت**")
            music_volume = st.slider("مستوى صوت الموسيقى", 0.0, 1.0, 0.1, 0.05); sfx_volume = st.slider("مستوى صوت المؤثرات", 0.0, 1.0, 0.4, 0.05)

    st.divider()
    st.subheader("⚙️ إعدادات المعالجة المجمعة")
    delay_between_posts = st.slider("⏳ فترة الانتظار بين كل فيديو (بالثواني)", 0, 300, 30, help="مفيد لتجنب حظر النشر من تليجرام.")

    if st.button("🚀 **ابدأ الإنتاج**", type="primary", use_container_width=True):
        urls = [url.strip() for url in urls_input.split('\n') if url.strip().startswith('http')]
        if not urls:
            st.error("الرجاء إدخال رابط واحد على الأقل.")
        else:
            # -- تحديد المسارات النهائية للملفات --
            logo_file_path = save_uploaded_file(logo_file_uploaded) or st.session_state.loaded_kit.get('logo_path') or ("logo.png" if os.path.exists("logo.png") else None)
            FONT_FILE = save_uploaded_file(font_file_uploaded) or st.session_state.loaded_kit.get('font_path') or "Amiri-Bold.ttf"
            intro_video_path = save_uploaded_file(intro_video_uploaded, "temp_media") or st.session_state.loaded_kit.get('intro_path')
            outro_video_path = save_uploaded_file(outro_video_uploaded, "temp_media") or st.session_state.loaded_kit.get('outro_path')
            
            if not os.path.exists(FONT_FILE):
                st.error(f"ملف الخط '{FONT_FILE}' غير موجود! يرجى رفعه أو تحميله من مجموعة علامة تجارية.")
            else:
                total_urls = len(urls)
                batch_progress = st.progress(0, text=f"بدء المعالجة المجمعة لـ {total_urls} رابط...")
                status_container = st.container()
                
                for i, url in enumerate(urls):
                    current_status = status_container.empty()
                    current_status.info(f"⏳ [{i+1}/{total_urls}] جاري تحليل الرابط: {url[:70]}...")
                    
                    scraped_data = scrape_article_data(url)
                    if not scraped_data:
                        current_status.error(f"!! [{i+1}/{total_urls}] فشل في تحليل الرابط. الانتقال للتالي...")
                        time.sleep(3)
                        continue

                    article_data = scraped_data
                    
                    manual_image_paths = [save_uploaded_file(img, "temp_media") for img in st.session_state.get('manual_images_uploader', [])]
                    image_paths = download_images(article_data.get('image_urls', []))
                    image_paths.extend(manual_image_paths)
                    image_paths = sorted(set(image_paths), key=image_paths.index)

                    if not image_paths:
                        if logo_file_path and os.path.exists(logo_file_path):
                            image_paths = [logo_file_path]
                        else:
                            current_status.error(f"!! [{i+1}/{total_urls}] لا توجد صور لهذا المقال. الانتقال للتالي...")
                            time.sleep(3)
                            continue

                    settings = {
                        'dimensions': dimensions, 'tts_audio_path': None, 'tts_volume': tts_volume, 'logo_file': logo_file_path,
                        'font_file': FONT_FILE, 'intro_video': intro_video_path, 'outro_video': outro_video_path,
                        'music_files': [save_uploaded_file(f) for f in music_files_uploaded], 'sfx_file': save_uploaded_file(sfx_file_uploaded),
                        'design_choice': design_choice, 'cat': final_cat, 'text_color': text_color, 'shadow_color': shadow_color,
                        'max_video_duration': max_video_duration, 'min_scene_duration': min_scene_duration, 'intro_duration': intro_duration, 'outro_duration': outro_duration,
                        'font_size': font_size, 'logo_size': logo_size_outro, 'music_volume': music_volume, 'sfx_volume': sfx_volume,
                        'enable_outro': enable_outro, 'pacing_multiplier': pacing_multiplier,
                    }

                    if enable_tts and article_data:
                        current_status.text("⏳ جاري إنشاء التعليق الصوتي...")
                        full_text_for_tts = article_data['title'] + ". " + article_data.get('content', '')
                        tts_audio_path = generate_tts_audio(full_text_for_tts, tld=tts_tld)
                        if tts_audio_path: settings['tts_audio_path'] = tts_audio_path
                    
                    video_file, thumb_file = create_story_video(article_data, image_paths, settings, current_status)
                    
                    if video_file and thumb_file:
                        current_status.text("📤 جاري نشر الفيديو إلى تليجرام...")
                        caption=[f"<b>{article_data['title']}</b>",""]
                        if url: caption.append(f"🔗 <b>المصدر:</b> {url}")
                        
                        success = send_video_to_telegram(video_file, thumb_file, "\n".join(caption), TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID)
                        if success:
                            current_status.success(f"✅ تم إنشاء ونشر الفيديو [{i+1}/{total_urls}] بنجاح!")
                        else:
                            current_status.error(f"!! [{i+1}/{total_urls}] تم إنشاء الفيديو ولكن فشل النشر.")
                        
                        # Clean up generated files for this iteration
                        for f in [video_file, thumb_file] + image_paths:
                             if f and os.path.exists(f) and ('brand_kits' not in f and 'uploads' not in f):
                                try: os.remove(f)
                                except OSError: pass
                    else:
                        current_status.error(f"❌ [{i+1}/{total_urls}] فشلت عملية إنشاء الفيديو لهذا العنصر.")
                    
                    batch_progress.progress((i + 1) / total_urls, text=f"اكتمل {i+1} من {total_urls}")
                    if i < total_urls - 1:
                        for j in range(delay_between_posts, 0, -1):
                            current_status.info(f"⏱️ الانتظار لمدة {j} ثانية قبل الفيديو التالي...")
                            time.sleep(1)
                
                status_container.success("🎉 اكتملت جميع مهام المعالجة المجمعة بنجاح!")