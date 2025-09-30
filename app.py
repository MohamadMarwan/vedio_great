# ==============================================================================
#     منصة إنتاج المحتوى الإخباري (الإصدار 12.2 - مصحح الأخطاء)
#     - إعادة إضافة الدوال المفقودة وتصحيح المتغيرات العامة
# ==============================================================================
import os
import random
import cv2
import numpy as np
import ffmpeg
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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
st.set_page_config(page_title="منصة إنتاج المحتوى الإخباري", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 منصة إنتاج المحتوى الإخباري المتكاملة")
st.markdown("v12.2 - إنتاج الفيديو والصور | قوالب متعددة | نشر مخصص")

# إنشاء مجلدات ضرورية
if not os.path.exists("uploads"): os.makedirs("uploads")
if not os.path.exists("temp_media"): os.makedirs("temp_media")
if not os.path.exists("brand_kits"): os.makedirs("brand_kits")

# ==============================================================================
#                             الدوال المساعدة العامة
# ==============================================================================
def ease_in_out_quad(t): return 2*t*t if t<0.5 else 1-pow(-2*t+2,2)/2
def process_text(text): return get_display(arabic_reshaper.reshape(text))
def draw_text(draw, pos, text, font, fill, shadow_color, offset=(2,2)):
    proc_text=process_text(text)
    draw.text((pos[0]+offset[0],pos[1]+offset[1]),proc_text,font=font,fill=shadow_color)
    draw.text(pos,proc_text,font=font,fill=fill)

def save_uploaded_file(uploaded_file, folder="uploads"):
    if uploaded_file is not None:
        path = os.path.join(folder, uploaded_file.name);
        with open(path, "wb") as f: f.write(uploaded_file.getbuffer());
        return path
    return None

def wrap_text(text, font, max_width):
    if not text: return []
    lines, words, current_line = [], text.split(), ''
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if font.getbbox(process_text(test_line))[2] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line); current_line = word
    lines.append(current_line)
    return [l for l in lines if l]

@st.cache_data(ttl=3600)
def generate_tts_audio(text, lang='ar', tld='com'):
    try:
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        path = f"temp_media/tts_audio_{random.randint(1000, 9999)}.mp3"
        tts.save(path); return path
    except Exception as e:
        st.error(f"!! فشل في إنشاء التعليق الصوتي: {e}"); return None

# >> تمت إعادة إضافتها <<: دالة رسم النص كلمة بكلمة للفيديو
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

# ==============================================================================
# دوال مساعدة خاصة بتصميم الصور
# ==============================================================================
def draw_text_with_shadow_image(draw, position, text, font, fill_color, shadow_color, shadow_offset=3):
    x, y = position
    processed_text = process_text(text)
    draw.text((x + shadow_offset, y + shadow_offset), processed_text, font=font, fill=shadow_color, stroke_width=2, anchor='mm')
    draw.text((x, y), processed_text, font=font, fill=fill_color, anchor='mm')

def create_base_image_design(background_image_path, W=1080, H=1080, logo_path="logo.png"):
    try:
        if background_image_path and os.path.exists(background_image_path):
            base_image = Image.open(background_image_path).convert("RGB")
        elif logo_path and os.path.exists(logo_path):
            logo_img = Image.open(logo_path).convert("RGB")
            base_image = logo_img.resize((W, H)).filter(ImageFilter.GaussianBlur(20))
        else:
            raise FileNotFoundError
    except (FileNotFoundError, IOError):
        base_image = Image.new('RGB', (W, H), (15, 15, 15))
    
    w, h = base_image.size
    target_ratio = W / H; img_ratio = w / h
    if img_ratio > target_ratio:
        new_w = int(h * target_ratio); left = (w - new_w) / 2; top = 0; right = left + new_w; bottom = h
    else:
        new_h = int(w / target_ratio); left = 0; top = (h - new_h) / 2; right = w; bottom = top + new_h
    return base_image.crop((left, top, right, bottom)).resize((W, H), Image.Resampling.LANCZOS)

def draw_footer_image(draw_or_img, W, H, logo_path, footer_text, font_path):
    try:
        logo = Image.open(logo_path).convert("RGBA").resize((50, 50))
        footer_font = ImageFont.truetype(font_path, 30)
        footer_text_proc = process_text(footer_text)
        text_bbox = footer_font.getbbox(footer_text_proc)
        total_width = text_bbox[2] + logo.width + 15
        start_x = (W - total_width) / 2; text_y = H - 45
        logo_x = int(start_x + text_bbox[2] + 15); logo_y = H - 70

        if isinstance(draw_or_img, Image.Image):
             final_image = draw_or_img; draw_context = ImageDraw.Draw(final_image, 'RGBA')
        else: final_image = draw_or_img.im; draw_context = draw_or_img
        
        draw_context.text((start_x, text_y), footer_text_proc, font=footer_font, fill="#CCCCCC", anchor="ls")
        final_image.paste(logo, (logo_x, logo_y), logo)
    except (FileNotFoundError, IOError):
        footer_font = ImageFont.truetype(font_path, 35)
        draw = ImageDraw.Draw(draw_or_img) if isinstance(draw_or_img, Image.Image) else draw_or_img
        draw.text((W/2, H-40), process_text(footer_text), font=footer_font, fill="#CCCCCC", anchor="mm")

# ==============================================================================
# دوال تصاميم الصور
# ==============================================================================
def design_cinematic(title, settings):
    W, H = 1080, 1080
    final_image = create_base_image_design(settings['image_path'], W, H, settings['logo_path'])
    draw = ImageDraw.Draw(final_image, 'RGBA')
    draw.rectangle([(0,0), (W,H)], fill=(0,0,0,70))
    tag_font = ImageFont.truetype(settings['font_path'], 45)
    tag_text_proc = process_text(settings['tag_name'])
    tag_bbox = tag_font.getbbox(tag_text_proc)
    tag_width = tag_bbox[2] + 60; tag_height = tag_bbox[3] + 30
    tag_x, tag_y = W - tag_width - 40, 40
    draw.rounded_rectangle([tag_x, tag_y, tag_x + tag_width, tag_y + tag_height], radius=tag_height/2, fill=settings['primary_color_rgba'])
    draw.text((tag_x + tag_width/2, tag_y + tag_height/2), tag_text_proc, font=tag_font, fill=settings['text_color'], anchor="mm")
    news_font = ImageFont.truetype(settings['font_path'], 100 if len(title) < 40 else 80)
    wrapped_lines = wrap_text(title, news_font, max_width=W - 120)
    if wrapped_lines:
        line_heights = [news_font.getbbox(process_text(line))[3] + 20 for line in wrapped_lines]
        total_text_height = sum(line_heights)
        plate_height = total_text_height + 60
        plate_y_start = H - plate_height - 120
        draw.rectangle([(0, plate_y_start), (W, plate_y_start + plate_height)], fill=(0, 0, 0, 160))
        text_y_start = plate_y_start + (plate_height/2) - (total_text_height/2) + 20
        for i, line in enumerate(wrapped_lines):
            draw_text_with_shadow_image(draw, (W / 2, text_y_start), line, news_font, settings['text_color'], settings['shadow_color'])
            text_y_start += line_heights[i]
    draw_footer_image(final_image, W, H, settings['logo_path'], "تابعنا", settings['font_path'])
    return final_image

def design_urgent(title, settings):
    W, H = 1080, 1080
    final_image = create_base_image_design(settings['image_path'], W, H, settings['logo_path'])
    draw = ImageDraw.Draw(final_image, 'RGBA')
    draw.rectangle([(0,0), (W,H)], fill=(0,0,0,70))
    frame_color = settings['primary_color']
    outer_border = 20; inner_border_gap = 6; inner_border_width = 5
    draw.rectangle([(0,0), (W,H)], outline=frame_color, width=outer_border)
    inner_pos = outer_border + inner_border_gap
    draw.rectangle([(inner_pos, inner_pos), (W-inner_pos, H-inner_pos)], outline=frame_color, width=inner_border_width)
    tag_font = ImageFont.truetype(settings['font_path'], 70)
    tag_y_pos = outer_border + inner_border_gap + 40
    draw_text_with_shadow_image(draw, (W/2, tag_y_pos), settings['tag_name'], tag_font, settings['text_color'], settings['shadow_color'], shadow_offset=4)
    text_margin = inner_pos + inner_border_width + 20
    news_font = ImageFont.truetype(settings['font_path'], 85 if len(title) < 50 else 70)
    wrapped_lines = wrap_text(title, news_font, max_width=W - (2 * text_margin))
    if wrapped_lines:
        line_heights = [news_font.getbbox(process_text(line))[3] + 15 for line in wrapped_lines]
        total_text_height = sum(line_heights)
        text_y_start = H / 2 - total_text_height / 2 + 40
        for i, line in enumerate(wrapped_lines):
            draw_text_with_shadow_image(draw, (W/2, text_y_start), line, news_font, settings['text_color'], settings['shadow_color'], shadow_offset=4)
            text_y_start += line_heights[i]
    draw_footer_image(final_image, W, H, settings['logo_path'], "تابعنا", settings['font_path'])
    return final_image

def design_luxury(title, settings):
    W, H = 1080, 1080
    final_image = create_base_image_design(settings['image_path'], W, H, settings['logo_path'])
    final_image = final_image.convert("L").convert("RGBA")
    draw = ImageDraw.Draw(final_image)
    gold_color = settings['primary_color']
    border_width = 25
    draw.rectangle([(0,0), (W,H)], outline=gold_color, width=border_width)
    draw.rectangle([(border_width+5, border_width+5), (W-border_width-5, H-border_width-5)], outline=gold_color, width=3)
    if settings['logo_path'] and os.path.exists(settings['logo_path']):
        logo = Image.open(settings['logo_path']).convert("RGBA").resize((90, 90))
        logo_pos = (W - logo.width - border_width - 10, border_width + 10)
        final_image.paste(logo, logo_pos, logo)
    plate_height = 300
    footer_font = ImageFont.truetype(settings['font_path'], 40)
    footer_y_pos = H - plate_height - 45 
    draw_text_with_shadow_image(draw, (W/2, footer_y_pos), "تابعنا", footer_font, settings['text_color'], (0,0,0,150), shadow_offset=2)
    draw.rectangle([(0, H - plate_height), (W, H)], fill=(0,0,0,200))
    news_font = ImageFont.truetype(settings['font_path'], 70 if len(title) < 60 else 60)
    wrapped_lines = wrap_text(title, news_font, max_width=W - 120)
    if wrapped_lines:
        line_heights = [news_font.getbbox(process_text(line))[3] + 10 for line in wrapped_lines]
        total_text_height = sum(line_heights)
        text_y_start = H - plate_height + (plate_height - total_text_height) / 2
        for i, line in enumerate(wrapped_lines):
            draw.text((W/2, text_y_start), process_text(line), font=news_font, fill=gold_color, anchor="mm")
            text_y_start += line_heights[i]
    return final_image

def design_arab(title, settings):
    W, H = 1080, 1080
    final_image = create_base_image_design(settings['image_path'], W, H, settings['logo_path'])
    draw = ImageDraw.Draw(final_image, 'RGBA')
    hex_color = settings['primary_color'].lstrip('#')
    primary_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    draw.rectangle([(0,0), (W,H)], fill=primary_color_rgb + (60,))
    if settings['logo_path'] and os.path.exists(settings['logo_path']):
        logo = Image.open(settings['logo_path']).convert("RGBA").resize((70, 70))
        final_image.paste(logo, (40, 40), logo)
    plate_height = 400
    for i in range(plate_height):
        alpha = int(200 * (i / plate_height)); draw.line([(0, H - i), (W, H - i)], fill=(0, 0, 0, alpha))
    tag_font = ImageFont.truetype(settings['font_path'], 50)
    tag_text = f"~ {settings['tag_name']} ~"; draw.text((W/2, H - plate_height + 40), process_text(tag_text), font=tag_font, fill=settings['text_color'], anchor="mm")
    news_font = ImageFont.truetype(settings['font_path'], 80 if len(title) < 55 else 68)
    wrapped_lines = wrap_text(title, news_font, max_width=W - 100)
    if wrapped_lines:
        line_heights = [news_font.getbbox(process_text(line))[3] + 15 for line in wrapped_lines]
        total_text_height = sum(line_heights)
        text_y_start = H - total_text_height / 2 - 110 
        for i, line in enumerate(wrapped_lines):
            draw_text_with_shadow_image(draw, (W/2, text_y_start), line, news_font, settings['text_color'], settings['shadow_color'])
            text_y_start += line_heights[i]
    return final_image

# ==============================================================================
#                      دوال إنتاج الفيديو
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
            pass
    return paths

def send_to_telegram(file_path, caption, token, channel_id, is_photo):
    try:
        bot_url = f"https://api.telegram.org/bot{token}/"
        endpoint = "sendPhoto" if is_photo else "sendVideo"
        url = bot_url + endpoint
        
        with open(file_path, 'rb') as file:
            files = {'photo': file} if is_photo else {'video': file}
            payload = {'chat_id': channel_id, 'caption': caption, 'parse_mode': 'HTML'}
            if not is_photo:
                payload['supports_streaming'] = True
                # For videos, we send the thumbnail as well
                thumb_path = file_path.replace(".mp4", ".jpg")
                if os.path.exists(thumb_path):
                    with open(thumb_path, 'rb') as thumb_file:
                        files['thumb'] = thumb_file
                        response = requests.post(url, data=payload, files=files, timeout=1800)
                else:
                    response = requests.post(url, data=payload, files=files, timeout=1800)
            else:
                 response = requests.post(url, data=payload, files=files, timeout=1800)

            if response.status_code == 200:
                return True
            else:
                st.error(f"!! فشل النشر: {response.status_code} - {response.text}")
                return False
    except requests.exceptions.RequestException as e:
        st.error(f"!! خطأ فادح أثناء الاتصال بتليجرام: {e}"); return False

def fit_image_to_frame_video(img, target_w, target_h, frame_idx, total_frames):
    img_w, img_h = img.size; target_aspect = target_w / target_h; img_aspect = img_w / img_h
    if img_aspect > target_aspect: new_h = target_h; new_w = int(new_h * img_aspect)
    else: new_w = target_w; new_h = int(new_w / img_aspect)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    zoom_factor=1.20; progress=frame_idx/total_frames; current_zoom=1+(zoom_factor-1)*ease_in_out_quad(progress)
    zoomed_w,zoomed_h=int(target_w*current_zoom),int(target_h*current_zoom)
    zoom_img = img.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)
    x_offset=(zoomed_w-target_w)/2; y_offset=(zoomed_h-target_h)*progress
    return zoom_img.crop((x_offset,y_offset,x_offset+target_w,y_offset+target_h))

def render_dynamic_split_scene_video(frame_idx, total_frames, text_lines, image, settings):
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

def render_cinematic_overlay_scene_video(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame_video(image, W, H, frame_idx, total_frames)
    draw = ImageDraw.Draw(frame, 'RGBA'); text_font = ImageFont.truetype(settings['font_file'], settings['font_size'])
    line_height = text_font.getbbox("ا")[3] + 20; plate_height = min((len(text_lines) * line_height) + 60, H // 2)
    draw.rectangle([(0, H - plate_height), (W, H)], fill=(0, 0, 0, 180)); total_words = len(" ".join(text_lines).split())
    words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    draw_text_word_by_word(draw, (40, H - plate_height, W - 80, plate_height), text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_modern_grid_scene_video(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame_video(image, W, H, frame_idx, total_frames).point(lambda p: p * 0.5)
    draw = ImageDraw.Draw(frame, 'RGBA'); padding = 80 if W > H else 40
    draw.rectangle([(padding, padding), (W - padding, H - padding)], outline=settings['cat']['color'], width=5)
    draw.rectangle([(padding + 10, padding + 10), (W - padding - 10, H - padding - 10)], fill=(0, 0, 0, 190))
    text_font = ImageFont.truetype(settings['font_file'], settings['font_size']); total_words = len(" ".join(text_lines).split())
    words_to_show = int((frame_idx / total_frames) * total_words * 1.5) + 1
    box = (padding + 40, padding + 40, W - 2 * (padding + 40), H - 2 * (padding + 40))
    draw_text_word_by_word(draw, box, text_lines, words_to_show, text_font, settings['text_color'], settings['shadow_color']); return frame

def render_news_ticker_scene_video(frame_idx, total_frames, text_lines, image, settings):
    W, H = settings['dimensions']; frame = fit_image_to_frame_video(image, W, H, frame_idx, total_frames)
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

def render_title_scene_video(writer, duration, text, image_path, settings):
    W, H = settings['dimensions']; FPS = 30; frames = int(duration * FPS); img = Image.open(image_path).convert("RGB")
    title_font = ImageFont.truetype(settings['font_file'], int(W / 12 if H > W else W / 18)); cat_font = ImageFont.truetype(settings['font_file'], int(W / 20 if H > W else W / 35))
    cat = settings['cat']
    for i in range(frames):
        frame = fit_image_to_frame_video(img, W, H, i, frames); draw = ImageDraw.Draw(frame, 'RGBA')
        draw.rectangle([(0, H * 0.6), (W, H)], fill=(0, 0, 0, 180)); cat_bbox = cat_font.getbbox(process_text(cat['name']))
        draw_text(draw, (W - cat_bbox[2] - 40, H * 0.65), cat['name'], cat_font, cat['color'], (0, 0, 0, 150))
        wrapped_lines = wrap_text(text, title_font, W - 80); y = H * 0.72
        for line in wrapped_lines:
            bbox = title_font.getbbox(process_text(line))
            draw_text(draw, (W - bbox[2] - 40, y), line, title_font, settings['text_color'], settings['shadow_color']); y += bbox[3] * 1.3
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

def render_source_outro_scene_video(writer, duration, logo_path, settings):
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

def create_story_video(article_data, image_paths, settings, status_placeholder):
    W, H = settings['dimensions']; FPS = 30
    if not image_paths:
        status_placeholder.error("!! خطأ فادح: لا توجد صور لإنشاء الفيديو.")
        return None, None

    render_function = {"ديناميكي": render_dynamic_split_scene_video, "سينمائي": render_cinematic_overlay_scene_video, "عصري": render_modern_grid_scene_video, "شريط إخباري": render_news_ticker_scene_video}[settings['design_choice']]
    
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
    
    status_placeholder.text("🎬 تصيير مشهد العنوان..."); render_title_scene_video(writer, settings['intro_duration'], article_data['title'], image_paths[0], settings)
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
        render_source_outro_scene_video(writer, settings['outro_duration'], settings['logo_file'], settings)
    
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
        output_video_name = os.path.join("temp_media", "final_news_story.mp4")
        if audio_inputs:
            mixed_audio=ffmpeg.filter(audio_inputs,'amix',duration='longest',inputs=len(audio_inputs))
            ffmpeg.output(vid_stream, mixed_audio, output_video_name, vcodec='libx264', acodec='aac', pix_fmt='yuv420p', preset='fast', crf=28, audio_bitrate='96k').overwrite_output().run(quiet=True)
        else: ffmpeg.output(vid_stream, output_video_name, vcodec='copy').run(quiet=True)
    except ffmpeg.Error as e: st.error(f"!! خطأ FFMPEG: {e.stderr.decode()}"); return None,None
    
    status_placeholder.text("🖼️ إنشاء الصورة المصغرة..."); 
    thumbnail_name = os.path.join("temp_media", "thumbnail.jpg")
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
    if not os.path.exists("brand_kits"): return []
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
    else:
        st.error("لم يتم العثور على ملف الإعدادات الخاص بمجموعة العلامة التجارية.")

# ==============================================================================
#                      الواجهة الرئيسية للتطبيق
# ==============================================================================

if 'loaded_kit' not in st.session_state:
    st.session_state['loaded_kit'] = {}

TELEGRAM_VIDEO_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "") 
# >> تم التغيير <<: قراءة التوكن الجديد الخاص بالصور
TELEGRAM_IMAGE_BOT_TOKEN = st.secrets.get("TELEGRAM_imege_TOKEN", "") 
TELEGRAM_CHANNELS = st.secrets.get("telegram_channels", {})

# -- إنشاء التبويبات الرئيسية --
video_tab, image_tab, brand_tab = st.tabs(["🎬 إنتاج الفيديو", "🖼️ تصميم الصور", "🎨 إدارة العلامة التجارية"])

# ========================== تبويب إدارة العلامة التجارية ==========================
with brand_tab:
    st.header("🎨 إدارة مجموعات العلامة التجارية (Brand Kits)")
    st.info("احفظ إعداداتك الحالية (الملفات والألوان) للوصول إليها بسرعة في المستقبل.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### تحميل مجموعة محفوظة")
        available_kits = get_brand_kits()
        if not available_kits:
            st.write("لا توجد مجموعات محفوظة بعد.")
        else:
            selected_kit_to_load = st.selectbox("اختر مجموعة لتحميلها:", [""] + available_kits, key="kit_loader")
            if st.button("📥 تحميل المجموعة"):
                if selected_kit_to_load:
                    load_brand_kit(selected_kit_to_load)
                    st.rerun()
    
    with col2:
        st.markdown("#### حفظ الإعدادات الحالية كمجموعة جديدة")
        new_kit_name = st.text_input("أدخل اسمًا لمجموعتك الجديدة:")
        st.warning("الحفظ يأخذ الإعدادات من تبويب 'إنتاج الفيديو'")
        if st.button("💾 حفظ المجموعة الحالية"):
            # We need to get the paths of the currently uploaded files for the video tab
            temp_logo_path = save_uploaded_file(st.session_state.get('logo_file_uploader_state'), "temp_media")
            temp_font_path = save_uploaded_file(st.session_state.get('font_file_uploader_state'), "temp_media")
            temp_intro_path = save_uploaded_file(st.session_state.get('intro_video_uploader_state'), "temp_media")
            temp_outro_path = save_uploaded_file(st.session_state.get('outro_video_uploader_state'), "temp_media")
            
            current_settings_for_kit = {
                "logo_file": temp_logo_path, "font_file": temp_font_path,
                "intro_video": temp_intro_path, "outro_video": temp_outro_path,
                "cat": {"color": st.session_state.get('video_cat_color_value', '#D32F2F')},
                "text_color": st.session_state.get('video_text_color_value', '#FFFFFF'),
                "shadow_color": st.session_state.get('video_shadow_color_value', '#000000')
            }
            save_brand_kit(new_kit_name, current_settings_for_kit)

# ========================== تبويب إنتاج الفيديو ==========================
with video_tab:
    video_sidebar_placeholder = st.sidebar.container()
    video_main_placeholder = st.container()
    
    # >> تم التصحيح <<: تعريف المتغير dimensions في مكان عام
    aspect_ratio_option = video_sidebar_placeholder.selectbox("اختر أبعاد الفيديو:", ("16:9 (أفقي)", "9:16 (عمودي)"), key="aspect_ratio")
    dimensions = (1920, 1080) if "16:9" in aspect_ratio_option else (1080, 1920)

    with video_sidebar_placeholder:
        st.header("إعدادات الفيديو")
        with st.expander("📂 ملفات الفيديو", expanded=True):
            logo_file_uploaded_vid = st.file_uploader("شعار الفيديو", type=["png"], key="logo_file_uploader_state")
            font_file_uploaded_vid = st.file_uploader("خط الفيديو", type=["ttf"], key="font_file_uploader_state")
            intro_video_uploaded_vid = st.file_uploader("مقدمة الفيديو", type=["mp4"], key="intro_video_uploader_state")
            outro_video_uploaded_vid = st.file_uploader("خاتمة الفيديو", type=["mp4"], key="outro_video_uploader_state")
        with st.expander("🎵 صوتيات الفيديو", expanded=True):
            TTS_ACCENTS = {"العربية (قياسية)": "com", "العربية (مصر)": "com.eg", "العربية (السعودية)": "com.sa"}
            selected_accent_name_vid = st.selectbox("اختر لهجة التعليق الصوتي:", list(TTS_ACCENTS.keys()))
            tts_tld_vid = TTS_ACCENTS[selected_accent_name_vid]
            enable_tts_vid = st.checkbox("📢 تفعيل التعليق الصوتي للفيديو")
            tts_volume_vid = st.slider("مستوى صوت التعليق", 0.0, 2.0, 1.0, 0.1, disabled=not enable_tts_vid)
            music_files_uploaded_vid = st.file_uploader("موسيقى خلفية للفيديو", type=["mp3"], accept_multiple_files=True)
            sfx_file_uploaded_vid = st.file_uploader("مؤثرات صوتية للفيديو", type=["mp3"])

    with video_main_placeholder:
        st.header("إنشاء فيديو إخباري آلي")
        st.info("يمكنك إدخال رابط واحد أو قائمة من الروابط (كل رابط في سطر) للمعالجة المجمعة.")
        urls_input = st.text_area("🔗 أدخل رابط المقال (أو قائمة روابط):", height=150, key="video_urls_input")
        
        manual_images_uploaded_vid = st.file_uploader("ارفع صورًا يدوية للفيديو", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="video_manual_images")
        
        st.divider(); st.subheader("تخصيص تصميم الفيديو")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("🎨 **التصميم والألوان**")
                design_choice_vid = st.selectbox("اختر تصميم الفيديو:", ("ديناميكي", "سينمائي", "عصري", "شريط إخباري"))
                NEWS_CATEGORIES={"1":{"name":"عاجل","color":"#D32F2F"}, "2":{"name":"أخبار","color":"#0080D4"}, "3":{"name":"رياضة","color":"#4CAF50"}}
                cat_name_vid = st.selectbox("اختر فئة الخبر للفيديو:", [v['name'] for v in NEWS_CATEGORIES.values()])
                cat_choice_key_vid = [k for k, v in NEWS_CATEGORIES.items() if v['name'] == cat_name_vid][0]
                cat_color_vid = st.color_picker("لون فئة الفيديو:", st.session_state.loaded_kit.get('cat_color', NEWS_CATEGORIES[cat_choice_key_vid]['color']), key='video_cat_color_value')
                final_cat_vid = {'name': cat_name_vid, 'color': cat_color_vid}
                text_color_vid = st.color_picker('لون نص الفيديو', st.session_state.loaded_kit.get('text_color', '#FFFFFF'), key='video_text_color_value')
                shadow_color_vid = st.color_picker('لون ظل نص الفيديو', st.session_state.loaded_kit.get('shadow_color', '#000000'), key='video_shadow_color_value')
        with col2:
             with st.container(border=True):
                st.markdown("⏱️ **المدة والإيقاع**")
                max_video_duration = st.slider("المدة القصوى للفيديو (ثانية)", 20, 180, 60)
                pacing_multiplier = st.slider("سرعة عرض المشاهد", 0.5, 2.0, 1.0, 0.1, help="أقل من 1.0 لعرض أسرع، أكبر من 1.0 لعرض أبطأ")
                min_scene_duration = st.slider("الحد الأدنى لمدة المشهد (ثانية)", 2.0, 10.0, 4.5, 0.5)
                intro_duration = st.slider("مدة مشهد العنوان (ثانية)", 2.0, 10.0, 4.0, 0.5)
                outro_duration = st.slider("مدة مشهد الخاتمة (ثانية)", 3.0, 15.0, 7.0, 0.5)
                
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("⚙️ **إعدادات متقدمة**")
                enable_outro = st.checkbox("🎬 تفعيل مشهد الخاتمة للفيديو", value=True)
                font_size = st.slider("حجم خط الفيديو", 30, 120, int(dimensions[0]/28))
                logo_size_outro = st.slider("حجم شعار الخاتمة", 100, 800, int(dimensions[0]/4.5))
        with col4:
            with st.container(border=True):
                st.markdown("🔊 **مستويات الصوت**")
                music_volume = st.slider("صوت الموسيقى", 0.0, 1.0, 0.1, 0.05); sfx_volume = st.slider("صوت المؤثرات", 0.0, 1.0, 0.4, 0.05)

        st.divider()
        st.subheader("⚙️ إعدادات النشر المجمع")
        delay_between_posts = st.slider("⏳ فترة الانتظار بين كل فيديو (ثانية)", 0, 300, 30)
        selected_channel_name_vid = st.selectbox("اختر القناة لنشر الفيديو:", list(TELEGRAM_CHANNELS.keys()), key="video_channel_select")

        if st.button("🚀 **ابدأ إنتاج الفيديو**", type="primary", use_container_width=True):
            urls = [url.strip() for url in urls_input.split('\n') if url.strip().startswith('http')]
            if not urls:
                st.error("الرجاء إدخال رابط واحد على الأقل.")
            else:
                logo_file_path = save_uploaded_file(logo_file_uploaded_vid) or st.session_state.loaded_kit.get('logo_path') or ("logo.png" if os.path.exists("logo.png") else None)
                FONT_FILE = save_uploaded_file(font_file_uploaded_vid) or st.session_state.loaded_kit.get('font_path') or "Amiri-Bold.ttf"
                intro_video_path = save_uploaded_file(intro_video_uploaded_vid, "temp_media") or st.session_state.loaded_kit.get('intro_path')
                outro_video_path = save_uploaded_file(outro_video_uploaded_vid, "temp_media") or st.session_state.loaded_kit.get('outro_path')
                
                if not os.path.exists(FONT_FILE):
                    st.error(f"ملف الخط '{FONT_FILE}' غير موجود!")
                else:
                    total_urls = len(urls)
                    batch_progress = st.progress(0, text=f"بدء المعالجة المجمعة لـ {total_urls} رابط...")
                    status_container = st.container()
                    target_channel_id_vid = TELEGRAM_CHANNELS.get(selected_channel_name_vid)

                    for i, url in enumerate(urls):
                        current_status = status_container.empty()
                        with current_status.container():
                            st.info(f"⏳ [{i+1}/{total_urls}] جاري تحليل الرابط: {url[:70]}...")
                            scraped_data = scrape_article_data(url)
                            if not scraped_data:
                                st.error(f"!! [{i+1}/{total_urls}] فشل تحليل الرابط. الانتقال للتالي..."); time.sleep(3); continue

                            article_data = scraped_data
                            manual_image_paths_vid = [save_uploaded_file(img, "temp_media") for img in manual_images_uploaded_vid]
                            image_paths = download_images(article_data.get('image_urls', []))
                            image_paths.extend(manual_image_paths_vid)
                            image_paths = sorted(set(image_paths), key=image_paths.index)

                            if not image_paths:
                                if logo_file_path and os.path.exists(logo_file_path): image_paths = [logo_file_path]
                                else: st.error(f"!! [{i+1}/{total_urls}] لا توجد صور. الانتقال للتالي..."); time.sleep(3); continue

                            settings = {
                                'dimensions': dimensions, 'tts_audio_path': None, 'tts_volume': tts_volume_vid, 'logo_file': logo_file_path,
                                'font_file': FONT_FILE, 'intro_video': intro_video_path, 'outro_video': outro_video_path,
                                'music_files': [save_uploaded_file(f) for f in music_files_uploaded_vid], 'sfx_file': save_uploaded_file(sfx_file_uploaded_vid),
                                'design_choice': design_choice_vid, 'cat': final_cat_vid, 'text_color': text_color_vid, 'shadow_color': shadow_color_vid,
                                'max_video_duration': max_video_duration, 'min_scene_duration': min_scene_duration, 'intro_duration': intro_duration, 'outro_duration': outro_duration,
                                'font_size': font_size, 'logo_size': logo_size_outro, 'music_volume': music_volume, 'sfx_volume': sfx_volume,
                                'enable_outro': enable_outro, 'pacing_multiplier': pacing_multiplier,
                            }

                            if enable_tts_vid and article_data:
                                st.text("⏳ جاري إنشاء التعليق الصوتي...")
                                full_text_for_tts = article_data['title'] + ". " + article_data.get('content', '')
                                tts_audio_path = generate_tts_audio(full_text_for_tts, tld=tts_tld_vid)
                                if tts_audio_path: settings['tts_audio_path'] = tts_audio_path
                            
                            video_file, thumb_file = create_story_video(article_data, image_paths, settings, st)
                            
                            if video_file and thumb_file:
                                st.text("📤 جاري نشر الفيديو إلى تليجرام...")
                                caption=[f"<b>{article_data['title']}</b>",""]
                                if url: caption.append(f"🔗 <b>المصدر:</b> {url}")
                                
                                # >> تم التغيير <<: استخدام المتغير الخاص ببوت الفيديو
                                success = send_to_telegram(video_file, caption, TELEGRAM_VIDEO_BOT_TOKEN, target_channel_id_vid, is_photo=False)
                                if success: st.success(f"✅ تم نشر الفيديو [{i+1}/{total_urls}] بنجاح!")
                                else: st.error(f"!! [{i+1}/{total_urls}] فشل النشر.")
                                
                                for f in [video_file, thumb_file] + image_paths:
                                     if f and os.path.exists(f) and ('brand_kits' not in f and 'uploads' not in f):
                                        try: os.remove(f)
                                        except OSError: pass
                            else: st.error(f"❌ [{i+1}/{total_urls}] فشلت عملية إنشاء الفيديو.")
                            
                            batch_progress.progress((i + 1) / total_urls, text=f"اكتمل {i+1} من {total_urls}")
                            if i < total_urls - 1:
                                for j in range(delay_between_posts, 0, -1):
                                    st.info(f"⏱️ الانتظار لمدة {j} ثانية قبل الفيديو التالي..."); time.sleep(1)
                
                status_container.success("🎉 اكتملت جميع مهام المعالجة المجمعة للفيديو بنجاح!")

# ========================== تبويب تصميم الصور ==========================
with image_tab:
    st.header("تصميم صورة إخبارية احترافية")
    
    col1_img, col2_img = st.columns(2)
    
    with col1_img:
        st.subheader("1. المحتوى")
        news_title_img = st.text_area("✍️ أدخل نص الخبر", height=150, key="img_news_title")
        article_url_img = st.text_input("🔗 أو اسحب العنوان من رابط", help="إذا تم توفير رابط، سيتم سحب العنوان والصورة الرئيسية منه تلقائيًا.", key="img_article_url")
        
        st.subheader("2. الصورة")
        image_source = st.radio("اختر مصدر الصورة:", ("رفع صورة", "استخدام رابط صورة", "السحب من رابط المقال (افتراضي)"), key="img_source")
        
        bg_image_path = None
        if image_source == "رفع صورة":
            uploaded_bg_img = st.file_uploader("ارفع صورة الخلفية", type=['png', 'jpg', 'jpeg'], key="img_bg_uploader")
            if uploaded_bg_img:
                bg_image_path = save_uploaded_file(uploaded_bg_img, "temp_media")
        elif image_source == "استخدام رابط صورة":
            image_url_input = st.text_input("الصق رابط الصورة هنا", key="img_url_input")
            if image_url_input:
                with st.spinner("جاري تنزيل الصورة..."):
                    downloaded = download_images([image_url_input])
                    if downloaded: bg_image_path = downloaded[0]

    with col2_img:
        st.subheader("3. التصميم والنشر")
        
        IMAGE_DESIGN_TEMPLATES = {
            "سينمائي": {"func": design_cinematic, "default_color": "#202020"},
            "عاجل": {"func": design_urgent, "default_color": "#C80000"},
            "فخم": {"func": design_luxury, "default_color": "#D4AF37"},
            "عربي": {"func": design_arab, "default_color": "#006478"},
        }
        selected_design_name = st.selectbox("🎨 اختر قالب التصميم:", list(IMAGE_DESIGN_TEMPLATES.keys()))
        design_info = IMAGE_DESIGN_TEMPLATES[selected_design_name]

        with st.expander("🖌️ تخصيص الألوان والخطوط", expanded=True):
            primary_color = st.color_picker("اللون الأساسي للقالب", design_info['default_color'], key="img_primary_color")
            text_color_img = st.color_picker("لون النص", st.session_state.loaded_kit.get('text_color', '#FFFFFF'), key="img_text_color")
            shadow_color_img = st.color_picker("لون ظل النص", st.session_state.loaded_kit.get('shadow_color', '#000000'), key="img_shadow_color")
            tag_name_img = st.text_input("نص الوسم (Tag)", value="أخبار", key="img_tag_name")
        
        st.subheader("4. النشر")
        if not TELEGRAM_CHANNELS:
            st.warning("لم يتم تعريف أي قنوات تليجرام في ملف secrets.toml")
            selected_channel_name_img = None
        else:
            selected_channel_name_img = st.selectbox("اختر القناة لنشر الصورة:", list(TELEGRAM_CHANNELS.keys()), key="img_channel_select")
        
        hashtag_img = st.text_input("الهاشتاجات (اختياري)", value="#أخبار #عاجل", key="img_hashtag")

    if st.button("🖼️ **إنشاء ونشر الصورة**", type="primary", use_container_width=True, key="generate_image_button"):
        final_news_title = news_title_img
        
        # منطق جديد وأبسط
        scraped_data = None
        if article_url_img:
           with st.spinner("جاري سحب البيانات من الرابط..."):
            scraped_data = scrape_article_data(article_url_img)
           if scraped_data:
            st.success("✅ تم سحب البيانات بنجاح.")
            # إذا لم يكن المستخدم قد كتب عنوانًا، استخدم العنوان المسحوب
            if not final_news_title: 
                final_news_title = scraped_data['title']
            # إذا اختار المستخدم السحب من الرابط ولم يكن قد رفع صورة بالفعل
            if image_source == "السحب من رابط المقال" and not bg_image_path:
                if scraped_data['image_urls']:
                    downloaded = download_images([scraped_data['image_urls'][0]])
                    if downloaded: 
                        bg_image_path = downloaded[0]
                else:
                    st.warning("لم يتم العثور على صورة في الرابط.")
        else:
            st.error("فشل في سحب البيانات من الرابط.")


            with st.spinner("جاري سحب البيانات من الرابط..."):
                scraped_data = scrape_article_data(article_url_img)
                if scraped_data:
                    if not final_news_title: final_news_title = scraped_data['title']
                    if image_source == "السحب من رابط المقال" and not bg_image_path and scraped_data['image_urls']:
                        downloaded = download_images([scraped_data['image_urls'][0]])
                        if downloaded: bg_image_path = downloaded[0]
                    st.success("✅ تم سحب البيانات بنجاح.")
                else:
                    st.error("فشل في سحب البيانات من الرابط.")

        if not final_news_title:
            st.error("الرجاء إدخال نص الخبر أو رابط صالح.")
        else:
            with st.spinner("جاري إنشاء التصميم..."):
                font_path_img = st.session_state.loaded_kit.get('font_path', "Amiri-Bold.ttf")
                logo_path_img = st.session_state.loaded_kit.get('logo_path', "logo.png")
                
                hex_color = primary_color.lstrip('#')
                primary_color_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                
                image_settings = {
                    "image_path": bg_image_path, "font_path": font_path_img, "logo_path": logo_path_img,
                    "tag_name": tag_name_img, "primary_color": primary_color,
                    "primary_color_rgba": primary_color_rgb + (200,),
                    "text_color": text_color_img, "shadow_color": shadow_color_img
                }
                
                design_function = design_info['func']
                final_image = design_function(final_news_title, image_settings)
                
                output_path = os.path.join("temp_media", "final_image.png")
                final_image.convert('RGB').save(output_path, quality=95)
                
                st.image(final_image, caption="الصورة النهائية", use_column_width=True)
            
            if selected_channel_name_img:
                with st.spinner("جاري النشر إلى تليجرام..."):
                    target_channel_id_img = TELEGRAM_CHANNELS[selected_channel_name_img]
                    caption_parts = [f"<b>{final_news_title}</b>", ""]
                    if article_url_img: caption_parts.append(f"🔗 <b>التفاصيل:</b> {article_url_img}")
                    if hashtag_img: caption_parts.extend(["", hashtag_img])
                    final_caption = "\n".join(caption_parts)
                    
                    # >> تم التغيير <<: استخدام المتغير الجديد الخاص ببوت الصور
                    success = send_to_telegram(output_path, final_caption, TELEGRAM_IMAGE_BOT_TOKEN, target_channel_id_img, is_photo=True)
                    if success:
                        st.success("✅ تم نشر الصورة بنجاح!")
                    else:
                        st.error("فشل نشر الصورة.")