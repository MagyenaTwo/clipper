import logging
import os
import re
import sqlite3
import subprocess
import sys
from typing import List
from urllib import response
import uuid
from fastapi import FastAPI, File, HTTPException, Request, Form, UploadFile, requests
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from moviepy import ImageClip, VideoFileClip, concatenate_videoclips
from pydantic import BaseModel
import requests
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
import youtube_transcript_api
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp
from ai_engine import analyze_transcript_with_ai, get_model_status, search_youtube_trending


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
COOKIES_PATH = "cookies.txt"
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(title="ClipSpark AI")
if not os.path.exists("static/clips"):
    os.makedirs("static/clips", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
FB_PAGE_ID = os.getenv("FB_PAGE_ID")
FB_ACCESS_TOKEN = os.getenv("FB_ACCESS_TOKEN")
CLIPS_DIR = "static/clips"
if not os.path.exists(CLIPS_DIR):
    os.makedirs(CLIPS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posted_videos (
            video_id TEXT PRIMARY KEY,
            fb_id TEXT,
            posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()
MODEL_OPTIONS = [ #nararouter
    {"name": "mimo-v2.5-free", "status": "operational"},
    {"name": "mistral-large", "status": "operational"},
    {"name": "mistral-medium-3-5", "status": "operational"},
    {"name": "mimo-v2.5-hermes", "status": "operational"},
    {"name": "mimo-v2.5-pro-hermes", "status": "operational"}
]
OPENROUTER_MODELS = [ #openrouter
    {"label": "Nemotron 3 Ultra (free)", "value": "nvidia/nemotron-3-ultra", "status": "operational"},
    {"label": "Laguna M.1 (free)", "value": "poolside/laguna-m-1", "status": "operational"},
    {"label": "Nemotron 3 Super (free)", "value": "nvidia/nemotron-3-super", "status": "operational"},
    {"label": "GPT OSS 120B (free)", "value": "openai/gpt-oss-120b", "status": "operational"},
    {"label": "Laguna XS.2 (free)", "value": "poolside/laguna-xs-2", "status": "operational"},
    {"label": "GPT OSS 20B (free)", "value": "openai/gpt-oss-20b", "status": "operational"},

    {"label": "Gemma 4 31B (free)", "value": "google/gemma-3-27b-it", "status": "operational"},
    {"label": "North Mini Code (free)", "value": "cohere/command-r7b", "status": "operational"},

    {"label": "Nemotron 3 Nano 30B A3B (free)", "value": "nvidia/nemotron-3-nano-30b-a3b", "status": "operational"},
    {"label": "Nemotron 3 Nano Omni (free)", "value": "nvidia/nemotron-3-nano-omni", "status": "operational"},
    {"label": "Nemotron Nano 9B V2 (free)", "value": "nvidia/nemotron-nano-9b-v2", "status": "operational"},
    {"label": "Nemotron Nano 12B 2 VL (free)", "value": "nvidia/nemotron-nano-12b-v2-vl", "status": "operational"},

    {"label": "Gemma 4 26B A4B (free)", "value": "google/gemma-3-27b-it", "status": "operational"},

    {"label": "Llama Nemotron Embed VL 1B V2 (free)", "value": "nvidia/llama-nemotron-embed-vl-1b-v2", "status": "operational"},
    {"label": "Llama Nemotron Rerank VL 1B V2 (free)", "value": "nvidia/llama-nemotron-rerank-vl-1b-v2", "status": "operational"},

    {"label": "LFM2.5-1.2B-Thinking (free)", "value": "liquid/lfm2.5-1.2b-thinking", "status": "operational"},
    {"label": "LFM2.5-1.2B-Instruct (free)", "value": "liquid/lfm2.5-1.2b-instruct", "status": "operational"},

    {"label": "Nemotron 3.5 Content Safety (free)", "value": "nvidia/nemotron-3.5-content-safety", "status": "operational"},

    {"label": "Qwen3 Next 80B A3B Instruct (free)", "value": "qwen/qwen3-next-80b-a3b-instruct", "status": "operational"},

    {"label": "Llama 3.3 70B Instruct (free)", "value": "meta-llama/llama-3.3-70b-instruct", "status": "operational"},
    {"label": "Venice Uncensored (free)", "value": "venice/uncensored", "status": "operational"},
    {"label": "Llama 3.2 3B Instruct (free)", "value": "meta-llama/llama-3.2-3b-instruct", "status": "operational"},

    {"label": "Hermes 3 405B Instruct (free)", "value": "nousresearch/hermes-3-405b", "status": "operational"},

    {"label": "Qwen3 Coder 480B A35B (free)", "value": "qwen/qwen3-coder-480b-a35b", "status": "operational"},
]
MODEL_STATUS_CACHE = {m["name"]: "operational" for m in MODEL_OPTIONS}
class ChannelSearchRequest(BaseModel):
    channel_name: str
    page: int = 1

class FacebookPostRequest(BaseModel):
    video_id: str
    title: str

def clean_filename(text: str) -> str:
    cleaned = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[-\s]+', '_', cleaned).strip('_')[:30]

def clean_script(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[\*\#]', '', text)
    text = re.sub(r'^(Scene|Opening|Closing|Visual).*?:', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def generate_openrouter(prompt, model):
    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return res.json()["choices"][0]["message"]["content"]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 'index.html' akan merender 'base.html' melalui template inheritance
    return templates.TemplateResponse("home.html", {"request": request, "title": "Beranda"})

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    # 'index.html' akan merender 'base.html' melalui template inheritance
    return templates.TemplateResponse("home.html", {"request": request, "title": "Beranda"})

@app.get("/generator", response_class=HTMLResponse)
async def generator(request: Request):
    # 'index.html' akan merender 'base.html' melalui template inheritance
    return templates.TemplateResponse("generator.html", {"request": request, "title": "Generator"})

@app.get("/shorts", response_class=HTMLResponse)
async def shorts(request: Request):
    # 'shorts.html' akan merender 'base.html' melalui template inheritance
    return templates.TemplateResponse("shorts.html", {"request": request, "title": "Shorts"})


def download_and_cut_video(youtube_url: str, start_time: int, end_time: int, output_title: str) -> str:
    safe_title = clean_filename(output_title)
    filename = f"{safe_title}_{start_time}_{end_time}.mp4"
    output_path = os.path.join("static/clips", filename)
    web_url = f"/static/clips/{filename}"
    
    if os.path.exists(output_path):
        return web_url
    
    command = [
        sys.executable, "-m", "yt_dlp",
        "--force-overwrites",
        "-f", "best[ext=mp4]",
        "--cookies", COOKIES_PATH,
        "--download-sections", f"*{start_time}-{end_time}",
        "--downloader", "ffmpeg",
        "--downloader-args", "ffmpeg:-c copy", 
        "-o", output_path,
        youtube_url
    ]
    
    try:
        subprocess.run(command, check=True)
        return web_url
    except Exception as e:
        logger.error(f"❌ Gagal: {str(e)}")
        return None

def generate_transcript_fallback(youtube_url: str) -> str:
    """Fungsi cadangan untuk membuat transkrip dari audio jika YouTube tidak menyediakan."""
    audio_path = "temp_audio.mp3"
    try:
        # Download audio saja menggunakan yt-dlp
        command = [
            sys.executable, "-m", "yt_dlp",
            "-x", "--audio-format", "mp3",
            "--cookies", COOKIES_PATH,
            "-o", audio_path,
            youtube_url
        ]
        subprocess.run(command, check=True)

        # Transkripsi menggunakan Whisper (Lokal & Gratis)
        model = whisper.load_model("base") # Bisa ganti "small" atau "medium"
        result = model.transcribe(audio_path)
        
        # Format ke bentuk [HH:MM:SS] text
        formatted = []
        for segment in result['segments']:
            start = int(segment['start'])
            text = segment['text']
            hours, rem = divmod(start, 3600)
            minutes, seconds = divmod(rem, 60)
            formatted.append(f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {text}")
            
        return "\n".join(formatted)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
def get_youtube_transcript(url: str):

    logger.info(f"📥 Menerima permintaan scraping untuk URL: {url}")

    try:

        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)

        if not video_id_match:

            return None, "Format URL YouTube tidak valid!"

       

        video_id = video_id_match.group(1)

        api_instance = youtube_transcript_api.YouTubeTranscriptApi()

        transcript_list_obj = api_instance.list(video_id)

       

        try:

            transcript_obj = transcript_list_obj.find_transcript(['en', 'id'])

        except Exception:

            transcript_obj = transcript_list_obj.find_best_transcript(['en', 'id'])



        transcript_list = transcript_obj.fetch()

        formatted_transcript = []

        for entry in transcript_list:

            start_time = entry.get('start', 0) if isinstance(entry, dict) else getattr(entry, 'start', 0)

            text = entry.get('text', '') if isinstance(entry, dict) else getattr(entry, 'text', '')

            hours = int(start_time // 3600)

            minutes = int((start_time % 3600) // 60)

            seconds = int(start_time % 60)

            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

            formatted_transcript.append(f"{timestamp} {text}")

           

        return "\n".join(formatted_transcript), None

    except Exception as e:

        logger.error(f"❌ Gagal mengambil transkrip: {str(e)}", exc_info=True)

        return None, f"Gagal mengambil transkrip otomatis: {str(e)}"

@app.get("/clip", response_class=HTMLResponse)
async def clip(request: Request):
    model_list = []
    for m in MODEL_OPTIONS:
        status = get_model_status(m["name"], API_URL, API_KEY)
        MODEL_STATUS_CACHE[m["name"]] = status
        model_list.append({"name": m["name"], "status": status})
        
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "models": model_list,
        "selected_model": "mistral-large"
    })

@app.post("/", response_class=HTMLResponse)
async def process_transcript(
    request: Request,
    youtube_url: str = Form(...), 
    selected_model: str = Form(...)
):
    error_msg = None
    results = None
    raw_response = None

    if not youtube_url.strip():
        error_msg = "Mohon masukkan URL YouTube terlebih dahulu!"
    else:
        # 1. Coba ambil transkrip asli
        transcript_text, yt_error = get_youtube_transcript(youtube_url)
        
        # 2. JIKA GAGAL, COBA GENERATE OTOMATIS MENGGUNAKAN WHISPER
        if yt_error:
            logger.info(f"Transkrip asli tidak ditemukan, mencoba generate via Whisper untuk: {youtube_url}")
            try:
                # Memanggil fungsi fallback (Anda harus menaruh fungsi ini di luar)
                transcript_text = generate_transcript_fallback(youtube_url)
                yt_error = None  # Berhasil generate, abaikan error YT
            except Exception as e:
                logger.error(f"Gagal generate transkrip otomatis: {str(e)}")
                error_msg = f"Tidak ada transkrip dan gagal generate otomatis: {str(e)}"

        if not error_msg:
            if not API_KEY or not API_URL:
                error_msg = "Konfigurasi API_KEY atau API_URL belum benar!"
            else:
                data = analyze_transcript_with_ai(selected_model, transcript_text, API_URL, API_KEY, "medium")
                if isinstance(data, dict) and "error" in data:
                    error_msg = data["error"]
                    raw_response = data.get("raw_response", None)
                    if "429" in error_msg: MODEL_STATUS_CACHE[selected_model] = "rate_limited"
                    elif "503" in error_msg or "500" in error_msg: MODEL_STATUS_CACHE[selected_model] = "down"
                else:
                    MODEL_STATUS_CACHE[selected_model] = "operational"
                    filtered_results = [clip for clip in data if isinstance(clip, dict) and int(clip.get('score', 0)) > 80]
                    if not filtered_results:
                        error_msg = "Tidak ada momen dengan skor di atas 80%."
                    else:
                        TARGET_DURATION = 60 
                        for index, clip in enumerate(filtered_results):
                            start = int(clip.get('start', 0))
                            end = start + TARGET_DURATION
                            title = clip.get('title', f'clip_{index}')
                            
                            clip['video_url'] = download_and_cut_video(youtube_url, start, end, title)
                            clip['end'] = end
                        results = filtered_results[:5]

    current_models = [{"name": m["name"], "status": MODEL_STATUS_CACHE.get(m["name"], "operational")} for m in MODEL_OPTIONS]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": current_models,
        "youtube_url": youtube_url,
        "selected_model": selected_model,
        "results": results,
        "error": error_msg,
        "raw_response": raw_response
    })
@app.get("/discover")
def discover(request: Request, q: str = None):
    videos = []

    if q:
        # Cukup panggil fungsi, hasil sudah pasti urut dari fungsi tsb
        videos = search_youtube_trending(q) 

        # HAPUS BARIS SORT DI SINI. Jangan lakukan sort lagi di sini.
        # Karena jika kamu melakukan sorted(videos, key=...) di sini,
        # dan 'views' adalah string (misal: "1.2M"), maka pengurutan akan rusak lagi!

    return templates.TemplateResponse(
        "discover.html",
        {"request": request, "videos": videos, "query": q}
    )

@app.get("/scripts", response_class=HTMLResponse)
async def scripts_get(request: Request):
    for model in MODEL_OPTIONS:
        model['status'] = get_model_status(model['name'], API_URL, API_KEY)

    return templates.TemplateResponse("scripts.html", {
        "request": request,
        "MODEL_OPTIONS": MODEL_OPTIONS,
        "OPENROUTER_MODELS": OPENROUTER_MODELS
    })


@app.post("/scripts", response_class=HTMLResponse)
async def scripts_post(request: Request, prompt: str = Form(...), model_name: str = Form(...)):

    script_result = "Gagal mendapatkan respons dari AI."

    try:
        if "/" in model_name:
            url = "https://openrouter.ai/api/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "ClipSpark"
            }

        else:
            url = API_URL

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Kamu adalah penulis naskah video narasi profesional. Buatkan naskah 1 menit tanpa instruksi tambahan."
                },
                {
                    "role": "user",
                    "content": f"Topik: {prompt}"
                }
            ]
        }

        response = requests.post(url, json=payload, headers=headers)

        data = response.json()
        raw_script = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        script_result = clean_script(raw_script)

    except Exception as e:
        script_result = f"Error: {str(e)}"

    return templates.TemplateResponse("scripts.html", {
        "request": request,
        "script": script_result,
        "MODEL_OPTIONS": MODEL_OPTIONS,
        "OPENROUTER_MODELS": OPENROUTER_MODELS
    })
# ==========================================
# ROUTE GENERATOR: POTONG VIDEO 180 DETIK
# ==========================================

def get_video_duration(file_path: str) -> float:
    """Mengambil durasi video menggunakan ffprobe"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

@app.post("/generate", response_class=HTMLResponse)
async def generate_clips(request: Request, youtube_url: str = Form(...)):
    unique_id = str(uuid.uuid4())[:8]
    download_path = f"static/clips/downloaded_{unique_id}.mp4"
    clips_result = []

    try:
        logger.info(f"Memulai proses download untuk URL: {youtube_url}")
        
        # 1. Download Video menggunakan yt-dlp secara instan
        # Menggunakan format mp4 terbaik yang sudah digabung (video+audio) up to 720p agar proses cepat
        ydl_cmd = [
            "yt-dlp",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
            "--merge-output-format", "mp4",
            "-o", download_path,
            youtube_url
        ]
        
        # Gunakan cookies jika filenya tersedia
        if os.path.exists(COOKIES_PATH):
            ydl_cmd.extend(["--cookies", COOKIES_PATH])
            
        subprocess.run(ydl_cmd, check=True)
        logger.info("Download YouTube selesai.")

        # 2. Ambil Total Durasi Video
        total_duration = get_video_duration(download_path)
        logger.info(f"Total durasi video: {total_duration} detik")

        # 3. Logika Pemotongan Per 180 Detik (3 Menit)
        interval = 180
        start_time = 0
        part_num = 1

        while start_time < total_duration:
            # Tentukan batas akhir (jika sisa video kurang dari 180 detik, ambil sisa maksimalnya)
            end_time = min(start_time + interval, total_duration)
            duration_segment = end_time - start_time

            output_filename = f"clip_{unique_id}_part_{part_num}.mp4"
            output_path = f"static/clips/{output_filename}"

            # Potong menggunakan FFmpeg dengan metode -c copy (Super Cepat & Tanpa Render Ulang)
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", download_path,
                "-c", "copy",
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Hitung format menit:detik untuk tampilan di HTML
            min_display = int(duration_segment // 60)
            sec_display = int(duration_segment % 60)
            duration_str = f"{min_display:02d}:{sec_display:02d}"

            # Simpan data klip untuk dikirim ke frontend template
            clips_result.append({
                "url": f"/{output_path}",
                "duration": duration_str
            })

            start_time += interval
            part_num += 1

        logger.info(f"Berhasil memotong video menjadi {len(clips_result)} klip.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Sistem gagal mengeksekusi perintah external: {e}")
    except Exception as e:
        logger.error(f"Terjadi error pada backend generator: {e}")
    finally:
        # Bersihkan file video asli yang di-download agar penyimpanan server tidak penuh
        if os.path.exists(download_path):
            os.remove(download_path)

    # Kembalikan ke halaman generator.html membawa data clips_result
    return templates.TemplateResponse(
        "generator.html", 
        {"request": request, "clips": clips_result}
    )
@app.post("/api/download-shorts")
async def process_shorts(url: str = Form(...)):
    if not url:
        raise HTTPException(status_code=400, detail="URL tidak boleh kosong")
        
    try:
        # Generate ID unik untuk nama file video agar tidak bentrok antar unduhan
        file_id = str(uuid.uuid4())
        output_template = f"static/clips/{file_id}.%(ext)s"
        
        # Konfigurasi yt-dlp untuk langsung download file mp4 utuh (video + audio) ke lokal disk
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Ekstrak metadata sekaligus lakukan proses download fisik ke server laptop
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'YouTube Short Video')
            
            # Cari ekstensi asli file yang terdownload (biasanya mp4)
            ext = info.get('ext', 'mp4')
            local_filename = f"{file_id}.{ext}"
            
            # Ini adalah URL statis lokal laptop yang bisa diakses oleh HP kamu
            local_video_url = f"/static/clips/{local_filename}"
            
            return JSONResponse({
                "status": "success",
                "title": video_title,
                "videoUrl": local_video_url  # Kita kirimkan path statis lokal kita
            })
            
    except Exception as e:
        logger.error(f"Error memproses YouTube Shorts: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Gagal mengunduh video ke server. Error: {str(e)}"
        }, status_code=500)

@app.get("/channels", response_class=HTMLResponse)
async def get_channels_page(request: Request):
    return templates.TemplateResponse("channels.html", {"request": request})

@app.post("/api/search-shorts")
async def search_youtube_shorts(data: ChannelSearchRequest):
    query = data.channel_name.strip()
    page = data.page if data.page > 0 else 1
    
    if not query:
        raise HTTPException(status_code=400, detail="Nama channel tidak boleh kosong")

    if not query.startswith('@') and not query.startswith('http'):
        query = f"@{query.replace(' ', '')}"
        
    url = f"https://www.youtube.com/{query}/shorts" if not query.startswith('http') else query

    limit = 10
    start_index = ((page - 1) * limit) + 1
    end_index = page * limit

    ydl_opts = {
        'playliststart': start_index,
        'playlistend': end_index,
        'quiet': True,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{CLIPS_DIR}/%(id)s.%(ext)s',
        'merge_output_format': 'mp4',
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            
            if not info_dict or 'entries' not in info_dict:
                return {"success": False, "message": "Tidak ada data ditemukan untuk halaman ini.", "shorts": []}

            channel_title = info_dict.get('title', query).replace(" - Shorts", "")

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT video_id FROM posted_videos")
            posted_ids = {row[0] for row in cursor.fetchall()}
            conn.close()

            extracted_shorts = []
            for entry in info_dict['entries']:
                if not entry:
                    continue
                
                video_id = entry.get('id')
                title = entry.get('title', 'Video Shorts')
                
                local_video_url = f"/static/clips/{video_id}.mp4"
                thumb_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                
                extracted_shorts.append({
                    "id": video_id,
                    "title": title,
                    "views": "Tersimpan Lokal", 
                    "duration": "Shorts",
                    "thumb": thumb_url,
                    "video_url": local_video_url,
                    "is_posted": video_id in posted_ids
                })

            if not extracted_shorts and page > 1:
                return {"success": False, "message": "Anda telah mencapai akhir halaman.", "shorts": []}

            return {
                "success": True,
                "current_page": page,
                "has_more": len(extracted_shorts) == limit,
                "channel": {
                    "name": channel_title,
                    "handle": query if query.startswith('@') else f"@{channel_title.lower().replace(' ', '')}",
                    "avatar_initial": channel_title[0].upper()
                },
                "shorts": extracted_shorts
            }

    except Exception as e:
        logger.error(f"Error downloading YouTube Shorts: {str(e)}")
        return {"success": False, "message": f"Gagal mengambil/mendownload data: {str(e)}", "shorts": []}

def crop_video_bottom(video_id: str):
    input_path = f"static/clips/{video_id}.mp4"
    if not os.path.exists(input_path) and os.path.exists(f"static/clips/{video_id}.mp5"):
        input_path = f"static/clips/{video_id}.mp5"
        
    output_path = f"static/clips/{video_id}_cropped.mp4"
    
    if os.path.exists(output_path):
        return output_path

    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', 'crop=iw:ih*0.95:0:0,pad=iw:ih:0:0:black',
        '-c:v', 'libx264', '-profile:v', 'main', '-level:v', '4.0', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', output_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    else:
        return input_path

@app.post("/api/post-facebook")
async def post_video_to_facebook(data: FacebookPostRequest):
    try:
        video_path = crop_video_bottom(data.video_id)
    except Exception as crop_error:
        return {"success": False, "message": f"Gagal memotong video: {str(crop_error)}"}
        
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="File video tidak ditemukan di server lokal.")
        
    try:
        file_size = os.path.getsize(video_path)
        
        init_url = f"https://graph.facebook.com/v25.0/{FB_PAGE_ID}/video_reels"
        init_payload = {
            'upload_phase': 'START',
            'access_token': FB_ACCESS_TOKEN
        }
        init_res = requests.post(init_url, data=init_payload).json()
        
        if "video_id" not in init_res:
            return {"success": False, "message": f"Gagal START: {init_res.get('error', {}).get('message', 'Unknown error')}"}
        
        fb_video_id = init_res["video_id"]
        fb_upload_url = init_res.get("upload_url", f"https://graph.facebook.com/v25.0/{fb_video_id}")
        
        headers = {
            'Authorization': f'Bearer {FB_ACCESS_TOKEN}',
            'offset': '0',
            'file_size': str(file_size)
        }
        
        with open(video_path, 'rb') as video_file:
            files = {
                'video_file_chunk': ('video.mp4', video_file, 'video/mp4')
            }
            upload_res = requests.post(fb_upload_url, headers=headers, files=files).json()
            
        if not upload_res.get("success") and "id" not in upload_res:
            return {"success": False, "message": f"Gagal Transfer File: {upload_res.get('error', {}).get('message', 'File ditolak oleh Facebook')}"}
            
        publish_url = f"https://graph.facebook.com/v25.0/{FB_PAGE_ID}/video_reels"
        publish_payload = {
            'upload_phase': 'FINISH',
            'video_id': fb_video_id,
            'video_state': 'PUBLISHED',
            'description': data.title,
            'access_token': FB_ACCESS_TOKEN
        }
        
        publish_res = requests.post(publish_url, data=publish_payload).json()
        
        if publish_res.get("success") or "id" in publish_res:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO posted_videos (video_id, fb_id) VALUES (?, ?)", (data.video_id, fb_video_id))
            conn.commit()
            conn.close()
            
            return {"success": True, "message": "Video berhasil diposting ke Facebook Reels!", "fb_id": fb_video_id}
        else:
            return {"success": False, "message": f"Gagal FINISH Publikasi: {publish_res.get('error', {}).get('message', 'Gagal mempublikasikan')}"}
            
    except Exception as e:
        return {"success": False, "message": f"Terjadi kesalahan sistem: {str(e)}"}