from datetime import datetime, timedelta
import json
import re
import requests
import yt_dlp

def format_views(num):
    if not num:
        return "0"

    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B".replace(".0", "")

    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M".replace(".0", "")

    if num >= 1_000:
        return f"{num/1_000:.1f}K".replace(".0", "")

    return str(num)
def get_model_status(model_name, api_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=3)
        if response.status_code == 200: return "operational"
        elif response.status_code == 429: return "rate_limited"
        else: return "down"
    except Exception: return "down"

def handle_ai_error(status_code, response_text):
    errors = {
        400: "400 Validation Error: Permintaan tidak valid.",
        401: "401 Unauthorized: API Key salah.",
        403: "403 Forbidden: Akses ditolak.",
        404: "404 Not Found: Model tidak ada.",
        413: "413 Bad Request: Input terlalu besar.",
        415: "415 Unsupported Media Type.",
        429: "429 Rate Limited: Kuota habis.",
        500: "500 Internal Error: Server bermasalah.",
        503: "503 Service Unavailable: Layanan sibuk."
    }
    return {"error": errors.get(status_code, f"Error {status_code}: {response_text}"), "raw_response": response_text}

# PERUBAHAN DI SINI: Tambahkan parameter reasoning_effort
def analyze_transcript_with_ai(model, transcript_text, api_url, api_key, reasoning_effort="medium"):
    try:
        truncated_transcript = transcript_text[:50000]
        prompt = f"""
Anda adalah pakar kurator video pendek dan ahli Copywriting Viral. Analisis transkrip berikut untuk mengekstrak momen-momen terbaik (skor > 80).

INSTRUKSI KRITIS:
1. Ekstrak momen menarik sebanyak mungkin, hingga maksimal 5 klip.
2. Respon HARUS berupa JSON array murni tanpa pembungkus markdown.
3. Teks di dalam JSON WAJIB Bahasa Indonesia.
4. Struktur "title": [Nama Tokoh]: [Hook] [Emoji] #Hashtag.

[
  {{ "title": "...", "start": 0, "end": 0, "score": 95, "reason": "..." }}
]

Transkrip: {truncated_transcript}
"""
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model, 
            "messages": [{"role": "user", "content": prompt}], 
            "temperature": 0.0, 
            "max_tokens": 5000,
            "reasoning_effort": reasoning_effort 
        }

        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return handle_ai_error(response.status_code, response.text)

        res_data = response.json()
        raw_content = res_data["choices"][0]["message"]["content"].strip()
        clean_content = re.sub(r"^```json\s*|^```\s*|```$", "", raw_content, flags=re.MULTILINE).strip()

        if not clean_content.endswith("]"):
            last_valid_object_end = clean_content.rfind("},")
            if last_valid_object_end != -1:
                clean_content = clean_content[: last_valid_object_end + 1] + "]"

        json_match = re.search(r"\[\s*\{.*\}\s*\]", clean_content, re.DOTALL)
        target_json = json_match.group(0) if json_match else clean_content

        try:
            results = json.loads(target_json)
            return results if isinstance(results, list) else {"error": "AI tidak mengembalikan format Array.", "raw_response": raw_content}
        except json.JSONDecodeError as je:
            return {"error": f"Gagal parsing JSON: {str(je)}", "raw_response": raw_content}

    except Exception as e:
        return {"error": f"Terjadi kesalahan sistem: {str(e)}"}
def search_youtube_trending(query):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(f"ytsearch20:{query}", download=False)

    videos = []
    for r in results.get("entries", []):
        raw_views = r.get("view_count") or 0
        video_id = r.get("id")

        videos.append({
            "title": r.get("title"),
            "url": f"https://youtube.com/watch?v={video_id}",
            "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            "views": format_views(raw_views),
            "raw_views": int(raw_views), # Pastikan integer
            "upload_date": r.get("upload_date")
        })

    # Cukup urutkan di sini SAJA
    videos.sort(key=lambda x: x["raw_views"], reverse=True)
    return videos