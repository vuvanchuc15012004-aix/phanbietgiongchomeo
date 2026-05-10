import csv
import os
import tempfile
import traceback
import urllib.parse
import io
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from classifier import predict_pet

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"
HISTORY_CSV = LOGS_DIR / "history.csv"

STATIC_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

CSV_HEADER = ["timestamp", "filename", "top1_name_en", "top1_name_vi", "confidence", "wiki_found"]


def _ensure_csv_header():
    if not HISTORY_CSV.exists():
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def _append_log(filename: str, result: dict, wiki_found: bool):
    _ensure_csv_header()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            filename,
            result.get("top1_name_en", ""),
            result.get("top1_name_vi", ""),
            f"{result.get('top1_confidence', 0):.1f}",
            "true" if wiki_found else "false",
        ])


async def _fetch_wiki_summary(client: httpx.AsyncClient, url: str) -> dict:
    try:
        resp = await client.get(url, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "summary": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
        pass
    return None


async def _get_wiki_info(name_vi: str, name_en: str) -> dict:
    default = {"wiki_summary": "", "wiki_url": "", "wiki_found": False}
    async with httpx.AsyncClient() as client:
        vi_title = name_vi.replace("Cho ", "").replace("Meo ", "")
        vi_url = f"https://vi.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(vi_title)}"
        result = await _fetch_wiki_summary(client, vi_url)
        if result:
            result["wiki_found"] = True
            return result
        en_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(name_en)}"
        result = await _fetch_wiki_summary(client, en_url)
        if result:
            result["wiki_found"] = True
            return result
        return default


app = FastAPI(
    title="Pet Classifier API",
    description="Nhan dien giong cho meo tu anh bang EfficientNetB0",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<html><body><h1>Pet Classifier</h1><p>static/index.html not found.</p></body></html>"


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    tmp_path = None
    print(f"DEBUG: Received file: {file.filename}, content_type: {file.content_type}")
    try:
        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/jpg",
            "image/gif",
            "image/webp",
            "image/jfif",
            "application/octet-stream",
        }

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": "true", "message": "File qua lon. Toi da 10MB."},
            )

        try:
            Image.open(io.BytesIO(contents)).verify()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": "true", "message": f"File khong phai anh hop le: {e}"},
            )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(TEMP_DIR), suffix=Path(file.filename).suffix, delete=False
            ) as tmp:
                tmp.write(contents)
                tmp_path = Path(tmp.name)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "true", "message": f"Khong luu duoc file tam: {e}"},
            )

        try:
            result = predict_pet(str(tmp_path))

            if result.get("error") == "true":
                return JSONResponse(
                    status_code=422,
                    content={"error": "true", "message": result.get("message", "Loi xu ly anh.")},
                )

            wiki = await _get_wiki_info(
                result.get("top1_name_vi", ""), result.get("top1_name_en", "")
            )

            _append_log(file.filename or "unknown", result, wiki["wiki_found"])

            return {
                "top1_name_vi": result.get("top1_name_vi"),
                "top1_name_en": result.get("top1_name_en"),
                "top1_confidence": result.get("top1_confidence"),
                "alternatives": result.get("alternatives", []),
                "wiki_summary": wiki.get("wiki_summary", ""),
                "wiki_url": wiki.get("wiki_url", ""),
            }

        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.get("/history")
async def get_history(limit: int = 50):
    _ensure_csv_header()
    if not HISTORY_CSV.exists() or HISTORY_CSV.stat().st_size == 0:
        return {"records": []}
    records = []
    with open(HISTORY_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        records = rows[-limit:] if limit else rows
    return {"records": records}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "EfficientNetB0",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("Starting Pet Classifier server...")
    print("Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
