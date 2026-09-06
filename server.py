from dotenv import load_dotenv

load_dotenv()

import os
import sys
import json
import logging
import threading
import hashlib
import uuid
import io
import gzip
import secrets
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import numpy as np
from scipy.ndimage import gaussian_filter
import pydicom
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    DeflatedExplicitVRLittleEndian,
    generate_uid
)
from pynetdicom import (
    AE,
    evt,
    AllStoragePresentationContexts,
    VerificationPresentationContexts
)
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
import requests
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PACS_SERVER")

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

for folder in [STORAGE_DIR, ASSETS_DIR, PUBLIC_DIR]:
    os.makedirs(folder, exist_ok=True)

MONGO_URI = "mongodb+srv://kr408368_db_user:6xIwZufgCpHADXXj@rabish.187hpqg.mongodb.net/?retryWrites=true&w=majority"

logger.info("Connecting to MongoDB Atlas Database...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["pacs_db"]

users_col = db["users"]
studies_col = db["studies"]
reports_col = db["reports"]

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_mongo_setup():
    try:
        users_col.create_index("username", unique=True)
        users_col.create_index("ae_title", unique=True)
        studies_col.create_index("study_uid", unique=True)
        studies_col.create_index("doctor_username")
        studies_col.create_index("received_ae_title")
        reports_col.create_index("study_uid", unique=True)

        if not users_col.find_one({"username": "admin"}):
            admin_doc = {
                "username": "admin",
                "password_hash": hash_password("admin123"),
                "role": "ADMIN",
                "ae_title": "MY_PACS",
                "doctor_name": "System Administrator",
                "degrees": "ADMIN",
                "reg_no": "SYS-01",
                "hospital_name": "CENTRAL PACS NETWORK",
                "hospital_info": "Master Hospital Network Hub",
                "logo_url": "",
                "sign_url": ""
            }
            users_col.insert_one(admin_doc)
            logger.info("Master Admin initialized: admin / admin123")
        else:
            logger.info("MongoDB Atlas Connected & Ready.")
    except Exception as e:
        logger.error(f"MongoDB Setup Error: {e}")

init_mongo_setup()

app = FastAPI(title="Medical PACS Server - Fast Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_decode_ae(val):
    if val is None:
        return "MY_PACS"
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='ignore').strip()
    return str(val).strip()

def extract_ae_titles_from_event(event):
    calling_ae = "UNKNOWN"
    called_ae = "MY_PACS"
    try:
        req = getattr(event.assoc, 'requestor', None)
        if req:
            calling_ae = safe_decode_ae(getattr(req, 'ae_title', None))
            prim = getattr(req, 'primitive', None)
            if prim and hasattr(prim, 'called_ae_title'):
                called_ae = safe_decode_ae(prim.called_ae_title)
        if called_ae == "MY_PACS" and hasattr(event.assoc, 'called_ae_title'):
            called_ae = safe_decode_ae(event.assoc.called_ae_title)
    except Exception as err:
        logger.warning(f"Error extracting AE metadata: {err}")
    return calling_ae.upper(), called_ae.upper()

def get_doctor_by_ae(called_ae: str):
    user = users_col.find_one({"ae_title": called_ae})
    if user:
        return user["username"], user["ae_title"]
    return "admin", "MY_PACS"

def handle_echo(event):
    calling_ae, called_ae = extract_ae_titles_from_event(event)
    logger.info(f"DICOM C-ECHO Verification from [{calling_ae}] to destination [{called_ae}]")
    return 0x0000

def handle_store(event):
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta

        calling_ae, called_ae = extract_ae_titles_from_event(event)
        target_doctor, matched_ae = get_doctor_by_ae(called_ae)

        sop_uid = getattr(ds, "SOPInstanceUID", None)
        if not sop_uid:
            sop_uid = f"gen_{uuid.uuid4().hex}"

        study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN_STUDY"))
        filepath = os.path.join(STORAGE_DIR, f"{sop_uid}.dcm")
        ds.save_as(filepath, write_like_original=False)

        p_name = str(getattr(ds, "PatientName", "ANONYMOUS")).replace("^", " ")
        p_id = str(getattr(ds, "PatientID", "UNKNOWN_ID"))
        p_age = str(getattr(ds, "PatientAge", "N/A"))
        p_sex = str(getattr(ds, "PatientSex", "N/A"))
        s_date = str(getattr(ds, "StudyDate", "N/A"))
        if len(s_date) == 8:
            s_date = f"{s_date[:4]}-{s_date[4:6]}-{s_date[6:]}"
        modality = str(getattr(ds, "Modality", "DX"))
        body_part = str(getattr(ds, "BodyPartExamined", "CHEST"))
        view_pos = str(getattr(ds, "ViewPosition", "AP/PA"))
        inst_num = int(getattr(ds, "InstanceNumber", 1))

        file_info = {
            "filename": f"{sop_uid}.dcm",
            "bodyPart": body_part,
            "viewPosition": view_pos,
            "instanceNumber": inst_num
        }

        studies_col.update_one(
            {"study_uid": study_uid},
            {
                "$set": {
                    "study_uid": study_uid,
                    "doctor_username": target_doctor,
                    "received_ae_title": matched_ae,
                    "calling_ae_title": calling_ae,
                    "patientName": p_name,
                    "patientId": p_id,
                    "patientAge": p_age,
                    "patientSex": p_sex,
                    "studyDate": s_date,
                    "modality": modality,
                    "bodyPart": body_part,
                },
                "$addToSet": {"files": file_info}
            },
            upsert=True
        )
        logger.info(f"DICOM received [{study_uid}] -> Doctor: [{target_doctor}] | AE: [{matched_ae}]")
        return 0x0000
    except Exception as e:
        logger.error(f"Error in C-STORE execution: {e}")
        return 0xC000

def start_dicom_listener():
    ae = AE(ae_title=b"MY_PACS")
    ae.require_called_aet = []
    ae.supported_contexts = AllStoragePresentationContexts + VerificationPresentationContexts
    for cx in ae.supported_contexts:
        cx.transfer_syntax = [
            ExplicitVRLittleEndian, ImplicitVRLittleEndian, DeflatedExplicitVRLittleEndian,
            '1.2.840.10008.1.2.4.50', '1.2.840.10008.1.2.4.51',
            '1.2.840.10008.1.2.4.57', '1.2.840.10008.1.2.4.70',
            '1.2.840.10008.1.2.4.80', '1.2.840.10008.1.2.4.90',
            '1.2.840.10008.1.2.4.91', '1.2.840.10008.1.2.5'
        ]
    handlers = [(evt.EVT_C_ECHO, handle_echo), (evt.EVT_C_STORE, handle_store)]
    try:
        ae.start_server(("0.0.0.0", 11112), block=True, evt_handlers=handlers)
    except Exception as err:
        logger.error(f"DICOM Listener Server Error: {err}")

threading.Thread(target=start_dicom_listener, daemon=True).start()

@lru_cache(maxsize=12)
def get_cached_pixel_array(filepath: str, mtime_ns: int):
    ds = pydicom.dcmread(filepath, force=True)
    return ds.pixel_array.astype(np.uint16, copy=False)

def get_pixel_array(filepath: str):
    try:
        mtime_ns = os.stat(filepath).st_mtime_ns
    except OSError:
        mtime_ns = 0
    return get_cached_pixel_array(filepath, mtime_ns)

@lru_cache(maxsize=12)
def get_cached_pixel_bytes(filepath: str, mtime_ns: int) -> bytes:
    return np.ascontiguousarray(get_cached_pixel_array(filepath, mtime_ns), dtype=np.uint16).tobytes()

def resolve_storage_path(filename: str):
    safe_name = os.path.basename(filename)
    filepath = os.path.join(STORAGE_DIR, safe_name)
    if os.path.isfile(filepath):
        return filepath
    candidates = [f for f in os.listdir(STORAGE_DIR) if f == safe_name or f.endswith(safe_name)]
    return os.path.join(STORAGE_DIR, candidates[0]) if candidates else None

def window_to_uint8(arr: np.ndarray, wc: float, ww: float, invert: bool = False) -> np.ndarray:
    low = float(wc) - float(ww) / 2.0
    high = float(wc) + float(ww) / 2.0
    if high <= low:
        high = low + 1.0
    out = np.clip((arr.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
    out = out.astype(np.uint8)
    if invert:
        out = 255 - out
    return out

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
OTP_TTL_SECONDS = int(os.getenv("ADMIN_OTP_TTL_SECONDS", "300"))

otp_col = db["admin_otps"]
try:
    otp_col.create_index("expires_at", expireAfterSeconds=0)
    otp_col.create_index("request_id", unique=True)
except Exception as e:
    logger.warning(f"OTP index setup warning: {e}")

def send_admin_otp(email_to: str, otp: str):
    if not SMTP_USER or not SMTP_APP_PASSWORD or not email_to:
        raise RuntimeError("Gmail OTP is not configured.")
    msg = EmailMessage()
    msg["Subject"] = "PACS Administrator OTP"
    msg["From"] = SMTP_USER
    msg["To"] = email_to
    msg.set_content(
        f"Your PACS Administrator verification code is: {otp}\n\n"
        f"This code expires in {OTP_TTL_SECONDS // 60} minutes."
    )
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
        smtp.login(SMTP_USER, SMTP_APP_PASSWORD)
        smtp.send_message(msg)

@app.post("/api/auth/request-admin-otp")
async def request_admin_otp(payload: dict):
    username = payload.get("username", "").strip().lower()
    password = payload.get("password", "")
    user = users_col.find_one({"username": username, "password_hash": hash_password(password), "role": "ADMIN"})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid administrator credentials.")

    email_to = (user.get("email") or ADMIN_EMAIL or "").strip()
    if not email_to:
        raise HTTPException(status_code=503, detail="Administrator Gmail is not configured.")

    otp = f"{secrets.randbelow(1000000):06d}"
    request_id = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=OTP_TTL_SECONDS)
    otp_col.delete_many({"username": username})
    otp_col.insert_one({
        "request_id": request_id,
        "username": username,
        "otp_hash": hash_password(otp),
        "expires_at": expires_at,
        "attempts": 0
    })
    try:
        send_admin_otp(email_to, otp)
    except Exception as e:
        otp_col.delete_one({"request_id": request_id})
        raise HTTPException(status_code=502, detail=f"Unable to send OTP: {e}")

    masked = email_to[:2] + "***" + email_to[email_to.find("@"):] if "@" in email_to else "email"
    return {"status": "otp_sent", "request_id": request_id, "email": masked, "expires_in": OTP_TTL_SECONDS}

@app.post("/api/auth/verify-admin-otp")
async def verify_admin_otp(payload: dict):
    request_id = str(payload.get("request_id", "")).strip()
    otp = str(payload.get("otp", "")).strip()
    if not request_id or not otp or not otp.isdigit() or len(otp) != 6:
        raise HTTPException(status_code=400, detail="Enter the 6-digit OTP.")

    rec = otp_col.find_one({"request_id": request_id})
    if not rec:
        raise HTTPException(status_code=401, detail="OTP expired or invalid.")
    if rec.get("attempts", 0) >= 5:
        otp_col.delete_one({"request_id": request_id})
        raise HTTPException(status_code=429, detail="Too many incorrect attempts.")
    if rec["otp_hash"] != hash_password(otp):
        otp_col.update_one({"request_id": request_id}, {"$inc": {"attempts": 1}})
        raise HTTPException(status_code=401, detail="Incorrect OTP.")

    session_token = secrets.token_urlsafe(32)
    otp_col.delete_one({"request_id": request_id})
    otp_col.insert_one({
        "request_id": "session:" + session_token,
        "username": rec["username"],
        "verified": True,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=3600)
    })
    return {"status": "verified", "otp_token": session_token, "username": rec["username"]}

def require_admin_otp(admin_username: str, otp_token: str):
    admin_username = (admin_username or "").strip().lower()
    if not otp_token:
        raise HTTPException(status_code=403, detail="Administrator OTP verification required.")
    rec = otp_col.find_one({"request_id": "session:" + otp_token, "username": admin_username, "verified": True})
    if not rec:
        raise HTTPException(status_code=403, detail="Administrator OTP session expired.")
    return True

@app.post("/api/auth/login")
async def login(payload: dict):
    username = payload.get("username", "").strip().lower()
    password = payload.get("password", "")
    user = users_col.find_one({"username": username, "password_hash": hash_password(password)})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    user.pop("password_hash", None)
    user.pop("_id", None)
    if user.get("role") == "ADMIN":
        user.pop("email", None)
        return {"status": "success", "requires_admin_otp": True, "user": user}
    return {"status": "success", "user": user}

@app.post("/api/admin/register-doctor")
async def register_doctor(
    admin_username: str = Form(...),
    otp_token: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    ae_title: str = Form(...),
    doctor_name: str = Form(...),
    degrees: str = Form(""),
    reg_no: str = Form(""),
    hospital_name: str = Form(""),
    hospital_info: str = Form(""),
    logo: UploadFile = File(None),
    signature: UploadFile = File(None)
):
    admin_user = users_col.find_one({"username": admin_username.strip().lower(), "role": "ADMIN"})
    if not admin_user:
        raise HTTPException(status_code=403, detail="Unauthorized access.")
    require_admin_otp(admin_username, otp_token)

    clean_u = username.strip().lower()
    clean_ae = ae_title.strip().upper()

    if users_col.find_one({"$or": [{"username": clean_u}, {"ae_title": clean_ae}]}):
        raise HTTPException(status_code=400, detail="Username or AE Title already exists.")

    logo_url = ""
    if logo and logo.filename:
        ext = logo.filename.split(".")[-1]
        logo_fn = f"logo_{clean_u}.{ext}"
        with open(os.path.join(ASSETS_DIR, logo_fn), "wb") as f:
            f.write(await logo.read())
        logo_url = f"/assets/{logo_fn}"

    sign_url = ""
    if signature and signature.filename:
        ext = signature.filename.split(".")[-1]
        sign_fn = f"sign_{clean_u}.{ext}"
        with open(os.path.join(ASSETS_DIR, sign_fn), "wb") as f:
            f.write(await signature.read())
        sign_url = f"/assets/{sign_fn}"

    new_doc = {
        "username": clean_u,
        "password_hash": hash_password(password),
        "role": "DOCTOR",
        "ae_title": clean_ae,
        "doctor_name": doctor_name,
        "degrees": degrees,
        "reg_no": reg_no,
        "hospital_name": hospital_name,
        "hospital_info": hospital_info,
        "logo_url": logo_url,
        "sign_url": sign_url
    }
    users_col.insert_one(new_doc)
    return {"status": "success", "message": f"Doctor '{doctor_name}' registered successfully."}

@app.get("/api/admin/list-doctors")
def list_doctors(admin_username: str = Query(...), otp_token: str = Query(...)):
    admin = users_col.find_one({"username": admin_username.strip().lower(), "role": "ADMIN"})
    if not admin:
        raise HTTPException(status_code=403, detail="Unauthorized")
    require_admin_otp(admin_username, otp_token)
    doctors = list(users_col.find({"role": "DOCTOR"}, {"_id": 0, "password_hash": 0}))
    return doctors

@app.get("/api/studies-meta")
def get_studies_meta(doctor_username: str = Query(None)):
    if not doctor_username:
        return []
    u = users_col.find_one({"username": doctor_username.strip().lower()})
    if not u:
        return []
    if u.get("role") == "ADMIN":
        query = {}
    else:
        query = {"doctor_username": u["username"]}
    studies = list(studies_col.find(query, {"_id": 0}))
    for s in studies:
        s["imageCount"] = len(s.get("files", []))
        s["files"].sort(key=lambda x: x.get("instanceNumber", 1))
    return studies

@app.get("/api/sync-studies")
def sync_storage_files():
    count = 0
    if not os.path.exists(STORAGE_DIR):
        return {"status": "error", "message": "Storage dir not found"}
    for fname in os.listdir(STORAGE_DIR):
        if not fname.lower().endswith(".dcm"): continue
        fpath = os.path.join(STORAGE_DIR, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN_STUDY"))
            p_name = str(getattr(ds, "PatientName", "ANONYMOUS")).replace("^", " ")
            p_id = str(getattr(ds, "PatientID", "UNKNOWN_ID"))
            p_age = str(getattr(ds, "PatientAge", "N/A"))
            p_sex = str(getattr(ds, "PatientSex", "N/A"))
            s_date = str(getattr(ds, "StudyDate", "N/A"))
            if len(s_date) == 8: s_date = f"{s_date[:4]}-{s_date[4:6]}-{s_date[6:]}"
            modality = str(getattr(ds, "Modality", "DX"))
            body_part = str(getattr(ds, "BodyPartExamined", "CHEST"))
            view_pos = str(getattr(ds, "ViewPosition", "AP/PA"))
            inst_num = int(getattr(ds, "InstanceNumber", 1))

            file_info = {
                "filename": fname,
                "bodyPart": body_part,
                "viewPosition": view_pos,
                "instanceNumber": inst_num
            }

            studies_col.update_one(
                {"study_uid": study_uid},
                {
                    "$set": {
                        "study_uid": study_uid,
                        "doctor_username": "admin",
                        "patientName": p_name,
                        "patientId": p_id,
                        "patientAge": p_age,
                        "patientSex": p_sex,
                        "studyDate": s_date,
                        "modality": modality,
                        "bodyPart": body_part,
                    },
                    "$addToSet": {"files": file_info}
                },
                upsert=True
            )
            count += 1
        except Exception:
            pass
    return {"status": "success", "synced_files": count}

@app.get("/api/decode/{filename:path}")
def decode_dicom(filename: str):
    filepath = resolve_storage_path(filename)
    if not filepath:
        return JSONResponse(status_code=404, content={"error": "File not found"})

    ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    wc = getattr(ds, "WindowCenter", 2048)
    ww = getattr(ds, "WindowWidth", 4096)
    if isinstance(wc, pydicom.multival.MultiValue): wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue): ww = float(ww[0])
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0]) if len(pixel_spacing) > 0 else 1.0
    col_spacing = float(pixel_spacing[1]) if len(pixel_spacing) > 1 else 1.0
    return {
        "width": int(ds.Columns), "height": int(ds.Rows), "wc": float(wc), "ww": float(ww),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
        "rowSpacing": row_spacing, "colSpacing": col_spacing
    }

@app.get("/api/preview/{filename:path}")
def preview_dicom(filename: str):
    filepath = resolve_storage_path(filename)
    if not filepath:
        return Response(status_code=404)
    try:
        from PIL import Image
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
        wc = getattr(ds, "WindowCenter", 2048)
        ww = getattr(ds, "WindowWidth", 4096)
        if isinstance(wc, pydicom.multival.MultiValue): wc = float(wc[0])
        if isinstance(ww, pydicom.multival.MultiValue): ww = float(ww[0])
        arr = get_pixel_array(filepath)
        img8 = window_to_uint8(arr, float(wc), float(ww), str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1")
        img = Image.fromarray(img8, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400, immutable"})
    except Exception as e:
        logger.error(f"Preview generation failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail="Preview generation failed")

@app.get("/api/raw-pixels/{filename:path}")
def raw_pixels(filename: str):
    filepath = resolve_storage_path(filename)
    if not filepath:
        return Response(status_code=404)
    mtime_ns = os.stat(filepath).st_mtime_ns
    data = get_cached_pixel_bytes(filepath, mtime_ns)
    compressed = gzip.compress(data, compresslevel=5, mtime=0)
    headers = {
        "Cache-Control": "public, max-age=86400, immutable",
        "Content-Encoding": "gzip",
        "Vary": "Accept-Encoding",
        "X-PACS-Pixel-Bytes": str(len(data)),
        "X-PACS-Transfer-Bytes": str(len(compressed))
    }
    return Response(content=compressed, media_type="application/octet-stream", headers=headers)

CENTRAL_WHATSAPP_TOKEN = "APKA_PERMANENT_YA_TEMPORARY_TOKEN"
CENTRAL_PHONE_NUMBER_ID = "APKA_CENTRAL_PHONE_NUMBER_ID"

@app.post("/api/send-whatsapp-media")
async def send_whatsapp_media(payload: dict):
    recipient_phone = payload.get("phone")
    media_url = payload.get("media_url")
    caption_text = payload.get("caption", "Aapki PACS Medical Report / Image.")
    
    if not recipient_phone or not media_url:
        raise HTTPException(status_code=400, detail="Phone number and media URL are required.")

    url = f"https://graph.facebook.com/v17.0/{CENTRAL_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {CENTRAL_WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "type": "image",
        "image": {
            "link": media_url,
            "caption": caption_text
        }
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return {"status": "success", "response": response.json()}
    else:
        raise HTTPException(status_code=500, detail=f"WhatsApp Media API Error: {response.text}")

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")