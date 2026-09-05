import os
import sys
import json
import logging
import threading
import hashlib
import uuid
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
import uvicorn

# 1. SETUP LOGGING & DIRECTORIES
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PACS_SERVER")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

for folder in [STORAGE_DIR, ASSETS_DIR, PUBLIC_DIR]:
    os.makedirs(folder, exist_ok=True)

# 2. MONGODB ATLAS CLOUD CONNECTION
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

# 3. FASTAPI SETUP
app = FastAPI(title="Medical PACS Server - Fast Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. DICOM ROUTER WITH STRICT AE TITLE MAPPING
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

# 5. PIXEL CACHING (Increased cache size for faster subsequent loads)
@lru_cache(maxsize=128)
def get_cached_pixel_bytes(filepath: str) -> bytes:
    ds = pydicom.dcmread(filepath, force=True)
    return ds.pixel_array.astype(np.uint16).tobytes()

@app.post("/api/auth/login")
async def login(payload: dict):
    username = payload.get("username", "").strip().lower()
    password = payload.get("password", "")
    user = users_col.find_one({"username": username, "password_hash": hash_password(password)})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    user.pop("password_hash", None)
    user.pop("_id", None)
    return {"status": "success", "user": user}

@app.post("/api/admin/register-doctor")
async def register_doctor(
    admin_username: str = Form(...),
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
def list_doctors(admin_username: str = Query(...)):
    admin = users_col.find_one({"username": admin_username.strip().lower(), "role": "ADMIN"})
    if not admin:
        raise HTTPException(status_code=403, detail="Unauthorized")
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
    filepath = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(filepath):
        candidates = [f for f in os.listdir(STORAGE_DIR) if f == filename or f.endswith(filename)]
        if candidates: filepath = os.path.join(STORAGE_DIR, candidates[0])
        else: return JSONResponse(status_code=404, content={"error": "File not found"})

    # OPTIMIZED: stop_before_pixels=True ensures metadata reads instantly without parsing 35MB pixel array
    ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    wc = getattr(ds, "WindowCenter", 2048)
    ww = getattr(ds, "WindowWidth", 4096)
    if isinstance(wc, pydicom.multival.MultiValue): wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue): ww = float(ww[0])

    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0]) if len(pixel_spacing) > 0 else 1.0
    col_spacing = float(pixel_spacing[1]) if len(pixel_spacing) > 1 else 1.0

    return {
        "width": int(ds.Columns),
        "height": int(ds.Rows),
        "wc": float(wc),
        "ww": float(ww),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
        "rowSpacing": row_spacing,
        "colSpacing": col_spacing
    }

@app.get("/api/raw-pixels/{filename:path}")
def raw_pixels(filename: str):
    filepath = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(filepath):
        candidates = [f for f in os.listdir(STORAGE_DIR) if f == filename or f.endswith(filename)]
        if candidates: filepath = os.path.join(STORAGE_DIR, candidates[0])
        else: return Response(status_code=404)

    # Uses LRU Cache + Browser Caching headers for instant subsequent loads
    data = get_cached_pixel_bytes(filepath)
    headers = {
        "Cache-Control": "public, max-age=86400",
        "Pragma": "cache",
    }
    return Response(content=data, media_type="application/octet-stream", headers=headers)

@app.post("/api/process-image")
async def process_image_backend(payload: dict):
    filename = payload.get("filename")
    brightness = float(payload.get("brightness", 0))
    contrast = float(payload.get("contrast", 1.0))
    smooth = float(payload.get("smooth", 0.0))
    sharp = float(payload.get("sharp", 0.0))
    gamma = float(payload.get("gamma", 1.0))

    filepath = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(filepath):
        candidates = [f for f in os.listdir(STORAGE_DIR) if f == filename or f.endswith(filename)]
        if candidates: filepath = os.path.join(STORAGE_DIR, candidates[0])
        else: raise HTTPException(status_code=404, detail="File not found")

    ds = pydicom.dcmread(filepath, force=True)
    arr = ds.pixel_array.astype(np.float32)

    mean_val = np.mean(arr)
    arr = (arr - mean_val) * contrast + mean_val + (brightness * 256.0)

    if smooth > 0.1:
        arr = gaussian_filter(arr, sigma=smooth)

    if sharp > 0:
        blurred = gaussian_filter(arr, sigma=1.0)
        arr = arr + (arr - blurred) * (sharp / 50.0)

    min_v, max_v = np.min(arr), np.max(arr)
    if max_v > min_v and gamma > 0:
        norm = (arr - min_v) / (max_v - min_v)
        arr = (np.power(norm, 1.0 / gamma) * (max_v - min_v)) + min_v

    arr = np.clip(arr, 0, 65535).astype(np.uint16)

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return Response(content=arr.tobytes(), media_type="application/octet-stream", headers=headers)

@app.post("/api/save-cropped-dicom")
async def save_cropped_dicom(payload: dict):
    original_filename = payload.get("original_filename")
    study_uid = payload.get("study_uid")
    crop_box = payload.get("crop_box")

    orig_path = os.path.join(STORAGE_DIR, original_filename)
    if not os.path.exists(orig_path):
        candidates = [f for f in os.listdir(STORAGE_DIR) if f == original_filename or f.endswith(original_filename)]
        if candidates:
            orig_path = os.path.join(STORAGE_DIR, candidates[0])
        else:
            raise HTTPException(status_code=404, detail="Original DICOM file not found")

    new_sop_uid = generate_uid()
    new_filename = f"crop_{uuid.uuid4().hex[:10]}.dcm"
    new_filepath = os.path.join(STORAGE_DIR, new_filename)

    ds = pydicom.dcmread(orig_path, force=True)
    arr = ds.pixel_array

    x1, y1, x2, y2 = [int(v) for v in crop_box]
    min_x, max_x = max(0, min(x1, x2)), min(arr.shape[1], max(x1, x2))
    min_y, max_y = max(0, min(y1, y2)), min(arr.shape[0], max(y1, y2))

    cropped_arr = arr[min_y:max_y, min_x:max_x]

    ds.Rows, ds.Columns = cropped_arr.shape
    ds.PixelData = cropped_arr.astype(arr.dtype).tobytes()
    ds.SOPInstanceUID = new_sop_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = f"{getattr(ds, 'SeriesDescription', '')} (Cropped ROI)"

    if hasattr(ds, "ImagePositionPatient"):
        try:
            spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
            ds.ImagePositionPatient[0] = float(ds.ImagePositionPatient[0]) + (min_x * float(spacing[1]))
            ds.ImagePositionPatient[1] = float(ds.ImagePositionPatient[1]) + (min_y * float(spacing[0]))
        except Exception:
            pass

    ds.save_as(new_filepath, write_like_original=False)

    file_info = {
        "filename": new_filename,
        "bodyPart": getattr(ds, "BodyPartExamined", "CROPPED"),
        "viewPosition": "CROP",
        "instanceNumber": int(getattr(ds, "InstanceNumber", 1)) + 100
    }

    studies_col.update_one(
        {"study_uid": study_uid},
        {"$addToSet": {"files": file_info}}
    )

    return {"status": "success", "new_filename": new_filename, "file_info": file_info}

@app.post("/api/report/save")
async def save_report(payload: dict):
    study_uid = payload.get("studyUid", "unknown")
    payload["saved_at"] = str(pydicom.valuerep.DT(None) if hasattr(pydicom, 'valuerep') else "now")
    reports_col.update_one({"study_uid": study_uid}, {"$set": payload}, upsert=True)
    return {"status": "success", "message": "Report saved successfully"}

@app.get("/api/report/{study_uid}")
def get_report(study_uid: str):
    rep = reports_col.find_one({"study_uid": study_uid}, {"_id": 0})
    return rep if rep else {"clinicalHistory": "", "findings": "", "impression": ""}

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")