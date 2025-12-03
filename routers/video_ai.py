import io
import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db_engine import get_db, SwapHistoryModel
from rate_limit import check_rate_limit

router = APIRouter(prefix="/video-ai", tags=["FaceSwap"])

# These will be set by main.py
face_app = None
face_swapper = None


def inject_models(app_model, swap_model):
    global face_app, face_swapper
    face_app = app_model
    face_swapper = swap_model


def _ensure_models_ready():
    if face_app is None or face_swapper is None:
        raise HTTPException(503, "FaceSwap models are not ready.")


@router.post("/swap")
async def swap_face(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _ensure_models_ready()

    src_bytes = await source.read()
    tgt_bytes = await target.read()

    src = cv2.imdecode(np.frombuffer(src_bytes, np.uint8), cv2.IMREAD_COLOR)
    tgt = cv2.imdecode(np.frombuffer(tgt_bytes, np.uint8), cv2.IMREAD_COLOR)

    if src is None or tgt is None:
        raise HTTPException(400, "Cannot decode uploaded images")

    src_faces = face_app.get(src)
    tgt_faces = face_app.get(tgt)

    if not src_faces or not tgt_faces:
        raise HTTPException(400, "No faces detected")

    result = face_swapper.get(
        tgt,
        tgt_faces[0],
        src_faces[0],
        paste_back=True,
    )

    ok, out_buf = cv2.imencode(".jpg", result)
    if not ok:
        raise HTTPException(500, "Encode failed")

    return StreamingResponse(
        io.BytesIO(out_buf.tobytes()),
        media_type="image/jpeg",
    )
