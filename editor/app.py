"""FastAPI backend of the annotation editor.

Run locally:   uvicorn editor.app:app --host 0.0.0.0 --port 8000
Environment:   EDITOR_DATA   storage root (default /data)
               EDITOR_TOKEN  shared secret; when set, every /api call needs
                             header X-Token or ?token=   (the page asks for it once)
               EDITOR_MODEL  optional ONNX segmenter for raster PDFs

The editor works in pixels of the base render (long side 2000 px) of the
plan region; ``scale.px_per_m`` converts to metres. Corrected plans are the
training set: GET /api/export gives image + JSON pairs.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from wallextractor.schema import WallPlan, validate
from .store import STATUSES, Store

DATA_ROOT = os.environ.get("EDITOR_DATA", "/data")
TOKEN = os.environ.get("EDITOR_TOKEN", "")
MODEL = os.environ.get("EDITOR_MODEL", "")
BASE_SIDE = 2000
EXPORT_SIDE = 1024

app = FastAPI(title="WallExtractor editor")
store = Store(DATA_ROOT)
STATIC = os.path.join(os.path.dirname(__file__), "static")


def auth(x_token: Optional[str] = Header(default=None), token: Optional[str] = Query(default=None)) -> None:
    if TOKEN and (x_token or token) != TOKEN:
        raise HTTPException(status_code=401, detail="token inválido")


# ---------------------------------------------------------------- extraction helpers
def _extract(pdf_path: str, page: int) -> tuple[dict, np.ndarray, dict]:
    """Run the extractor. Returns (plan dict, rgb of the base render, meta)."""
    import pymupdf

    from wallextractor.infer import extract_vector, load_segmenter, segment_image
    from wallextractor.pdf import render_page
    from wallextractor.vector_walls import _inside, page_segments, wall_pen_width
    from wallextractor.vectorize import mask_to_plan

    meta: Dict[str, Any] = {"primitives": []}
    res = extract_vector(pdf_path, page=page, max_side=BASE_SIDE)
    if res is not None:
        plan, rgb = res
        meta["notes"] = getattr(plan, "notes", [])
        # thick- and mid-pen segments in image pixels, for snapping in the editor
        with pymupdf.open(pdf_path) as doc:
            pg = doc[page - 1]
            segs = page_segments(pg)
            pen = wall_pen_width(segs)
            x0, y0, x1, y1 = plan.source.region_pt
            s = plan.source.px_per_pt or 1.0
            prims = []
            for sg in segs:
                if pen is None or sg.width < 0.2 * pen or not _inside(sg, (x0, y0, x1, y1)) or sg.length < 1.0:
                    continue
                prims.append([round((sg.a[0] - x0) * s, 1), round((sg.a[1] - y0) * s, 1),
                              round((sg.b[0] - x0) * s, 1), round((sg.b[1] - y0) * s, 1),
                              1 if abs(sg.width - pen) < 1e-3 else 0])
            meta["primitives"] = prims[:20000]
        return plan.to_dict(), rgb, meta
    # raster fallback
    rgb, px_per_pt = render_page(pdf_path, page=page, max_side=BASE_SIDE)
    if MODEL and os.path.isfile(MODEL):
        mask = segment_image(load_segmenter(MODEL), rgb, size=768)
        plan = mask_to_plan(mask, source_file=os.path.basename(pdf_path), page=page, kind="raster")
    else:
        plan = mask_to_plan(np.zeros(rgb.shape[:2], np.uint8), source_file=os.path.basename(pdf_path), page=page,
                            kind="raster")
        meta["notes"] = ["PDF sem vetores e sem modelo raster configurado (EDITOR_MODEL): anote do zero"]
    plan.image.dpi = round(72.0 * px_per_pt, 2)
    plan.source.px_per_pt = round(px_per_pt, 4)
    return plan.to_dict(), rgb, meta


# ---------------------------------------------------------------- API
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "plans": len(store.list())}


@app.get("/api/plans", dependencies=[Depends(auth)])
def list_plans(status: Optional[str] = None) -> List[dict]:
    return store.list(status)


@app.post("/api/plans", dependencies=[Depends(auth)])
async def create_plan(file: UploadFile = File(...), title: str = Form(""), page: int = Form(1)) -> dict:
    pid = store.new_id()
    pdf_path = store.pdf_path(pid)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        plan, rgb, meta = _extract(pdf_path, page)
    except Exception as exc:  # noqa: BLE001
        os.remove(pdf_path)
        raise HTTPException(status_code=400, detail=f"não consegui processar o PDF: {exc}") from exc
    Image.fromarray(rgb).save(store.render_path(pid))
    return store.create(pid, title or os.path.splitext(file.filename or pid)[0], file.filename or "", page, plan, meta)


@app.get("/api/plans/{pid}", dependencies=[Depends(auth)])
def get_plan(pid: str) -> dict:
    row = store.get(pid)
    if row is None:
        raise HTTPException(404)
    return row


@app.put("/api/plans/{pid}/annotation", dependencies=[Depends(auth)])
def save_annotation(pid: str, plan: Dict[str, Any]) -> dict:
    if store.get(pid) is None:
        raise HTTPException(404)
    problems = validate(plan)
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    store.save_correction(pid, plan)
    return {"ok": True}


@app.post("/api/plans/{pid}/status", dependencies=[Depends(auth)])
def set_status(pid: str, body: Dict[str, str]) -> dict:
    if store.get(pid) is None:
        raise HTTPException(404)
    st = body.get("status", "")
    if st not in STATUSES:
        raise HTTPException(422, f"status deve ser um de {STATUSES}")
    store.set_status(pid, st)
    return {"ok": True, "status": st}


@app.post("/api/plans/{pid}/title", dependencies=[Depends(auth)])
def set_title(pid: str, body: Dict[str, str]) -> dict:
    store.set_title(pid, body.get("title", "")[:200])
    return {"ok": True}


@app.delete("/api/plans/{pid}", dependencies=[Depends(auth)])
def delete_plan(pid: str) -> dict:
    store.delete(pid)
    return {"ok": True}


@app.get("/api/plans/{pid}/image", dependencies=[Depends(auth)])
def base_image(pid: str):
    p = store.render_path(pid)
    if not os.path.isfile(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/plans/{pid}/pdf", dependencies=[Depends(auth)])
def source_pdf(pid: str):
    """The uploaded PDF, to re-run the extractor offline against the corrections."""
    row = store.get(pid)
    if row is None or not os.path.isfile(store.pdf_path(pid)):
        raise HTTPException(404)
    return FileResponse(store.pdf_path(pid), media_type="application/pdf", filename=row.get("file") or f"{pid}.pdf")


@app.get("/api/plans/{pid}/tile", dependencies=[Depends(auth)])
def tile(pid: str, x: float, y: float, w: float, h: float, s: float = 2.0):
    """Re-render a window of the plan (base-image pixels) at ``s`` times the base resolution."""
    import pymupdf

    row = store.get(pid)
    if row is None:
        raise HTTPException(404)
    src = row["machine"]["source"]
    region = src.get("region_pt")
    ppp = src.get("px_per_pt") or 1.0
    s = max(0.5, min(s, 8.0))
    if w * s > 4096 or h * s > 4096:
        raise HTTPException(422, "tile grande demais")
    with pymupdf.open(store.pdf_path(pid)) as doc:
        pg = doc[row["page"] - 1]
        if region:
            clip = pymupdf.Rect(region[0] + x / ppp, region[1] + y / ppp, region[0] + (x + w) / ppp,
                                region[1] + (y + h) / ppp)
        else:
            clip = pymupdf.Rect(x / ppp, y / ppp, (x + w) / ppp, (y + h) / ppp)
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(ppp * s, ppp * s), clip=clip, alpha=False, colorspace=pymupdf.csRGB)
        png = pix.tobytes("png")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


def _export_pair(row: dict) -> tuple[bytes, dict]:
    """Image (long side EXPORT_SIDE) + plan JSON rescaled to it."""
    plan = row["corrected"] or row["machine"]
    img = Image.open(store.render_path(row["id"])).convert("RGB")
    W, H = img.size
    k = EXPORT_SIDE / max(W, H)
    img = img.resize((max(1, round(W * k)), max(1, round(H * k))), Image.LANCZOS)
    p = json.loads(json.dumps(plan))

    def sc(pt):
        return [round(pt[0] * k, 2), round(pt[1] * k, 2)]

    for wl in p.get("walls", []):
        wl["start"], wl["end"], wl["thickness"] = sc(wl["start"]), sc(wl["end"]), round(wl["thickness"] * k, 2)
        wl["polygon"] = [sc(q) for q in wl["polygon"]] if wl.get("polygon") else None
    for op in p.get("openings", []):
        op["start"], op["end"], op["width"] = sc(op["start"]), sc(op["end"]), round(op["width"] * k, 2)
        op["polygon"] = [sc(q) for q in op["polygon"]] if op.get("polygon") else None
    p["image"] = {"width": img.size[0], "height": img.size[1], "dpi": None}
    if p.get("scale", {}).get("px_per_m"):
        p["scale"]["px_per_m"] = round(p["scale"]["px_per_m"] * k, 4)
    p["export"] = {"plan_id": row["id"], "status": row["status"], "title": row["title"], "from": "corrected" if row["corrected"] else "machine"}
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), p


@app.get("/api/export", dependencies=[Depends(auth)])
def export(status: str = "corrected"):
    """Zip of <id>.png + <id>.json for every plan with the given status (training pairs)."""
    rows = [store.get(r["id"]) for r in store.list(status if status != "all" else None)]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    manifest = []
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            if row is None:
                continue
            png, p = _export_pair(row)
            z.writestr(f"{row['id']}.png", png)
            z.writestr(f"{row['id']}.json", json.dumps(p, ensure_ascii=False, indent=1))
            manifest.append({"id": row["id"], "title": row["title"], "status": row["status"], "updated": row["updated"],
                             "walls": len(p["walls"]), "openings": len(p["openings"])})
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    return FileResponse(tmp.name, media_type="application/zip", filename=f"wallextractor-{status}.zip")


@app.get("/api/export/{pid}", dependencies=[Depends(auth)])
def export_one(pid: str):
    row = store.get(pid)
    if row is None:
        raise HTTPException(404)
    _png, p = _export_pair(row)
    return JSONResponse(p)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
