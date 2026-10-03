"""FastAPI backend of the annotation editor.

Run locally:   uvicorn editor.app:app --host 0.0.0.0 --port 8000
Environment:   EDITOR_DATA   storage root (default /data)
               EDITOR_TOKEN  shared secret; when set, every /api call needs
                             header X-Token or ?token=   (the page asks for it once)
               EDITOR_MODEL  ONNX segmenter for raster PDFs (default <EDITOR_DATA>/model.onnx,
                             uploaded with PUT /api/model)

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
MODEL = os.environ.get("EDITOR_MODEL", "") or os.path.join(DATA_ROOT, "model.onnx")
_segmenter: Dict[str, Any] = {"mtime": None, "obj": None}


def _get_segmenter():
    """The raster model, loaded once and reloaded when the file changes (a new upload)."""
    from wallextractor.infer import load_segmenter

    mtime = os.path.getmtime(MODEL)
    if _segmenter["mtime"] != mtime:
        _segmenter["obj"], _segmenter["mtime"] = load_segmenter(MODEL), mtime
    return _segmenter["obj"]
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

    from wallextractor.infer import extract_vector, segment_image
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
        mask = segment_image(_get_segmenter(), rgb, size=768)
        plan = mask_to_plan(mask, source_file=os.path.basename(pdf_path), page=page, kind="raster")
    else:
        plan = mask_to_plan(np.zeros(rgb.shape[:2], np.uint8), source_file=os.path.basename(pdf_path), page=page,
                            kind="raster")
        meta["notes"] = ["PDF sem vetores e sem modelo raster configurado (PUT /api/model): anote do zero"]
    plan.image.dpi = round(72.0 * px_per_pt, 2)
    plan.source.px_per_pt = round(px_per_pt, 4)
    return plan.to_dict(), rgb, meta


# ---------------------------------------------------------------- API
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "plans": len(store.list()), "model": os.path.isfile(MODEL)}


def _model_info() -> dict:
    import hashlib

    if not os.path.isfile(MODEL):
        return {"present": False, "path": MODEL}
    h = hashlib.sha256()
    with open(MODEL, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"present": True, "path": MODEL, "bytes": os.path.getsize(MODEL), "sha256": h.hexdigest(),
            "modified": os.path.getmtime(MODEL)}


@app.get("/api/model", dependencies=[Depends(auth)])
def get_model() -> dict:
    return _model_info()


@app.put("/api/model", dependencies=[Depends(auth)])
async def put_model(file: UploadFile = File(...)) -> dict:
    """Replace the raster segmenter (ONNX). Loaded and checked before it replaces the current one."""
    import onnxruntime as ort

    os.makedirs(os.path.dirname(MODEL) or ".", exist_ok=True)
    tmp = MODEL + ".upload"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
    except Exception as exc:  # noqa: BLE001
        os.remove(tmp)
        raise HTTPException(status_code=400, detail=f"ONNX inválido: {exc}") from exc
    os.replace(tmp, MODEL)
    return {**_model_info(), "input_shape": [str(d) for d in shape]}


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


AI_README = """# Planta para correção com IA (Claude, ChatGPT ou outro modelo com visão)

Arquivos:
- `planta.png`: a planta como o editor mostra ({w} x {h} px).
- `planta_numerada.png`: a mesma imagem com a anotação atual por cima: paredes em vermelho
  (w1, w2, ...), portas em verde e janelas em azul (o1, o2, ...).
- `planta.json`: a anotação atual (saída da máquina ou a última correção salva).
- `conferir.py`: desenha um `correcao.json` sobre `planta.png` (`python conferir.py correcao.json`
  gera `conferencia.png`; precisa só do Pillow).
- `PROMPT.txt`: o pedido, pronto para colar no chat junto com os arquivos.

Coordenadas em pixels de `planta.png`, origem no canto superior esquerdo, x para a direita, y para
baixo. Escala: {scale}.

A anotação atual é um rascunho automático: pode ter paredes picotadas em muitos pedaços, paredes
que não existem, aberturas faltando ou com o tipo errado.

## O que anotar

- Paredes: só as paredes da edificação cortadas pela planta (alvenaria, desenhadas com duas linhas
  paralelas, às vezes com preenchimento ou hachura). Não são paredes: projeção da cobertura, linhas
  de cota e de chamada, textos, móveis, louças, piso/azulejo, escadas, eixos e o contorno de áreas
  abertas (garagem coberta, varanda) quando não há parede desenhada.
- Portas: folha (linha) com o arco de abertura, ou porta de correr. Janelas: caixilho (linhas finas
  dentro da espessura da parede). Etiquetas ajudam: P1, P2… são portas; J1, J2… são janelas.

## Formato de `planta.json` e da resposta

- `walls`: `{{"id": "w1", "start": [x, y], "end": [x, y], "thickness": px}}`. A linha vai pelo eixo
  da parede (no meio entre as duas faces) e `thickness` é a distância entre as faces, em pixels.
  Paredes que se encontram terminam no cruzamento dos eixos. Em cada vão de porta ou janela a parede
  é interrompida: um trecho termina numa borda do vão e outro começa na outra.
- `openings`: `{{"id": "o1", "type": "door" | "window", "start": [x, y], "end": [x, y],
  "wall_id": "w3"}}`, sobre o eixo da parede, de uma borda do vão até a outra (a largura da abertura);
  `wall_id` é um dos trechos vizinhos (pode ficar de fora: o editor liga pela geometria).

Exemplo de resposta (uma parede com uma porta no meio):

```json
{{"plan_id": "{pid}", "image": {{"width": {w}, "height": {h}}},
 "walls": [{{"id": "w1", "start": [100, 200], "end": [300, 200], "thickness": 12}},
           {{"id": "w2", "start": [380, 200], "end": [600, 200], "thickness": 12}}],
 "openings": [{{"id": "o1", "type": "door", "start": [300, 200], "end": [380, 200], "wall_id": "w1"}}],
 "notas": "dúvidas, se houver"}}
```

## Pedido

Compare a anotação com `planta.png`, corrija paredes e aberturas (posição, pontas, espessura, tipo,
o que falta e o que sobra) e devolva `correcao.json` com a lista COMPLETA de `walls` e `openings`
(não só as mudanças), JSON válido, com `plan_id` = "{pid}" e `image` = {{"width": {w}, "height": {h}}}.
Coordenadas sempre em pixels de `planta.png`, mesmo que a imagem tenha sido reduzida para você ver.
Se puder rodar código: comece do `planta.json` (juntar trechos alinhados ajuda), rode
`python conferir.py correcao.json`, olhe `conferencia.png` e ajuste antes de entregar. Se não puder
gerar arquivo, responda só com o JSON num bloco de código: o editor aceita o texto colado.
No editor, "Importar correção" (arquivo) ou "Colar resposta" (texto) carrega o resultado; nada é
marcado como corrigido até o revisor conferir.
"""

AI_PROMPT = """Anexei os arquivos de uma planta baixa (planta.png, planta_numerada.png, planta.json, LEIA-ME.md
e conferir.py; se vierem num .zip, descompacte). Siga o LEIA-ME.md: corrija as paredes, portas e
janelas de planta.json comparando com planta.png e me devolva correcao.json com a lista completa, em
pixels de planta.png ({w} x {h}), plan_id "{pid}". Se puder rodar código, confira com conferir.py antes
de entregar. Se não puder gerar arquivo, responda só com o JSON num bloco de código.
"""

AI_CHECK = '''# Desenha correcao.json sobre planta.png: paredes em vermelho, portas em verde, janelas em azul.
# Uso: python conferir.py correcao.json  ->  conferencia.png
import json, sys
from PIL import Image, ImageDraw

path = sys.argv[1] if len(sys.argv) > 1 else "correcao.json"
plan = json.load(open(path, encoding="utf-8"))
img = Image.open("planta.png").convert("RGB")
iw = (plan.get("image") or {}).get("width") or img.width
if iw != img.width:
    print(f"aviso: image.width {iw} diferente de planta.png ({img.width}); as coordenadas devem ser de planta.png")
img = Image.blend(img, Image.new("RGB", img.size, "white"), 0.35)
d = ImageDraw.Draw(img)
lw = max(2, round(max(img.size) / 500))
for w in plan.get("walls", []):
    d.line([tuple(w["start"]), tuple(w["end"])], fill=(200, 30, 30), width=lw)
for o in plan.get("openings", []):
    col = (20, 140, 70) if o.get("type") == "door" else (40, 90, 220)
    d.line([tuple(o["start"]), tuple(o["end"])], fill=col, width=lw * 3)
for x in plan.get("walls", []) + plan.get("openings", []):
    d.text(((x["start"][0] + x["end"][0]) / 2 + 4, (x["start"][1] + x["end"][1]) / 2 + 4), str(x.get("id", "")), fill=(0, 0, 0))
img.save("conferencia.png")
print(f"conferencia.png: {len(plan.get('walls', []))} paredes, {len(plan.get('openings', []))} aberturas")
'''


def _ai_plan(plan: dict, pid: str, title: str, size) -> dict:
    """The annotation trimmed to what a model needs to read and write back, coordinates to 0.1 px."""
    def r(p):
        return [round(p[0], 1), round(p[1], 1)]

    openings = []
    for o in plan.get("openings", []):
        item = {"id": o["id"], "type": o.get("type"), "start": r(o["start"]), "end": r(o["end"]),
                "wall_id": o.get("wall_id"), "code": o.get("code")}
        openings.append({k: v for k, v in item.items() if v is not None})
    return {"plan_id": pid, "title": title, "image": {"width": size[0], "height": size[1]},
            "scale": plan.get("scale") or {"px_per_m": None},
            "walls": [{"id": w["id"], "start": r(w["start"]), "end": r(w["end"]), "thickness": round(w["thickness"], 1)}
                      for w in plan.get("walls", [])],
            "openings": openings}


def _ai_overlay(img: Image.Image, plan: dict) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    out = Image.blend(img.convert("RGB"), Image.new("RGB", img.size, "white"), 0.35)
    d = ImageDraw.Draw(out)
    side = max(img.size)
    lw = max(2, round(side / 600))
    font = ImageFont.load_default(size=max(12, round(side / 90)))

    def label(p, text, color):
        x, y = p
        box = d.textbbox((x, y), text, font=font, anchor="mm")
        d.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill="white", outline=color)
        d.text((x, y), text, fill=color, font=font, anchor="mm")

    for w in plan.get("walls", []):
        d.line([tuple(w["start"]), tuple(w["end"])], fill=(200, 30, 30), width=lw)
    colors = {"door": (20, 140, 70), "window": (40, 90, 220)}
    for o in plan.get("openings", []):
        d.line([tuple(o["start"]), tuple(o["end"])], fill=colors.get(o.get("type"), (90, 90, 90)), width=lw * 3)
    for w in plan.get("walls", []):
        label(((w["start"][0] + w["end"][0]) / 2, (w["start"][1] + w["end"][1]) / 2), w["id"], (200, 30, 30))
    for o in plan.get("openings", []):
        label(((o["start"][0] + o["end"][0]) / 2, (o["start"][1] + o["end"][1]) / 2), o["id"],
              colors.get(o.get("type"), (90, 90, 90)))
    return out


@app.get("/api/plans/{pid}/ai", dependencies=[Depends(auth)])
def ai_package(pid: str):
    """Zip for asking a vision model (Claude, ChatGPT...) to correct a plan: image, numbered overlay,
    trimmed annotation, instructions, a ready prompt and a script that draws the answer for checking."""
    row = store.get(pid)
    if row is None or not os.path.isfile(store.render_path(pid)):
        raise HTTPException(404)
    img = Image.open(store.render_path(pid)).convert("RGB")
    plan = _ai_plan(row["corrected"] or row["machine"], pid, row["title"], img.size)
    ppm = plan["scale"].get("px_per_m")
    fmt = dict(w=img.size[0], h=img.size[1], pid=pid,
               scale=f"{ppm:.2f} px por metro (paredes costumam ter 0,10 a 0,25 m)" if ppm else "desconhecida")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, im in (("planta.png", img), ("planta_numerada.png", _ai_overlay(img, plan))):
            b = io.BytesIO()
            im.save(b, "PNG", optimize=True)
            z.writestr(name, b.getvalue())
        z.writestr("planta.json", json.dumps(plan, ensure_ascii=False, indent=1))
        z.writestr("LEIA-ME.md", AI_README.format(**fmt))
        z.writestr("PROMPT.txt", AI_PROMPT.format(**fmt))
        z.writestr("conferir.py", AI_CHECK)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["title"])[:60] or pid
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="ia_{safe}.zip"'})


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
