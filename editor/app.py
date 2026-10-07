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
import math
import os
import shutil
import tempfile
import time
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from wallextractor.schema import WallPlan, validate
from .store import ENGINES, JOB_STATUSES, STATUSES, Store

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
    row["ai_job"] = store.job_latest(pid)
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
- `recortes/`: a planta em pedaços ampliados {zoom}x, com grade das coordenadas ORIGINAIS de
  `planta.png` a cada {step} px (linhas finas vermelhas, valores nas bordas). Use os recortes para
  ler as coordenadas com precisão; nunca copie posições da imagem ampliada sem converter.
{tiles}

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
  dentro da espessura da parede). Etiquetas ajudam: P1, P2… são portas; J1, J2… são janelas. Uma
  etiqueta repetida (duas P2) indica tipos iguais em lugares diferentes: cada uma é uma abertura.
- Passagem sem porta (vão livre, "VÃO 80x2,10", arco sem folha): deixe a parede interrompida no vão,
  sem abertura; o formato só tem porta e janela.

## Formato de `planta.json` e da resposta

- `walls`: `{{"id": "w1", "start": [x, y], "end": [x, y], "thickness": px}}`. A linha vai pelo eixo
  da parede (no meio entre as duas faces) e `thickness` é a distância entre as faces, em pixels.
  A parede é DIVIDIDA em todo encontro (em L, em T ou em cruz) e em todo vão: cada trecho vai de um
  encontro ou borda de vão até o próximo. Num T, a parede que continua vira dois trechos que terminam
  no eixo da parede que chega; em L, os dois trechos terminam no cruzamento dos eixos.
- `openings`: `{{"id": "o1", "type": "door" | "window", "start": [x, y], "end": [x, y],
  "wall_id": "w3"}}`, sobre o eixo da parede, de uma borda do vão até a outra (a largura da abertura);
  `wall_id` é um dos trechos vizinhos (pode ficar de fora: o editor liga pela geometria).
  Porta: entre os batentes, onde a linha da parede é interrompida (não pela folha nem pelo arco).
  Janela: só a extensão do caixilho; prolongamentos coloridos, linhas de chamada e o círculo da
  etiqueta não fazem parte da largura.

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
Para ler coordenadas, use os `recortes/` (a grade dá o valor exato). Se puder rodar código: comece
do `planta.json` (juntar trechos alinhados ajuda, depois dividir nos encontros e vãos), rode
`python conferir.py correcao.json`, olhe `conferencia.png` e ajuste antes de entregar. Se não puder
gerar arquivo, responda só com o JSON num bloco de código: o editor aceita o texto colado.
No editor, "Importar correção" (arquivo) ou "Colar resposta" (texto) carrega o resultado; nada é
marcado como corrigido até o revisor conferir.
"""

AI_PROMPT = """Anexei os arquivos de uma planta baixa (planta.png, planta_numerada.png, planta.json, LEIA-ME.md,
conferir.py e a pasta recortes/; se vierem num .zip, descompacte). Siga o LEIA-ME.md: corrija as paredes, portas e
janelas de planta.json comparando com planta.png e me devolva correcao.json com a lista completa, em
pixels de planta.png ({w} x {h}), plan_id "{pid}". Se puder rodar código, confira com conferir.py antes
de entregar. Use os recortes com grade para ler as coordenadas. Se não puder gerar arquivo, responda
só com o JSON num bloco de código.
"""


def _ai_tiles(img: Image.Image, tile: int = 700, overlap: float = 0.15, zoom: int = 2, step: int = 50):
    """The plan in overlapping zoomed pieces with a grid labelled in ORIGINAL pixel coordinates, so a model
    can read exact positions instead of estimating them on a downscaled image. Returns [(name, png, box)]."""
    from PIL import ImageDraw, ImageFont

    W, H = img.size
    font = ImageFont.load_default(size=13)

    def starts(n):
        if n <= tile:
            return [0]
        k = math.ceil((n - tile) / (tile * (1 - overlap))) + 1
        return [round(i * (n - tile) / (k - 1)) for i in range(k)]

    out = []
    for r, y0 in enumerate(starts(H), 1):
        for c, x0 in enumerate(starts(W), 1):
            x1, y1 = min(W, x0 + tile), min(H, y0 + tile)
            im = img.crop((x0, y0, x1, y1)).resize(((x1 - x0) * zoom, (y1 - y0) * zoom), Image.NEAREST).convert("RGB")
            d = ImageDraw.Draw(im)
            for x in range((x0 // step + 1) * step, x1, step):
                X = (x - x0) * zoom
                d.line([(X, 0), (X, im.height)], fill=(230, 120, 120), width=1)
                for yy in (2, im.height - 16):
                    d.text((X + 2, yy), str(x), fill=(200, 0, 0), font=font, stroke_width=2, stroke_fill="white")
            for y in range((y0 // step + 1) * step, y1, step):
                Y = (y - y0) * zoom
                d.line([(0, Y), (im.width, Y)], fill=(230, 120, 120), width=1)
                for xx in (2, im.width - 34):
                    d.text((xx, Y + 2), str(y), fill=(200, 0, 0), font=font, stroke_width=2, stroke_fill="white")
            b = io.BytesIO()
            im.save(b, "PNG", compress_level=6)
            out.append((f"recortes/r{r}c{c}_x{x0}-{x1}_y{y0}-{y1}.png", b.getvalue(), (x0, y0, x1, y1)))
    return out

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


AI_BATCH_README = """# Lote de plantas para correção com IA

Este zip tem {n} plantas, uma por pasta. Cada pasta é um pacote completo e independente, com o seu
próprio `LEIA-ME.md` (regras de anotação e formato), `planta.png`, `planta_numerada.png`,
`planta.json`, `recortes/` e `conferir.py`.

| Pasta | plan_id | Imagem |
|---|---|---|
{rows}

## Como trabalhar

1. Leia o `LEIA-ME.md` de uma das pastas: as regras são as mesmas para todas as plantas.
2. Para CADA pasta, em ordem, sem pedir confirmação entre uma e outra: corrija a planta seguindo o
   `LEIA-ME.md` dela (use os `recortes/` para ler as coordenadas e o `conferir.py` para conferir) e
   grave o resultado em `correcoes/<plan_id>.json`, no formato do `correcao.json` do LEIA-ME
   (com o `plan_id` e a `image` daquela planta). Depois de cada planta, acrescente uma linha em
   `correcoes/progresso.txt`: `<plan_id> ok` ou `<plan_id> pulada: <motivo>`.
3. Coordenadas sempre em pixels da `planta.png` da própria pasta.
4. No fim, junte tudo num único `correcoes.json`, um objeto com uma chave por planta:
   `{{"<plan_id>": {{"plan_id": ..., "image": ..., "walls": [...], "openings": [...], "notas": "..."}}, ...}}`,
   e entregue `correcoes.json` (e, se puder, `correcoes.zip` com a pasta `correcoes/`).
5. Se a sessão for interrompida ou o limite de tempo acabar, entregue o que já tiver em
   `correcoes.json`. Quando eu disser "continue", retome pela primeira planta que ainda não tem
   arquivo em `correcoes/` (confira em `progresso.txt`).
6. Só depois de corrigir TODAS as plantas (as correções vêm antes; se o tempo estiver acabando,
   entregue as correções e deixe este passo), entregue mais dois arquivos que cobrem o lote inteiro:
   - `ferramentas.py`: os scripts que você usou nas correções, num único arquivo executável
     (Python + Pillow/numpy), uma função por ferramenta, com um comentário curto dizendo que problema
     ela resolve. Sem coordenadas nem valores fixos de uma planta: parâmetros relativos (pixels da
     imagem ou espessura da parede), para servir em outras plantas.
   - `relatorio_lote.md`: copie o modelo abaixo e PREENCHA todas as tabelas, com números contados
     nos `planta.json` originais comparados às suas correções (não descreva o método em geral).

```markdown
# Relatório do lote

## 1. Problemas do rascunho (planta.json original)
| Problema | Plantas onde aparece (plan_id) | Ocorrências no lote | Exemplo (plan_id, id original) |
|---|---|---:|---|
| Parede real picotada em vários trechos | | | |
| Parede onde não existe (cota, projeção, piso, móvel, escada...) | | | |
| Eixo da parede deslocado | | | |
| Espessura errada | | | |
| Parede passando por cima de um vão | | | |
| Porta onde não existe | | | |
| Janela onde não existe | | | |
| Abertura faltando | | | |
| Tipo trocado (porta x janela) | | | |
| Largura/posição da abertura imprecisa | | | |
| (outros que você encontrou) | | | |

## 2. O que um script resolve
| Problema | Resolve sozinho? (sim / em parte / não) | Função em ferramentas.py | Parâmetros (relativos) | Onde falha |
|---|---|---|---|---|

## 3. Decisões que exigiram olhar a imagem
| plan_id | Decisão | Por quê (o que na imagem decidiu) |
|---|---|---|

## 4. Padrão x exceção
- Repetiu em quase todas as plantas: ...
- Apareceu em uma planta só: ...

## 5. O que mudar no rascunho automático (em ordem de impacto)
| # | Mudança | Problemas da seção 1 que resolve | Ganho estimado no lote |
|---|---|---|---|
```

No editor, "Importar lote" aceita o `correcoes.json` ou os vários `<plan_id>.json` de uma vez; cada
correção entra como rascunho na sua planta, para revisão.
"""

AI_BATCH_PROMPT = """Anexei um zip com {n} plantas baixas para corrigir, uma por pasta. Descompacte e siga o
LEIA-ME_LOTE.md da raiz: corrija cada planta seguindo o LEIA-ME.md da pasta dela, uma depois da outra, sem
me pedir confirmação entre elas, gravando correcoes/<plan_id>.json e anotando correcoes/progresso.txt. No fim,
me entregue correcoes.json com todas as correções (uma chave por plan_id). Se o tempo acabar antes, entregue o
que já tiver; quando eu disser "continue", retome pela primeira planta sem arquivo em correcoes/.
Só depois de terminar todas as plantas, entregue também ferramentas.py e relatorio_lote.md cobrindo o lote
inteiro, preenchendo o modelo de relatório do passo 6 do LEIA-ME_LOTE.md com números contados (as correções
vêm antes: se faltar tempo, deixe estes dois).
"""

AI_BATCH_MAX = 40


def _safe(title: str, pid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:60] or pid


def _ai_prompt_header(text: str) -> dict:
    """The prompt also travels in a header (URL-encoded), so the page can show it next to the download."""
    from urllib.parse import quote

    return {"X-AI-Prompt": quote(text), "Access-Control-Expose-Headers": "X-AI-Prompt"}


def _ai_files(row: dict) -> list:
    """The files of one plan's package as [(name, bytes)]: image, numbered overlay, trimmed annotation,
    instructions, a ready prompt, the checking script and the grid tiles."""
    pid = row["id"]
    img = Image.open(store.render_path(pid)).convert("RGB")
    plan = _ai_plan(row["corrected"] or row["machine"], pid, row["title"], img.size)
    ppm = plan["scale"].get("px_per_m")
    tiles = _ai_tiles(img)
    fmt = dict(w=img.size[0], h=img.size[1], pid=pid, zoom=2, step=50,
               scale=f"{ppm:.2f} px por metro (paredes costumam ter 0,10 a 0,25 m)" if ppm else "desconhecida",
               tiles="\n".join(f"  - `{n}`: x de {b[0]} a {b[2]}, y de {b[1]} a {b[3]}" for n, _p, b in tiles))
    files = []
    for name, im in (("planta.png", img), ("planta_numerada.png", _ai_overlay(img, plan))):
        b = io.BytesIO()
        im.save(b, "PNG", compress_level=6)
        files.append((name, b.getvalue()))
    files += [("planta.json", json.dumps(plan, ensure_ascii=False, indent=1).encode("utf-8")),
              ("LEIA-ME.md", AI_README.format(**fmt).encode("utf-8")),
              ("PROMPT.txt", AI_PROMPT.format(**fmt).encode("utf-8")),
              ("conferir.py", AI_CHECK.encode("utf-8"))]
    files += [(name, png) for name, png, _box in tiles]
    return files


@app.get("/api/plans/{pid}/ai", dependencies=[Depends(auth)])
def ai_package(pid: str):
    """Zip for asking a vision model (Claude, ChatGPT...) to correct one plan."""
    row = store.get(pid)
    if row is None or not os.path.isfile(store.render_path(pid)):
        raise HTTPException(404)
    buf = io.BytesIO()
    prompt = ""
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in _ai_files(row):
            z.writestr(name, data)
            if name == "PROMPT.txt":
                prompt = data.decode("utf-8")
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="ia_{_safe(row["title"], pid)}.zip"',
                             **_ai_prompt_header(prompt)})


@app.post("/api/plans/{pid}/faces", dependencies=[Depends(auth)])
def faces_draft(pid: str) -> dict:
    """A new draft from the base render with the face-pair detector (wallextractor.faces): walls as pairs of
    parallel lines, doors and windows by the drawing's colours. Not saved: the page loads it like an
    imported answer, so the reviewer can undo it."""
    from wallextractor.faces import DesenhoNaoSuportado, pipeline_automatico

    if store.get(pid) is None or not os.path.isfile(store.render_path(pid)):
        raise HTTPException(404)
    try:
        draft = pipeline_automatico(store.render_path(pid), plan_id=pid)
        if len(draft["walls"]) < 4:  # replacing the screen with an (almost) empty draft would only lose work
            raise DesenhoNaoSuportado(
                f"o detector de faces só achou {len(draft['walls'])} parede(s) nesta planta; o estilo do desenho "
                "não serve para ele, use o rascunho da máquina")
        return draft
    except DesenhoNaoSuportado as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # noqa: BLE001 - a drawing the detector cannot read is reported, not a 500
        raise HTTPException(422, f"o detector de faces não conseguiu ler esta planta ({type(exc).__name__}: {exc})")


@app.post("/api/ai/batch", dependencies=[Depends(auth)])
def ai_batch(body: Dict[str, Any]):
    """One zip with a full package per plan (one folder each) and batch instructions at the root, for a
    model to correct the plans one after the other on its own and hand back a single correcoes.json."""
    ids = [str(i) for i in (body.get("ids") or [])]
    if not ids:
        raise HTTPException(422, "nenhuma planta selecionada")
    if len(ids) > AI_BATCH_MAX:
        raise HTTPException(422, f"no máximo {AI_BATCH_MAX} plantas por lote")
    rows = []
    for pid in ids:
        row = store.get(pid)
        if row is None or not os.path.isfile(store.render_path(pid)):
            raise HTTPException(404, f"planta {pid} não encontrada")
        rows.append(row)
    buf = io.BytesIO()
    table = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            folder = f"{_safe(row['title'], row['id'])}_{row['id']}"
            for name, data in _ai_files(row):
                z.writestr(f"{folder}/{name}", data)
            w, h = Image.open(store.render_path(row["id"])).size
            table.append(f"| `{folder}/` | {row['id']} | {w} x {h} |")
        z.writestr("LEIA-ME_LOTE.md", AI_BATCH_README.format(n=len(rows), rows="\n".join(table)))
        prompt = AI_BATCH_PROMPT.format(n=len(rows))
        z.writestr("PROMPT_LOTE.txt", prompt)
    stamp = __import__("time").strftime("%Y%m%d-%H%M")
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="ia_lote_{len(rows)}_plantas_{stamp}.zip"',
                             **_ai_prompt_header(prompt)})


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


# ---------------------------------------------------------------- Claude Code on the user's computer
# The editor cannot reach the user's computer, so the computer comes to the editor: scripts/claude_worker.py
# claims queued plans, downloads the same package as "Pedir ajuda à IA", runs `claude -p` (engine "claude") or
# `codex exec` (engine "codex", ChatGPT subscription; engine "openai", the OpenAI API with the job's model) on it
# and posts the correction back; engine "anthropic" is `claude -p` billed to the Anthropic API. The correction is kept in the job and loaded into the plan when the reviewer opens it.
_workers: Dict[str, float] = {}  # worker name -> last time it asked for work
_worker_engines: Dict[str, List[str]] = {}  # worker name -> engines it runs


@app.post("/api/ai/jobs", dependencies=[Depends(auth)])
def ai_jobs_add(body: Dict[str, Any]) -> dict:
    ids = [str(i) for i in body.get("ids") or []]
    engine = body.get("engine") or "claude"
    model = str(body.get("model") or "").strip() or None
    if model is not None and not all(ch.isalnum() or ch in ".-_" for ch in model):
        raise HTTPException(422, "nome de modelo inválido")
    if not ids:
        raise HTTPException(422, "nenhuma planta escolhida")
    if engine not in ENGINES:
        raise HTTPException(422, f"engine deve ser um de {ENGINES}")
    added, skipped = [], []
    for pid in ids:
        if store.get(pid) is None:
            skipped.append({"pid": pid, "why": "planta não está no editor"})
        elif store.job_add(pid, engine, model) is None:
            skipped.append({"pid": pid, "why": "já está na fila"})
        else:
            added.append(pid)
    return {"added": added, "skipped": skipped}


@app.get("/api/ai/jobs", dependencies=[Depends(auth)])
def ai_jobs_list() -> dict:
    now = time.time()
    return {"jobs": store.job_list(), "stats": store.job_stats(), "workers": [{"name": k, "seen_s": round(now - v), "engines": _worker_engines.get(k, [])}
                                                for k, v in _workers.items()]}


@app.post("/api/ai/jobs/claim", dependencies=[Depends(auth)])
def ai_jobs_claim(body: Dict[str, Any]) -> dict:
    worker = str(body.get("worker") or "worker")[:60]
    engines = [str(e) for e in body.get("engines") or ["claude"]]
    _workers[worker] = time.time()
    _worker_engines[worker] = [e for e in engines if e in ENGINES]
    job = store.job_claim(worker, engines)
    if job is not None:
        job["title"] = (store.get(job["pid"]) or {}).get("title")
    return {"job": job}


@app.post("/api/ai/jobs/{jid}", dependencies=[Depends(auth)])
def ai_jobs_update(jid: int, body: Dict[str, Any]) -> dict:
    """Progress from the worker ({message}), its end ({status: done, result} or {status: error, message}),
    a cancel from the page ({status: cancelled}) or the page saying it loaded the result ({applied: true})."""
    job = store.job_get(jid)
    if job is None:
        raise HTTPException(404)
    status = body.get("status")
    if status is not None and status not in JOB_STATUSES:
        raise HTTPException(422, f"status deve ser um de {JOB_STATUSES}")
    if job["status"] in ("cancelled", "done", "error") and status not in (None, "cancelled"):
        return job  # a worker reporting on a job the reviewer cancelled: ignored, and it sees the status
    result = body.get("result")
    if status == "done":
        if not isinstance(result, dict) or not isinstance(result.get("walls"), list) or not isinstance(result.get("openings", []), list):
            raise HTTPException(422, "result precisa de walls e openings")
    if job["worker"]:
        _workers[job["worker"]] = time.time()
    msg = body.get("message")
    def num(key, kind):
        try:
            return kind(body[key]) if body.get(key) is not None else None
        except (TypeError, ValueError):
            return None

    return store.job_update(jid, status=status, message=str(msg)[:2000] if msg is not None else None,
                            result=result if status == "done" else None,
                            applied=bool(body["applied"]) if "applied" in body else None,
                            cost_usd=num("cost_usd", float), tokens=num("tokens", int), minutes=num("minutes", float))


@app.middleware("http")
async def revalidate_static(request, call_next):
    """The page and its scripts are revalidated (ETag) on every load, so a deploy is picked up at once."""
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

