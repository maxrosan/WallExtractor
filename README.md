# WallExtractor

Extrai paredes e aberturas (portas e janelas) de PDFs de plantas baixas e
devolve um JSON com segmentos, espessura e aberturas.

Documentação de decisão:

- `docs/datasets.md`: datasets candidatos para treino.
- `docs/escolha-do-modelo.md`: por que a geometria fica com um modelo de
  segmentação pequeno (roda em CPU com poucos GB) e o VLM só entra para
  semântica, fora do servidor.

## Arquitetura

```
PDF ──► wallextractor.pdf ──► PDF vetorial? ──sim──► primitivas (PyMuPDF) ──► regras ──┐
                                   │                                                    ▼
                                   └──não──► render 1024 px ──► SegFormer (ONNX, CPU) ──► máscara
                                                                                          │
                                                        wallextractor.vectorize ◄─────────┘
                                                                  │
                                                                  ▼
                                                   JSON (wallextractor.schema)
```

## Uso

Inferência (servidor, só CPU):

```
pip install -r requirements.txt
python -m wallextractor.infer planta.pdf --model runs/b1/best.onnx --out planta.json --overlay planta.png
```

Preparar dados e treinar (GPU):

```
pip install -r requirements-train.txt
python scripts/fetch_cubicasa_subset.py --out data/cubicasa5k --train 400 --val 100
python -m wallextractor.cubicasa --root data/cubicasa5k --out data/prepared
python -m wallextractor.train_seg --data data/prepared --out runs/b1 --epochs 15 --size 512 --batch 8 --export-onnx
```

Treinar no RunPod sem SSH (só HTTPS, via API REST + Jupyter do template):

```
export RUNPOD_API_KEY=...
python scripts/runpod_ctl.py gpus
python scripts/runpod_ctl.py create --branch <branch> --train 400 --val 100 --epochs 15
python scripts/runpod_ctl.py log --tail 40
python scripts/runpod_ctl.py get results/metrics.jsonl
python scripts/runpod_ctl.py get results/best.onnx
python scripts/runpod_ctl.py terminate
```

O pod se encerra sozinho depois de ficar ocioso (`--idle-minutes`) e tem um
limite duro de tempo (`--hard-minutes`) para proteger o saldo.

## Formato de saída

```json
{
 "version": "0.1",
 "source": {"file": "planta.pdf", "page": 1, "kind": "raster"},
 "image": {"width": 1024, "height": 880, "dpi": 91.9},
 "scale": {"px_per_m": null, "method": null},
 "walls": [{"id": "w1", "start": [x, y], "end": [x, y], "thickness": 11.0, "kind": "external", "polygon": null}],
 "openings": [{"id": "o1", "type": "door", "start": [x, y], "end": [x, y], "width": 30.0, "wall_id": "w1"}]
}
```

Coordenadas em pixels da imagem de trabalho, origem no canto superior esquerdo.

## Licenças de dados

CubiCasa5K é CC BY-NC 4.0: serve para o protótipo e para pesquisa. A versão
de produção deve ser treinada em ResPlan (CC BY 4.0), MSD e dados próprios.
