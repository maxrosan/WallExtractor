# Editor de anotações

Aplicação web para corrigir a saída do extrator planta por planta, com as
correções virando dados de treino. Código em `editor/` (backend FastAPI e
página estática) e `Dockerfile` na raiz.

## O que faz

- **Fila de correção.** PDFs enviados pela página ou por `scripts/enqueue.py`
  são extraídos no servidor (ramo vetorial; raster se `EDITOR_MODEL` apontar
  para um `.onnx`) e entram como *pendente*. O revisor abre, corrige, marca
  como *corrigida* e a próxima abre sozinha.
- **Editor de geometria.** Paredes e aberturas são segmentos com espessura,
  na escala já resolvida pelo extrator. Arrastar move; pontas redimensionam;
  ferramentas de parede, porta e janela desenham com a medida em metros ao
  vivo. Encaixe nas pontas de outras paredes, em 0/90° e nas linhas do
  próprio PDF (a pena grossa e a pena média são enviadas como primitivas),
  então uma correção sai com a coordenada exata do desenho.
- **Quadro de esquadrias.** Larguras por código, com "aplicar" a todas as
  aberturas do código. Ao arrastar uma porta ou janela, o comprimento
  encaixa nas larguras do quadro.
- **Escala.** Vem das cotas; a ferramenta Escala recalibra com dois cliques e
  uma distância.
- **Exportação para treino.** `GET /api/export?status=corrected` devolve um
  zip com `<id>.png` (lado maior 1024 px) e `<id>.json` (WallPlan nas
  coordenadas dessa imagem, com metros). `scripts/fetch_corrections.py` baixa
  e `wallextractor.annotations` rasteriza as máscaras no formato do
  `train_seg`. A saída da máquina fica guardada ao lado da correção para
  medir a evolução do sistema.

## Atalhos

| Tecla | Ação |
|---|---|
| V / W / D / J / C | selecionar, parede, porta, janela, escala |
| Delete | excluir seleção |
| Ctrl+Z / Ctrl+Y | desfazer / refazer |
| [ / ] | espessura da parede selecionada −1 / +1 cm |
| Esc | cancelar desenho ou seleção |
| roda do mouse | zoom (acima de 125% a planta é re-renderizada nítida do PDF) |

## Rodar local

```
pip install -r requirements-editor.txt
EDITOR_DATA=./data/editor EDITOR_TOKEN=segredo uvicorn editor.app:app --port 8000
```

Abra http://localhost:8000, informe o token, envie PDFs.

## EasyPanel

1. Crie um serviço **App** a partir do repositório GitHub, branch desta
   entrega, build por **Dockerfile** (na raiz).
2. Volume: monte um volume persistente em `/data` (PDFs, renders e o banco
   SQLite ficam lá).
3. Variáveis de ambiente: `EDITOR_TOKEN` (obrigatória, é a senha da página) e,
   opcionalmente, `EDITOR_MODEL=/data/best.onnx` para plantas raster.
4. Porta 8000, domínio com HTTPS pelo próprio EasyPanel.
5. Memória: o contêiner usa entre 300 e 600 MB em uso normal. Fica dentro dos
   6 a 8 GB do servidor com folga.

## Ciclo com o treino

```
# enviar plantas para a fila
python scripts/enqueue.py --editor https://SEU-DOMINIO --token SEGREDO plantas/*.pdf

# no RunPod (scripts/runpod_job.sh), com WE_EDITOR_URL e EDITOR_TOKEN no ambiente do pod:
# as plantas corrigidas são baixadas e entram no treino com --extra-train, repetidas 20x
python scripts/runpod_ctl.py create --branch <branch> ...   # passe WE_EDITOR_URL/EDITOR_TOKEN pelo env do pod
```

Manualmente:

```
python scripts/fetch_corrections.py --editor https://SEU-DOMINIO --token SEGREDO --out data/corrections --prepared data/prepared_corr
python -m wallextractor.train_seg --data data/prepared --extra-train data/prepared_corr/train --extra-val data/prepared_corr/val ...
```

Os mesmos pares imagem + JSON servem para o fine-tuning do Qwen-VL quando
esse passo entrar: a imagem é o input e o JSON é o alvo.

## API

| Método e rota | Uso |
|---|---|
| `POST /api/plans` (multipart `file`, `title`, `page`) | envia um PDF, extrai e enfileira |
| `GET /api/plans?status=pending` | fila |
| `GET /api/plans/{id}` | metadados, saída da máquina, correção, primitivas |
| `PUT /api/plans/{id}/annotation` | salva a correção (JSON WallPlan) |
| `POST /api/plans/{id}/status` `{"status": "corrected"}` | pendente / corrigida / pulada |
| `GET /api/plans/{id}/image` | render base (2000 px) |
| `GET /api/plans/{id}/tile?x&y&w&h&s` | recorte re-renderizado do PDF para zoom |
| `GET /api/export?status=corrected` | zip de pares de treino |

Todas as rotas `/api` exigem o header `X-Token` (ou `?token=`) quando
`EDITOR_TOKEN` está definido.
