# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# WallExtractor

Extrai paredes, portas e janelas de plantas baixas em PDF e gera um JSON
(`WallPlan`, ver `wallextractor/schema.py`). Projeto em português: docs,
mensagens ao usuário e textos do editor em pt-BR; código e commits em inglês.

Não há suíte de testes nem linter configurados. A verificação é rodar o
extrator em PDFs reais e olhar os overlays, e medir contra as correções do
editor (`scripts/eval_openings.py`).

## Estrutura

- `wallextractor/vector_walls.py`: ramo vetorial (PDF de CAD). É o caminho
  principal. Paredes = pares de linhas da pena mais grossa; escala pelo rótulo
  "ESCALA 1:x" e pelas cotas; aberturas em `find_openings` (folha aberta,
  folha fechada, caixilho, vão, etiquetas P/J, quadro de esquadrias).
- `wallextractor/infer.py`: `extract_vector` (vetorial) e o segmentador raster
  (ONNX/torch).
- `wallextractor/train_seg.py`, `cubicasa.py`, `styles.py`, `eval_seg.py`:
  treino e avaliação do SegFormer (ramo raster, para PDF escaneado ou imagem).
- `editor/`: editor de anotações (FastAPI + Konva, `static/app.js` em JS puro).
  `Dockerfile` na raiz.
  "Pedir ajuda à IA" baixa `ia_<título>.zip` (`planta.png`, `planta_numerada.png`,
  `planta.json`, `LEIA-ME.md`, `PROMPT.txt`, `conferir.py`), feito para o Claude
  ou o ChatGPT; quando o usuário trouxer esse zip, siga o LEIA-ME e devolva
  `correcao.json` (pixels de `planta.png`, paredes interrompidas nos vãos,
  aberturas no vão), conferido com `conferir.py`.
  "Corrigir com Claude" / "com ChatGPT" põe plantas numa fila no servidor;
  `scripts/claude_worker.py --engines claude,codex` no Lenovo (com `EDITOR_TOKEN` no
  ambiente) pega cada uma, roda `claude -p` ou `codex exec` no mesmo pacote com um
  `verificar.py` e devolve a correção, que entra como rascunho quando a
  planta é aberta (ver `docs/editor.md`).
- `wallextractor/vlm_data.py`, `train_vlm.py`, `eval_vlm.py`: fine-tuning do
  Qwen3-VL (LoRA só no modelo de linguagem, visão congelada). Alvo compacto
  `{"walls":[[x1,y1,x2,y2,t]],"doors":[...],"windows":[...]}` em 0–1000
  relativo à imagem, paredes ordenadas de cima para baixo; `decode_text`
  aceita resposta truncada e religa aberturas às paredes pela geometria.
  Métrica: F1 de paredes/portas/janelas a 1,5% e 5% do lado maior.
- `scripts/eval_openings.py`: mede portas e janelas do ramo vetorial contra
  as correções do editor. Use antes e depois de mexer em `find_openings`.
- `scripts/enqueue.py`, `fetch_corrections.py`: enviar PDFs ao editor e
  baixar correções.
- `docs/experimentos.md`: registro de TODAS as rodadas (E1-E4 treino, V1-V2
  vetorial). Toda medição nova entra lá, com tabela e o que foi descartado.
- `docs/editor.md`: uso, API e deploy do editor. `docs/datasets.md`: fontes,
  inclusive plantas brasileiras públicas.

## Comandos

Dependências em camadas: `requirements.txt` (inferência, só CPU, **sem
torch**), `requirements-editor.txt` (+ FastAPI), `requirements-train.txt`
(+ transformers/onnx; o torch vem da imagem de treino), `requirements-vlm.txt`
(+ transformers 4.57.6 fixo, peft, para o Qwen3-VL).

```
# inferência (tenta o vetorial; o raster só roda se o vetorial falhar e exige --model)
python -m wallextractor.infer planta.pdf --model runs/b1/best.onnx --out planta.json --overlay planta.png
python -m wallextractor.infer planta.pdf --no-vector --model ...       # força o ramo raster
python -m wallextractor.infer planta.pdf --answers respostas.json      # larguras confirmadas {"P1": 0.70}

# só o ramo vetorial, com overlays para inspeção
python scripts/try_vector.py out_dir a.pdf b.pdf [--page N]

# portas/janelas do vetorial contra as plantas corrigidas no editor
python scripts/eval_openings.py --download data/gt --editor URL --token T
python scripts/eval_openings.py data/gt --tol 0.25 -v

# dados e treino (GPU)
python scripts/fetch_cubicasa_subset.py --out data/cubicasa5k --train 400 --val 100
python -m wallextractor.cubicasa --root data/cubicasa5k --out data/prepared
python scripts/fetch_corrections.py --editor URL --token T --out data/corrections --prepared data/prepared_corr
python -m wallextractor.train_seg --data data/prepared --extra-train data/prepared_corr/train \
    --extra-val data/prepared_corr/val --out runs/b1 --epochs 15 --size 512 --batch 8 --export-onnx
python -m wallextractor.eval_seg --model runs/b1/best --data data/prepared --split val --styles solid,hatch45,outline

# Qwen3-VL (GPU): dados, treino LoRA, avaliação (base sem --adapter = zero-shot)
python -m wallextractor.vlm_data --cubicasa data/prepared --corrections data/prepared_corr --out data/vlm \
    --side 1024 --repeat-corr 20 --restyle-prob 0.5
python -m wallextractor.train_vlm --data data/vlm --out runs/vlm4b --epochs 2 --grad-accum 8
python -m wallextractor.eval_vlm --data data/vlm --adapter runs/vlm4b/best --source editor --out runs/vlm4b/eval.json

# editor local
EDITOR_DATA=./data/editor EDITOR_TOKEN=segredo uvicorn editor.app:app --port 8000
```

`scripts/runpod_ctl.py` treina no RunPod só por HTTPS (exige
`RUNPOD_API_KEY`); hoje o treino roda no Pichau (abaixo).

`data/`, `runs/`, `results/`, `*.onnx` e `*.pdf` estão no `.gitignore`.

## Arquitetura

`wallextractor.infer.extract` escolhe o ramo:

1. **Vetorial** (sem rede neural, < 2 s/prancha). Só é tentado se a página
   tem ≥ 50 desenhos e devolve ≥ 4 paredes; senão cai no raster. Localiza a
   região da PLANTA BAIXA na prancha (rótulo ou cluster mais denso da pena
   grossa), lê a escala, pareia linhas paralelas a 8–40 cm e emite a linha
   média como parede; `schedule.py` lê o "QUADRO DE ESQUADRIAS" e sobrescreve
   larguras quando o quadro é texto real. Trabalha em pontos PDF;
   `extract_vector` converte para pixels do recorte da região (lado maior
   2048). O que o extrator não resolve vai para `plan.questions` ("DÚVIDA:").
2. **Raster**: render da página → SegFormer MiT-B1 em ONNX na CPU
   (`imageops`, compartilhado com o treino) → máscara de 4 classes
   (`cubicasa.CLASSES`: background, wall, door, window) →
   `vectorize.mask_to_plan` (distance transform, esqueleto, Hough, fusão de
   colineares).

O servidor de produção (EasyPanel, 6–8 GB de RAM, sem GPU) não tem torch:
`infer`, `imageops`, `vectorize`, `vector_walls` e `editor` precisam importar
só com `requirements.txt`. Motivação em `docs/escolha-do-modelo.md`.

Ciclo de dados: CubiCasa5K (`cubicasa.py`: SVG → `WallPlan` + máscaras;
CC BY-NC, só protótipo) → `styles.py` redesenha as paredes nos estilos
brasileiros (hachura, contorno duplo, cinza) como augmentação → `train_seg`
(melhor checkpoint por IoU de parede, exporta ONNX). As correções do editor
(pares imagem + JSON de `/api/export`, em pixels de um render base de lado
2000) viram máscaras em `annotations.py` e entram com `--extra-train`,
sobreamostradas 20×.

Formato: coordenadas em pixels da imagem de trabalho, origem no canto
superior esquerdo; `scale.px_per_m` converte para metros. Ao mudar o schema,
atualize `SCHEMA_VERSION` e `schema.validate`; editor e `annotations.py`
consomem o mesmo JSON.

## Editor em produção (EasyPanel)

- Serviço `wallextractor-editor`, projeto `botai`,
  https://botai-wallextractor-editor.uuclvw.easypanel.host
- Volume `/data`: PDFs, SQLite com plantas e correções, `model.onnx`
  (modelo raster). Sobrevive a deploys. Nunca apagar o serviço nem o volume.
- Variáveis no ambiente da sessão: `EASYPANEL_URL`, `EASYPANEL_TOKEN`,
  `EASYPANEL_PROJECT`. A senha do editor é a `EDITOR_TOKEN` do serviço: leia
  pela API (`/api/inspectAppService`, campo `env`) sem imprimir.
- API do painel em `$EASYPANEL_URL/api/openapi.json`, header
  `Authorization: Bearer`. Redeploy: `POST /api/deployAppService`
  `{"projectName","serviceName","forceRebuild":true}`; trocar branch:
  `POST /api/updateAppSourceGit`; acompanhar: `GET /api/listActions`.
- Depois do deploy há ~40 s de 502 enquanto o contêiner sobe.
- Modelo raster: `PUT /api/model` (multipart, header `X-Token`), sem redeploy.
- O serviço é construído a partir da branch configurada no painel. Se a
  branch for apagada (ex.: após merge), aponte o serviço para `main` antes.

## GPU local: PC "Pichau" (MCP `Pichau`)

- RTX 5060 Ti 16 GB (Blackwell: exige CUDA ≥ 12.8). Docker e NVIDIA
  Container Toolkit DENTRO do WSL Ubuntu-24.04 (não há Docker Desktop).
- Imagem `we-train` (base `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`).
- Tudo em `/root/we/`: `repo/`, `data/` (CubiCasa 1600/400 já baixado e
  preparado; correções em `prepared_corr/`), `results_e4/` (modelo E4),
  `run.sh` (pipeline dados → correções → treino), logs `*.log`.
- `C:\Users\Pichau\.wslconfig` desliga o desligamento por ociosidade do WSL;
  sem isso os contêineres morrem quando nenhum `wsl.exe` está aberto. Após
  mudar o arquivo é preciso `wsl --shutdown`.
- Aspas PowerShell → wsl → bash → python quebram: grave o script em
  `C:\we\*.sh` (LF) e rode `wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/we/x.sh`.
- `executar_powershell` tem limite de ~60 s; para mais, `iniciar_tarefa`
  ou `setsid nohup` dentro do WSL.
- Qwen: imagem `we-vlm` (= `we-train` + `requirements-vlm.txt`), cópia do repo
  em `/root/we/repo_vlm` (atualizada por `git bundle` em `C:\we\we.bundle`,
  sem push), pesos em `/root/we/hf` (cache do Hugging Face), pipeline
  `run_vlm.sh`, resultados em `/root/we/results_vlm`.
- Arquivo do Pichau para cá: `publicar_arquivo` (link temporário, use
  validade curta).

## Regras

- Nunca escrever tokens ou senhas no chat, no repositório ou em logs. Se for
  inevitável passar um segredo por comando, pergunte antes.
- Plantas de projetos reais (e nomes de proprietários) ficam fora do repo.
  Só as de SJC (Prefeitura de São José dos Campos) são públicas.
- Mudança no ramo vetorial: rodar `scripts/eval_openings.py --tol 0.25` nas
  plantas corrigidas antes e depois, confirmar que as paredes não mudam e
  que `schema.validate` passa. Reverter o que piora, e registrar.
- Mudança no editor: testar no Chromium com Playwright (servindo
  `editor/static` e simulando `/api` com `page.route`) antes de publicar.
- Instalar software, criar tarefas/serviços ou apagar arquivos nos PCs do
  usuário (Pichau, Asus) exige confirmação.

## Estado (2026-10-03)

- Trabalho direto na `main` (a branch `claude/vector-false-doors` acompanha).
  O EasyPanel constrói da `main`; cada mudança no editor é publicada com
  redeploy. A branch `claude/jolly-brown-ij69cl` pode ser apagada.
- Fila do editor (2026-10-07): 68 corrigidas, nenhuma pendente (4 de escritório,
  29 vetoriais de SJC, 17 do portfólio raster de SJC, 18 escaneadas de Bauru;
  parte com ajuda de IA em lote ou do worker do Claude). Em 2026-10-07 as paredes
  que passavam por cima de 16 aberturas em 8 plantas foram cortadas nas bordas
  do vão (backup em `data/backup_before_cut/`). Cópia local antiga em `data/gt`
  (33 plantas).
- Vetorial V3 (branch `claude/vector-false-doors`, tol. 25 cm, 33 plantas,
  gabarito limpo): portas 167/179 com 11 falsas (F1 0,94), janelas 142/167
  com 17 falsas (F1 0,87). As 39 portas falsas herdadas da V2 foram tiradas
  das correções em 2026-10-03 (backup em `data/backup_before_rm/`).
- Direção: o caminho principal é o modelo treinado nas correções (SegFormer
  E5 e depois o Qwen-VL), não refinar heurísticas por estilo de desenho.
- Raster E6 (2026-10-05, publicado no editor): IoU nas 10 de validação
  (50 plantas) parede 0,837 / porta 0,746 / janela 0,824 (E5: 0,739 / 0,612
  / 0,694). Modelo em `/root/we/results_e6/best.onnx`.
- Qwen: o Q1 decorou as plantas do treino (1,00 nas que viu). Nas 6 de
  validação que nenhum viu, F1 a 1,5%: Q2 paredes 0,18 (pontas) / 0,73
  (comprimento), portas 0,18, janelas 0,41; Q1 0,11 / 0,69 / 0,27 / 0,29.
  Adaptadores em `/root/we/results_vlm/run1` (Q1) e `run2` (Q2).
- Divisões de dados no Pichau: `prepared_corr_33`/`vlm_33` (33 plantas, E5 e
  Q1) e `prepared_corr`/`vlm` (50 plantas, E6 e Q2).

## Próximos passos

1. Fixar um conjunto de teste de plantas que nenhum modelo vê (ex.: as 10 de
   validação atuais) e só acrescentar plantas novas ao treino, para as
   comparações não se contaminarem.
2. Qwen: mais plantas de estilos variados antes de treinar de novo; repetição
   menor; considerar o 8B com QLoRA se os dados crescerem.
3. Ramo raster no editor: recortar a região da planta antes de segmentar
   (hoje moldura e carimbo viram "paredes") e segmentar em blocos.
