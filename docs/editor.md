# Editor de anotações

Aplicação web para corrigir a saída do extrator planta por planta, com as
correções virando dados de treino. Código em `editor/` (backend FastAPI e
página estática) e `Dockerfile` na raiz.

## O que faz

- **Fila de correção.** PDFs enviados pela página ou por `scripts/enqueue.py`
  são extraídos no servidor (ramo vetorial; raster se `EDITOR_MODEL` apontar
  para um `.onnx`) e entram como *pendente*. O revisor abre, corrige, marca
  como *corrigida* e a próxima abre sozinha. No ramo raster, o rascunho do
  SegFormer passa por `vectorize.clean_plan` (trechos quase alinhados ao eixo
  endireitados, colineares juntados, cantos ligados, paredes cortadas nos
  vãos e divididas nos encontros); nas 7 plantas de validação que nenhum
  modelo viu, o F1 de paredes por pontas a 1,5% foi de 0,15 para 0,56.
- **Editor de geometria.** Paredes e aberturas são segmentos com espessura,
  na escala já resolvida pelo extrator. Porta e janela se desenham arrastando
  sobre uma parede (a abertura pode passar do fim dela, para o vão que o
  extrator deixa) ou entre dois pontos fora de parede, para o vão entre duas
  paredes ou do fim de uma até a face de outra; ao soltar, a abertura é
  alinhada à parede em cuja linha ela está. Arrastar move; pontas redimensionam;
  ferramentas de parede, porta e janela desenham com a medida em metros ao
  vivo. Encaixe nas pontas de outras paredes, em 0/90° e nas linhas do
  próprio PDF (a pena grossa e a pena média são enviadas como primitivas),
  então uma correção sai com a coordenada exata do desenho.
- **Quadro de esquadrias.** Larguras por código, com "aplicar" a todas as
  aberturas do código. Ao arrastar uma porta ou janela, o comprimento
  encaixa nas larguras do quadro.
- **Escala.** Vem das cotas; a ferramenta Escala recalibra com dois cliques e
  uma distância.
- **Rascunho pelo detector de faces.** "Refazer com detector de faces"
  (painel da direita, `POST /api/plans/{id}/faces`) troca a anotação da tela
  por um rascunho lido direto do render (`wallextractor.faces`: paredes como
  pares de linhas paralelas a uma espessura estimada da imagem, portas e
  janelas pelas cores laranja/azul, paredes já divididas nos encontros e
  vãos). Menos de 1 s por planta. Entra como uma importação: salva como
  rascunho, mostra os avisos e Ctrl+Z volta. Feito para plantas coloridas
  de BIM, como o portfólio de SJC; em desenhos de outro estilo pode perder
  paredes (medição em `docs/experimentos.md`). Quando a planta não tem
  paredes como duas linhas paralelas (paredes preenchidas, cópias coloridas
  ou desbotadas, como a maioria das de Bauru) ou o resultado tem menos de 4
  paredes, o servidor recusa com essa explicação e a tela não muda.
- **Ajuda da IA (Claude, ChatGPT…).** "Pedir ajuda à IA" (painel da direita)
  salva a planta e baixa `ia_<título>.zip` (`GET /api/plans/{id}/ai`) com
  `planta.png` (render base), `planta_numerada.png` (anotação atual por cima,
  paredes w1… e aberturas o1… numeradas), `planta.json` (só paredes e
  aberturas, em pixels de `planta.png`), `LEIA-ME.md` (o que é parede, porta
  e janela, as convenções de eixo, espessura e vão, e um exemplo de
  resposta), `PROMPT.txt` (o pedido pronto para colar no chat; o modal do
  download também mostra esse texto com um botão "Copiar texto", vindo no
  cabeçalho `X-AI-Prompt` da resposta) e `conferir.py`
  (desenha a resposta sobre a planta, para o modelo conferir antes de
  entregar) e `recortes/` (a planta em pedaços de 700 px com 15% de
  sobreposição, ampliados 2x, com grade das coordenadas originais a cada
  50 px, para o modelo ler pixels exatos). O LEIA-ME fixa as convenções das
  correções: a parede é dividida em todo encontro (L, T, cruz) e vão; porta
  entre os batentes; janela só pelo caixilho. A resposta entra por "Importar correção" (arquivo) ou "Colar
  resposta" (texto do chat; o JSON é achado mesmo dentro de um bloco de
  código). A importação aceita `door`/`window` ou porta/janela, segmentos
  como `start`/`end`, `x1…y2` ou `[x1, y1, x2, y2, t]`, `doors`/`windows`
  separados, ids e `wall_id` ausentes (a abertura é ligada pela geometria) e
  coordenadas de uma imagem reduzida (reescala por `image.width`). Troca as
  paredes e aberturas da tela (escala, quadro e origem ficam), salva como
  rascunho e não muda o status; Ctrl+Z volta à anotação anterior. Depois
  da importação o painel lista avisos para conferir (parede sobre um vão,
  abertura sem parede na sua linha, segmento torto, ponto fora da imagem,
  parede curta demais); são só alertas, nada é bloqueado. Inclinação abaixo
  de 1° é deriva de escaneamento e não gera aviso; quando 40% ou mais das
  paredes estão inclinadas (foto ou escaneamento torto, perspectiva), vira um
  aviso só com a faixa de ângulos, e só inclinações isoladas são listadas.
- **Lotes para a IA.** Na fila, cada planta tem uma caixa de marcar ("todas"
  marca a lista do filtro). "Baixar lote para IA" (`POST /api/ai/batch`
  `{"ids": [...]}`, até 40 plantas) baixa um zip com uma pasta
  `<título>_<id>/` por planta, cada uma com o pacote completo, e na raiz
  `PROMPT_LOTE.txt` e `LEIA-ME_LOTE.md`: a IA corrige uma planta depois da
  outra sem pedir confirmação, grava `correcoes/<plan_id>.json` e
  `correcoes/progresso.txt`, entrega `correcoes.json` (`{"<plan_id>": correção}`)
  e, se for interrompida, retoma com "continue". Depois de todas as plantas
  (as correções vêm antes), entrega também `ferramentas.py` (os scripts
  usados, generalizados) e `relatorio_lote.md`, preenchendo um modelo com
  tabelas fixas: problemas do rascunho com plantas e ocorrências, o que um
  script resolve (função, parâmetros, onde falha), decisões que exigiram
  olhar a imagem, padrão x exceção e mudanças no rascunho por impacto. Um
  primeiro lote com o pedido em texto corrido devolveu só um resumo
  genérico; se ainda vier curto, pedir na mesma conversa que preencha o
  modelo. "Importar lote" aceita o
  `correcoes.json`, uma lista de correções com `plan_id` ou vários `.json`
  (o `plan_id` vem de dentro ou do nome do arquivo); cada correção entra como
  rascunho na sua planta, que continua pendente, e um resumo lista planta a
  planta o que entrou, as notas da IA e os avisos. Chave e `plan_id`
  divergentes, planta repetida no lote e planta que não está no editor
  aparecem como erro, sem importar.
- **IA no computador (Claude ou ChatGPT).** "Corrigir com Claude" e
  "Corrigir com ChatGPT" (no painel da planta) e as versões com (N) para as
  plantas marcadas na fila põem as plantas numa fila no servidor (`ai_jobs`
  no SQLite, com `engine` = `claude` ou `codex`). O editor não alcança o computador
  do usuário; quem trabalha é `scripts/claude_worker.py`, rodando no Lenovo:
  pede a próxima planta (`POST /api/ai/jobs/claim`), baixa o mesmo pacote do
  "Pedir ajuda à IA", grava ao lado um `verificar.py` (erro para parede
  passando por cima de vão, parede minúscula fora de batente, abertura sem
  parede alinhada, ponto fora da imagem; aviso para segmento torto e
  espessura fora do padrão) e roda `claude -p` na pasta, só com leitura,
  escrita na pasta e `python`, sem web; o ChatGPT roda pelo Codex CLI
  (`codex exec --sandbox workspace-write`, o pedido pela entrada padrão e as
  duas imagens inteiras anexadas). O worker usa a CLI que vem com o app do
  Codex (`%LOCALAPPDATA%\OpenAI\Codex\bin\*\codex.exe`) quando existe,
  porque a do npm global pode estar velha demais para o modelo da conta
  (erro "requires a newer version of Codex"). O Claude deve rodar o `verificar.py`
  até não sobrar erro. O worker confere de novo e devolve o `correcao.json`
  com um resumo (paredes, aberturas, minutos, passos, custo estimado, erros
  que sobraram, o resumo do Claude). Durante o trabalho, a cada 20 s, o
  painel mostra o último passo ("olhando recortes/r1c2…", "rodando python
  verificar.py…") e a fila marca a planta ("na fila do Claude", "Claude
  corrigindo", "Claude: pronta", "Claude ✓"). Quando a planta pronta é
  aberta, a correção entra como importação (com os avisos; Ctrl+Z desfaz) e
  a planta continua pendente até a revisão. "Cancelar" encerra o `claude` no
  próximo aviso de progresso; um job sem notícias por 15 min vira erro.
  "Carregar de novo" repõe a correção do Claude. O painel avisa quando nenhum
  computador pediu trabalho nos últimos 90 s.

  No Lenovo, com `EDITOR_TOKEN` no ambiente (a senha nunca vai na linha de
  comando nem para a sessão do Claude):

  ```
  python scripts/claude_worker.py              # fica pedindo trabalho até Ctrl+C
  python scripts/claude_worker.py --engines claude,codex   # Claude e ChatGPT
  python scripts/claude_worker.py --once       # uma planta e sai
  python scripts/claude_worker.py --model sonnet --timeout-min 30
  ```

  Cada job guarda a pasta (pacote, `correcao.json`, `claude.jsonl` ou
  `codex.jsonl` com a sessão inteira) em `~/.wallextractor/claude_jobs/`, fora do repositório,
  para a sessão não carregar o `CLAUDE.md` do projeto.
- **Exportação para treino.** `GET /api/export?status=corrected` devolve um
  zip com `<id>.png` (lado maior 1024 px) e `<id>.json` (WallPlan nas
  coordenadas dessa imagem, com metros). `scripts/fetch_corrections.py` baixa
  e `wallextractor.annotations` rasteriza as máscaras no formato do
  `train_seg`. A saída da máquina fica guardada ao lado da correção para
  medir a evolução do sistema.

## Girar

Clique com o botão direito sobre uma parede ou abertura. A ponta mais próxima
do clique fica marcada em laranja e serve de eixo para as opções "na ponta
marcada"; as opções "no centro" giram em torno do meio.

**Girar para a parede** (só em aberturas) resolve o caso da porta que a
máquina desenhou em cima da folha aberta: testa ±90° em torno de cada ponta
(a dobradiça) e do centro e escolhe a posição que cai no vão entre duas
paredes alinhadas. Se o resultado sair do lado errado, Ctrl+Z e use "Girar
90°" na ponta marcada.

Ao girar uma abertura, ela é encaixada na parede alinhada mais próxima; se não
houver nenhuma, fica solta (sem `wall_id`). Ao girar uma parede, as aberturas
dela giram junto.

## Copiar e colar

Ctrl+C copia a parede, porta ou janela selecionada; Ctrl+V cola centrada onde
está o mouse (ou use "Colar aqui" no botão direito). Porta e janela coladas
sobre uma parede, inclusive no vão entre duas paredes alinhadas, entram
alinhadas a ela, com o mesmo código e largura; fora de parede ficam soltas.
A cópia guarda a escala de origem, então uma janela copiada de uma planta
cola com a mesma medida em metros em outra. Ctrl+D duplica ao lado: parede
paralela, abertura na mesma parede; não altera o que está copiado.

## Atalhos

| Tecla | Ação |
|---|---|
| V / W / D / J / C | selecionar, parede, porta, janela, escala |
| Delete | excluir seleção |
| Ctrl+Z / Ctrl+Y | desfazer / refazer |
| [ / ] | espessura da parede selecionada −1 / +1 cm |
| R / Shift+R | girar a seleção 90° no sentido horário / anti-horário, em torno do centro |
| Ctrl+C / Ctrl+V | copiar a seleção / colar onde está o mouse |
| Ctrl+D | duplicar ao lado |
| botão direito | menu da parede ou abertura: girar no centro ou na ponta marcada (a mais próxima do clique), outro ângulo, girar para a parede, copiar, duplicar, colar, dividir, excluir; num ponto vazio, colar aqui |
| Esc | cancelar desenho ou seleção e voltar a Selecionar |
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
3. Variáveis de ambiente: `EDITOR_TOKEN` (obrigatória, é a senha da página).
   O modelo para plantas raster fica em `/data/model.onnx` (ou no caminho de
   `EDITOR_MODEL`) e é enviado com `PUT /api/model`, sem redeploy.
4. Porta 8000, domínio com HTTPS pelo próprio EasyPanel.
5. Memória: o contêiner usa entre 300 e 600 MB em uso normal. Fica dentro dos
   6 a 8 GB do servidor com folga.

### Deploy atual

O editor está no ar desde 24/09/2026:

- Painel: projeto `botai`, serviço **App** `wallextractor-editor`.
- URL: https://botai-wallextractor-editor.uuclvw.easypanel.host
- Origem: `github.com/maxrosan/WallExtractor`, branch `main`, build pelo `Dockerfile` da raiz.
- Volume `wallextractor-data` montado em `/data` (PDFs, renders e o SQLite).
  O banco é SQLite nesse volume; o Postgres (`PGDB`) ainda não é usado.
- Variáveis do serviço: `EDITOR_TOKEN`, `EDITOR_DATA`, `PYTHONUNBUFFERED`.
  A senha fica só no painel; não copie para o repositório.
- Fila inicial: as quatro plantas brasileiras, todas *pendente*.

### Redeploy pela API do painel

A API do EasyPanel está descrita em `$EASYPANEL_URL/api/openapi.json`. As
chamadas usam `Authorization: Bearer $EASYPANEL_TOKEN`. Deixe no ambiente,
fora do repositório:

```
EASYPANEL_URL=https://...        # URL do painel, em HTTPS
EASYPANEL_TOKEN=...              # token de API do painel
EASYPANEL_PROJECT=botai
```

```
H="Authorization: Bearer $EASYPANEL_TOKEN"
Q="projectName=$EASYPANEL_PROJECT&serviceName=wallextractor-editor"

# a sessão vale?
curl -sS "$EASYPANEL_URL/api/getSession" -H "$H"

# configuração do serviço (fonte, branch, volume). A resposta traz env com a
# senha: não imprima nem salve esse JSON inteiro.
curl -sS "$EASYPANEL_URL/api/inspectAppService?$Q" -H "$H" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["source"], d["mounts"])'

# redeploy (puxa o último commit da branch configurada e reconstrói a imagem)
curl -sS -X POST "$EASYPANEL_URL/api/deployAppService" -H "$H" \
  -H 'Content-Type: application/json' \
  -d "{\"projectName\":\"$EASYPANEL_PROJECT\",\"serviceName\":\"wallextractor-editor\",\"forceRebuild\":true}"

# acompanhar: a última ação do tipo deployment deve chegar a status "done"
curl -sS "$EASYPANEL_URL/api/listActions?$Q&limit=3" -H "$H"
```

Para trocar a branch, altere a fonte do serviço no painel antes de
redeployar. Depois do deploy, confira:

```
E=https://botai-wallextractor-editor.uuclvw.easypanel.host
curl -sS "$E/api/health"                          # {"ok":true,"plans":N}, sem token
curl -sS "$E/api/plans" -H "X-Token: $EDITOR_TOKEN"   # 401 sem o header
```

A senha pode ser lida do painel sem ser exibida:

```
EDITOR_TOKEN=$(curl -sS "$EASYPANEL_URL/api/inspectAppService?$Q" -H "$H" | python3 -c \
  'import json,sys; e=json.load(sys.stdin)["env"]; print(dict(l.split("=",1) for l in e.splitlines() if "=" in l)["EDITOR_TOKEN"])')
```

Como o SQLite e os PDFs ficam no volume `/data`, o redeploy mantém a fila e
as correções.

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
| `GET /api/plans/{id}/pdf` | PDF original (para rodar o extrator offline contra as correções) |
| `GET /api/plans/{id}/tile?x&y&w&h&s` | recorte re-renderizado do PDF para zoom |
| `GET /api/export?status=corrected` | zip de pares de treino |
| `GET /api/model` | modelo raster em uso: presença, tamanho, sha256 |
| `PUT /api/model` (multipart `file`) | troca o modelo raster (ONNX); validado antes de substituir o atual |
| `POST /api/ai/jobs` `{"ids": [...], "engine": "claude" \| "codex"}` | põe plantas na fila da IA no computador |
| `GET /api/ai/jobs` | jobs (o mais novo primeiro) e computadores vistos (`seen_s`) |
| `POST /api/ai/jobs/claim` `{"worker": nome, "engines": [...]}` | o worker pega o job mais antigo das filas que atende |
| `POST /api/ai/jobs/{id}` `{"message"}` / `{"status": "done", "result"}` / `{"status": "error" \| "cancelled"}` / `{"applied": true}` | progresso, fim, cancelamento, correção carregada |

Todas as rotas `/api` exigem o header `X-Token` (ou `?token=`) quando
`EDITOR_TOKEN` está definido.
