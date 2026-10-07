# Experimentos

Registro dos treinos, com custo. Objetivo da primeira rodada: verificar se a
ideia (segmentação leve para geometria, rodando em CPU) converge antes de
aumentar o dataset.

## E1: convergência com 400 plantas, 512 px (2026-09-24)

| Item | Valor |
|---|---|
| Dados | CubiCasa5K, 400 treino / 100 val (primeiras entradas de train.txt e val.txt), só `F1_scaled.png` + `model.svg` |
| Modelo | SegFormer MiT-B1, 4 classes (fundo, parede, porta, janela), pré-treino ImageNet |
| Entrada | lado maior redimensionado para 512 px, crop aleatório com zoom em 50% das amostras, flips e rotações de 90° |
| Treino | 15 épocas, batch 8, AdamW 6e-5, decaimento polinomial, bf16, pesos de classe 0,5/1/2/2 |
| GPU | RTX A4500 20 GB, US$ 0,19/h |
| Tempo | 136 s de treino; job inteiro (download com rate limit + conversão + treino + export) ~15 min |
| Custo | ~US$ 0,08 acumulados no pod até o fim desta rodada |

IoU na validação (100 plantas) por época:

| Época | Parede | Porta | Janela | Loss treino |
|---|---|---|---|---|
| 1 | 0,231 | 0,000 | 0,029 | 1,041 |
| 2 | 0,417 | 0,009 | 0,265 | 0,734 |
| 3 | 0,448 | 0,063 | 0,346 | 0,560 |
| 5 | 0,469 | 0,100 | 0,399 | 0,390 |
| 8 | 0,507 | 0,226 | 0,427 | 0,303 |
| 10 | 0,516 | 0,207 | 0,443 | 0,276 |
| 12 | 0,525 | 0,245 | 0,470 | 0,254 |
| 15 | **0,533** | 0,254 | 0,473 | 0,244 |

Leitura:

- **Converge.** Loss cai de forma monótona e o IoU de parede sobe a cada
  época sem sinal de saturação em 15 épocas. Não há overfitting visível:
  o IoU de validação ainda melhora quando a loss de treino chega a 0,24.
- **O teto está na resolução, não nos dados.** A 512 px uma parede interna
  do CubiCasa5K tem 3 a 5 px de espessura; um erro de 1 px na borda custa
  20 a 30% de IoU. Isso explica o 0,53 mais do que o tamanho do dataset.
- **Porta é a classe difícil** (0,25): é pequena e o desenho da folha da
  porta confunde com parede. Janela vai melhor (0,47) porque tem preenchimento
  próprio.
- **GPU ociosa.** 136 s para 15 épocas significa que dá para treinar 10 vezes
  mais longo, em resolução maior, por centavos. A rodada E2 faz isso.

Artefatos: `best.onnx` com 55 MB (MiT-B1 em fp32; int8 deve dar ~14 MB),
carregável no servidor só com `onnxruntime`.

## E2: 768 px, 40 épocas, mesmos 400 dados

Mesmo pod e dados de E1; batch 4, 472 s de treino. Melhor época: 38.

| | Parede | Porta | Janela |
|---|---|---|---|
| E1 (512 px, 15 ép.) | 0,533 | 0,254 | 0,473 |
| E2 (768 px, 40 ép.) | **0,646** | **0,451** | **0,637** |

A resolução era mesmo o gargalo: +11 pontos em parede e quase o dobro em
porta, com os mesmos 400 exemplos.

## E3: E2 + augmentação de estilo (paredes redesenhadas)

`wallextractor/styles.py` redesenha os pixels de parede em hachura 45°/135°,
hachura cruzada, cinza ou só contorno, mantendo o resto da imagem. Treino
igual ao E2 com `--restyle-prob 0.5` (metade das amostras em estilo
aleatório). 965 s de treino. Avaliação da validação com as paredes
redesenhadas em cada estilo (`wallextractor.eval_seg`):

| Estilo da validação | E2 parede | E3 parede | E2 porta | E3 porta | E2 janela | E3 janela |
|---|---|---|---|---|---|---|
| sólido (original) | 0,646 | 0,634 | 0,451 | 0,443 | 0,637 | 0,618 |
| hachura 45° | 0,700 | **0,889** | 0,448 | 0,504 | 0,646 | 0,713 |
| hachura 135° | 0,697 | **0,889** | 0,448 | 0,505 | 0,644 | 0,716 |
| só contorno | 0,551 | **0,766** | 0,412 | 0,471 | 0,604 | 0,677 |
| cinza | 0,664 | **0,857** | 0,453 | 0,501 | 0,636 | 0,702 |
| cruzada | 0,732 | **0,889** | 0,449 | 0,504 | 0,659 | 0,716 |

Leitura:

- A augmentação custa 1 ponto no estilo original e ganha de 19 a 22 pontos
  nos outros. O modelo passa a ser indiferente ao estilo da parede.
- Os estilos sintéticos dão IoU maior que o original porque o redesenho
  produz bordas limpas alinhadas à máscara, enquanto o CubiCasa5K original
  tem borda anti-aliased e anotação com folga. Serve para medir robustez a
  estilo, não como número absoluto.
- Contorno é o estilo mais difícil (0,77): o interior branco da parede se
  confunde com o cômodo. É o caso das plantas brasileiras vetoriais, ver
  abaixo.

Custo acumulado do pod nas três rodadas mais avaliações: ~US$ 0,17.

## E4: 4x mais dados + correções do editor, GPU local (2026-10-02)

Primeira rodada fora do RunPod: PC "Pichau" (RTX 5060 Ti 16 GB, Blackwell),
Docker dentro do WSL2 com a imagem `pytorch/pytorch:2.8.0-cuda12.8` (CUDA
12.8 é o mínimo para essa GPU). Mesma receita de E3 (MiT-B1, 768 px,
40 épocas, batch 4, `--restyle-prob 0.5`), com:

- CubiCasa5K 1.600 treino / 400 validação (E1-E3: 400 / 100);
- as 11 plantas brasileiras corrigidas no editor, 9 no treino (repetidas
  20x, 180 amostras) e 2 na validação.

Download do CubiCasa: 89 min (limite de taxa do Zenodo, 2,7 s por planta).
Treino: 50 min (75 s por época). Custo: só energia.

| | Parede | Porta | Janela | Parede contorno | Parede hachura 45° |
|---|---|---|---|---|---|
| E2 | 0,646 | 0,451 | 0,637 | 0,551 | 0,700 |
| E3 | 0,634 | 0,443 | 0,618 | 0,766 | 0,889 |
| **E4** (época 40) | **0,747** | **0,562** | **0,723** | **0,895** | **0,945** |

A validação mudou (402 plantas em vez de 100, incluindo 2 brasileiras), então
a comparação é indicativa. A melhor época foi a última: o modelo ainda
melhorava, e mais épocas devem render mais. Modelo em
`/root/we/results_e4/best.onnx` (55 MB) no Pichau; script em `/root/we/run.sh`.

**Só nas plantas brasileiras** (IoU por pixel, `wallextractor.eval_seg`, estilo
original):

| | Parede | Porta | Janela |
|---|---|---|---|
| 2 fora do treino (SJC m06-c, casa habitacional) | 0,758 | 0,353 | 0,642 |
| 9 usadas no treino (20x) | 0,884 | 0,681 | 0,745 |

O número honesto é o das 2 fora do treino: parede e janela no nível do
CubiCasa, porta bem abaixo (0,35). As 9 do treino mostram que o modelo
consegue aprender o desenho brasileiro; com mais plantas corrigidas, a
distância entre as duas linhas deve cair.

**No editor.** O modelo foi publicado em `/data/model.onnx` pelo
`PUT /api/model` (sha256 `aad7ef1d…`). Teste com o m06-c rasterizado (PDF só
imagem, fora do treino): 6 s no servidor, paredes e janelas da planta no
lugar, a maioria das portas no vão. Mas o ramo raster segmenta a folha
inteira: moldura, faixas do carimbo e brasão viram "paredes" (149
segmentos), e a planta, que ocupa um terço da folha, entra no modelo
reduzida junto com a folha a 768 px. Próximo passo do ramo raster: recortar
a região da planta antes de segmentar e segmentar em blocos na resolução
cheia.

## Plantas brasileiras reais (2026-09-24)

Quatro PDFs de projetos residenciais (um repetido) foram recebidos para
teste. Ficam fora do repositório por conterem dados pessoais.

| Arquivo | Prancha | Páginas | Primitivas | Observação |
|---|---|---|---|---|
| ARQ_MINHA_CASA_MINHA_VIDA | A2 1684×1191 pt | 1 | 5.124 | só a planta, sem carimbo |
| ARQ_CASA_HABITACIONAL | A3 1191×842 pt | 6 | 6.480 na p1 | planta baixa + localização + quadro de esquadrias + carimbo |
| PROJETO_RESIDENCIAL PR_01 | A1 2384×1684 pt | 1 | 18.742 | planta baixa + cobertura + localização |
| PROJETO_RESIDENCIAL PR_02 | A3, rotação 270° | 1 | 16.649 | planta baixa + cobertura |

O que isso muda no plano:

1. **Todos são vetoriais**, então o ramo vetorial é o caminho principal.
2. **A parede não é polígono preenchido nem traço grosso.** Nos objetos de
   desenho ela aparece como polilinhas de contorno com traço de 0,7 a 1 pt
   e o preto que se vê no render vem de linhas paralelas coladas ou hachura.
   As regras atuais de `pdf.wall_candidates_from_primitives` acharam de 2 a
   31 paredes por planta, ou seja, não servem para esse estilo. Precisa de
   agrupamento de linhas paralelas próximas em pares (parede = duas linhas
   paralelas a 13–20 cm), como no trabalho da Universidade de Alberta.
3. **A planta baixa ocupa uma fração da prancha.** Renderizar a folha inteira
   em 1024 px deixa a parede com 2 px. É preciso localizar a região da
   planta baixa primeiro. O texto do PDF tem "PLANTA BAIXA" e "ESC. 1:75" /
   "ESCALA 1:50" com posição, o que dá a âncora e a escala.
4. **As cotas estão no texto** (dezenas de números no formato 3.62), o que
   permite estimar px por metro por consenso, sem VLM.
5. Para o ramo raster, o estilo real é "sólido fino" com contorno; E3 mostra
   que a augmentação já cobre isso razoavelmente.

### Ramo vetorial v1 (`wallextractor/vector_walls.py`)

Implementado no mesmo dia, em cima do que os PDFs mostraram:

1. **Localização da planta baixa na prancha.** Os traços da pena mais grossa
   são agrupados em blocos; o bloco escolhido é o mais próximo de um rótulo
   "PLANTA BAIXA" exato (o carimbo repete "PLANTA BAIXA, PLANTA DE
   COBERTURA..." e é descartado pela vírgula e pela pontuação por tinta).
   Sem rótulo, vale o bloco com mais tinta. Cobertura, fachadas, cortes e
   localização ficam de fora.
2. **Pena de parede.** A espessura de traço mais grossa com pelo menos 10
   segmentos dentro da região (0,96 a 1,56 pt nos quatro arquivos). Portas e
   janelas usam pena mais fina.
3. **Pareamento de paralelas.** Dois segmentos da pena de parede, paralelos
   a menos de 2° e distantes entre 8 e 40 cm, viram uma parede na linha média
   do trecho em que se sobrepõem, com espessura igual à distância. Depois os
   trechos colineares são fundidos.
4. **Escala por consenso das cotas.** Para cada número no formato 3.62 dentro
   da região, a linha fina mais próxima com a mesma orientação (fundindo os
   pedaços partidos em volta do texto) dá um candidato comprimento/valor; a
   mediana do maior grupo concordante é a escala. **Ela vale mais que o
   rótulo**: três das quatro pranchas foram plotadas com "ajustar à página".

| Arquivo | Rótulo | Escala nominal | Escala efetiva (cotas) | Paredes | Comprimento total | Espessura mediana | Cotado no desenho |
|---|---|---|---|---|---|---|---|
| MINHA_CASA_MINHA_VIDA | nenhum | nenhuma | 1:52,6 (22 cotas) | 31 | 93,5 m | 0,130 m | .13 |
| CASA_HABITACIONAL | achado | 1:75 | 1:69,2 (23 cotas), 108% | 31 | 104,6 m | 0,130 m | .13 |
| PROJETO_RESIDENCIAL PR_01 | achado | 1:50 | 1:51,4 (33 cotas), 97% | 34 | 79,7 m | 0,122 m | .12 |
| PROJETO_RESIDENCIAL PR_02 (rot. 270°) | achado | 1:100 | 1:91,1 (34 cotas), 110% | 42 | 102,8 m | 0,120 m | .12 |

A espessura mediana extraída coincide com a espessura cotada nos quatro
desenhos, o que valida escala e pareamento ao mesmo tempo. Nos overlays as
paredes internas e externas e o muro do lote aparecem no lugar; faltam
alguns trechos curtos junto a portas e ainda não há portas e janelas no
ramo vetorial (elas estão na pena de 0,72 pt e são o próximo passo).

Custo: zero. Tudo roda em CPU em menos de 2 s por prancha.

### Portas e janelas no ramo vetorial (`find_openings`)

O que os desenhos mostraram: a porta é desenhada fechada, como um retângulo
fino de 0,7 a 0,8 m paralelo à parede e deslocado uns 0,2 m do eixo, com o
arco de giro tracejado (ou em polilinha), e as linhas da parede continuam
por trás; a janela é um grupo de 2 a 3 linhas de pena intermediária dentro
da faixa da parede, e às vezes é desenhada com a própria pena da parede
(vira um "pedaço de parede" curto). Toda esquadria leva uma etiqueta "P2"
ou "J1" do quadro de esquadrias, colocada a menos de 0,5 m do vão.

Regras, em ordem:

1. Candidatos por parede: folha fechada (par de linhas paralelas à parede,
   fora da faixa, 0,6–1,3 m, confirmada por ≥3 pedaços de arco no círculo de
   raio igual à folha em torno da dobradiça); folha aberta (segmento
   perpendicular com arco encadeado ou ≥5 pedaços no círculo); caixilho
   (≥2 linhas com extremos coincidentes dentro da faixa); vão entre paredes
   colineares.
2. Cada etiqueta é casada com o candidato de menor custo (distância mais
   penalidade se as pistas não combinam com o tipo). Quando o desenho tem
   etiquetas, só candidatos etiquetados viram abertura; sem etiquetas, valem
   as pistas geométricas.
3. Etiqueta sem candidato vira abertura de confiança 0,5 na parede mais
   próxima: um pedaço curto de parede ao lado dela é a própria janela
   desenhada com a pena da parede.

| Arquivo | Etiquetas P / J | Portas / janelas extraídas | Confiança 0,5 |
|---|---|---|---|
| MINHA_CASA_MINHA_VIDA | 3 / 3 | 3 / 3 | 0 |
| CASA_HABITACIONAL | 7 / 7 | 7 / 7 | 0 |
| PROJETO_RESIDENCIAL PR_01 | 7 / 7 | 7 / 7 | 0 |
| PROJETO_RESIDENCIAL PR_02 | 7 / 7 | 7 / 7 | 2 |

Contagem e posição batem com as etiquetas nas quatro plantas.

**Larguras (correção das fusões).** A primeira versão unia folhas
sobrepostas (folha + batente + móvel encostado) e caixilhos que só tocavam a
ponta da parede, dando portas de 1,3 a 2,8 m. Agora: duas folhas sobrepostas
nunca se fundem, ficam como candidatas separadas e a etiqueta escolhe a de
menor custo (com arco de giro, largura entre 0,55 e 1,05 m); caixilho só vale
se pelo menos 70% dele está sobre a parede; interrupção acima de 2,5 m não é
janela (é passagem ou frente de garagem); e a etiqueta sem candidato usa o
pedaço curto de parede só se ele tem entre 0,4 e 1,2 m, senão largura padrão
(0,80 porta, 0,60 janela) centrada na etiqueta.

Larguras extraídas contra o quadro de esquadrias, onde ele existe:

| Arquivo | Portas (quadro) | Portas extraídas | Janelas (quadro) | Janelas extraídas |
|---|---|---|---|---|
| CASA_HABITACIONAL | P1 0,70 ×2, P2 0,80 ×4, P3 1,00 | 0,70 ×2, 0,80 ×4, 0,80? | J1 0,40 ×2, J2 0,50 ×2, J3 1,40, J4 1,20 ×2 | 0,61 ×2, 0,50, 0,60?, 1,13, 1,20, 1,50 |
| PROJETO_RESIDENCIAL PR_01/02 | P1 0,60/0,70, P2 0,70, P3, P4 | 0,62–0,72, 1,03 (P4) | J1 0,60, J4 1,60 | 0,61, 1,64 |
| MINHA_CASA_MINHA_VIDA | sem quadro na prancha | 0,80 ×2, 0,80 | sem quadro | 0,50, 1,23, 1,40 |

"?" marca abertura de confiança 0,5 (só etiqueta). Portas ficam dentro de
±0,05 m do quadro; janelas por vão saem 0,2 a 0,3 m mais largas que o
quadro porque o vão inclui o peitoril/batente, e as basculantes pequenas
(J1 0,40) saem com o retângulo desenhado (0,61). Para o produto, a largura
do quadro de esquadrias vale mais que a geometria: o próximo passo natural é
ler a tabela "QUADRO DE ESQUADRIAS" do PDF (é texto) e sobrescrever a largura
pelo código da etiqueta.

Sem etiquetas no desenho o sistema ainda funciona só com as pistas
geométricas, com mais falsos positivos em móveis encostados nas paredes.

### Quadro de esquadrias e dúvidas (`wallextractor/schedule.py`)

O quadro é lido do texto do PDF em qualquer página: um código ("P1", "J3")
fora da região do desenho seguido, na mesma linha, de pelo menos dois
números no formato 0,70 (LARG, ALT, opcionalmente PEIT), depois TIPO,
MATERIAL e QUANT. A largura do quadro sobrescreve a geometria
(`width_source = "schedule"`), e altura, peitoril e tipo vão para o JSON.

| Arquivo | Quadro | Efeito |
|---|---|---|
| CASA_HABITACIONAL | 7 linhas lidas como texto | As 14 aberturas com largura do quadro, quantidades batem (sem dúvidas) |
| PROJETO_RESIDENCIAL PR_01 / PR_02 | tabela desenhada em contorno (928 paths, 0 palavras) | Larguras da geometria, dúvida gerada |
| MINHA_CASA_MINHA_VIDA | sem quadro na prancha | Larguras da geometria, dúvida gerada |

**Dúvidas para o humano.** O JSON traz `questions`: quadro ilegível (lista
as larguras por código para confirmar), quantidade do quadro diferente da
encontrada, e aberturas posicionadas só pela etiqueta. As respostas vão num
JSON `{"P1": 0.70, "J1": 0.40}` passado ao `infer --answers`, que vale mais
que quadro e geometria (`width_source = "answer"`). Esses arquivos de
resposta são a anotação humana do conjunto de teste e o material para o
fine-tuning futuro do Qwen (imagem da planta + JSON corrigido).

Próximos passos: (a) fechar os trechos curtos de parede nos cantos;
(b) OCR ou reconhecimento dos glifos em contorno para ler quadros desenhados
como paths; (c) coletar as respostas e montar `data/annotations/` (fora do
repositório) como conjunto de teste.

## V1: portas e janelas do ramo vetorial contra as correções do editor (2026-09-25)

Gabarito: as 4 plantas brasileiras corrigidas no editor (28 portas, 27 janelas).
As paredes da máquina não precisaram de nenhuma correção (138 de 138); todo o
erro estava nas aberturas. Medida com `scripts/eval_openings.py`: acerto quando
o tipo coincide e o centro e a largura ficam dentro da tolerância, com direção
dentro de 20°.

| | Portas certas | Portas erradas | Portas giradas | F1 portas | F1 janelas |
|---|---|---|---|---|---|
| Antes (tol. 15 cm) | 4 / 28 | 20 | 18 | 0,15 | 0,71 |
| Depois (tol. 15 cm) | 19 / 28 | 5 | 2 | 0,73 | 0,71 |
| Depois (tol. 25 cm) | 22 / 28 | 2 | 2 | 0,85 | 0,71 |

A tolerância de 25 cm é a que vale para comparar: parte do gabarito foi feita
com "Girar para a parede", que gira em torno da ponta da folha desenhada pela
máquina (sobre o eixo da parede), e não da dobradiça real. Essas portas ficaram
10 a 17 cm deslocadas no gabarito, e a nova extração cai no vão real.

Causa: nas plantas brasileiras a porta costuma ficar ao lado de um canto, com o
vão indo do fim de uma parede até a face da parede perpendicular. (1) O detector
de folha aberta exigia a dobradiça sobre o trecho de parede e descartava essas
portas; como não há parede alinhada do outro lado, também não havia "vão". (2) A
folha aberta fica paralela à parede do canto, a 15-20 cm dela, e o detector de
folha fechada a tomava por uma porta fechada naquela parede: a porta girada.

Correção em `find_openings`: a folha aberta é detectada primeiro, com a
dobradiça onde a folha encontra a face da parede (a linha da folha pode passar
dela) e aceita além do fim da parede quando o vão encosta nesse fim; o arco
bezier dá a outra ombreira; num arco em polilinha, o lado é o quarto de círculo
com mais fatias de 10° preenchidas (antes, a contagem de peças era vencida por
maçaneta e hachuras). Linhas usadas como folha aberta não entram mais no
detector de folha fechada.

Testado e descartado: (a) manter candidatos sem etiqueta com folha e arco em
desenhos etiquetados: recuperaria 3 portas da Minha Casa Minha Vida, mas criou
50+ falsos positivos (móveis e louças com arco); (b) aceitar caixilho de janela
além do fim da parede (mesmo caso de canto, janelas J3 dos dois projetos residenciais PR_01 e PR_02):
piorou janelas e portas porque os candidatos novos disputam as etiquetas.

Ficam de fora: portas e janelas sem etiqueta (Minha Casa Minha Vida), janelas
de canto (J3), janelas estreitas de 0,37-0,40 m e duas portas ainda giradas.

## V2: aberturas fora de parede e janelas no vão (2026-09-25)

Gabarito ampliado: as 4 plantas brasileiras de escritório mais 7 plantas
populares da Prefeitura de São José dos Campos (m01, m04, m06-c, m09, m12,
m13-a, m15-b, de `docs/datasets.md`), corrigidas no editor. 74 portas e 62
janelas. As paredes seguem sem nenhuma correção nas 11 plantas.

| tol. 25 cm | Portas certas | Portas erradas | Giradas | F1 portas | Janelas certas | Janelas erradas | F1 janelas |
|---|---|---|---|---|---|---|---|
| V1 | 57 / 74 | 33 | 10 | 0,70 | 18 / 62 | 6 | 0,42 |
| V2 | 67 / 74 | 27 | 2 | 0,80 | 45 / 62 | 11 | 0,76 |

Nas 4 plantas de escritório a V2 mantém as portas (22 / 28, 2 erradas) e
perde 2 janelas (18 → 16): um candidato novo de caixilho sem parede toma a
etiqueta J2 de banheiro num ponto 25 cm fora do centro marcado.

O desenho de SJC é de outro tipo: sem etiquetas P/J, sem quadro de
esquadrias, e o extrator interrompe a parede em cada abertura. Três
convenções novas em `find_openings`:

1. **Janela no vão.** Quatro linhas de pena média, paralelas e dentro da
   faixa da parede, com pontas coincidentes, no vão entre duas paredes
   alinhadas. Antes, o caixilho só contava sobre um trecho de parede.
2. **Porta atravessando o fim de uma parede.** A folha desenhada ao lado da
   parede, com a dobradiça no fim dela, e o arco da folha até a outra
   ombreira, na perpendicular ao fim da parede: é a porta entre duas paredes
   paralelas, sem parede desenhada ao longo do vão. Sai com `wall_id`
   vazio (o editor a alinha a uma parede se houver).
3. **Janela numa linha sem parede.** Três ou mais linhas paralelas dentro
   de uma espessura de parede, com pontas coincidentes e as duas pontas
   encostando em parede: o trecho inteiro entre duas paredes é abertura.

Também: caixilho de 3+ linhas ao lado de uma folha fica como janela
separada, em vez de se fundir à porta; e folha com arco vence duas linhas
no vão (porta desenhada fechada) quando descrevem a mesma abertura.

Testado e descartado: rejeitar folhas com duas ou mais "gêmeas" paralelas de
mesmo comprimento (piso cerâmico desenhado com a mesma pena da folha): tirou
3 portas falsas e 6 verdadeiras.

Ficam de fora: nas plantas de SJC, folhas falsas em armários e louças (os
27 erros de porta são quase todos aí); janelas estreitas de 0,37-0,40 m;
janelas de canto (J3) e aberturas sem etiqueta nas plantas de escritório.

## V2 nas 22 plantas novas de SJC (2026-10-03)

As 22 plantas restantes de SJC (m02-a a m15-a) foram corrigidas no editor
sobre a saída da V2; total agora de 33 plantas. Nenhuma parede corrigida em
nenhuma das 33. Código inalterado: só medição
(`scripts/eval_openings.py data/gt --tol 0.25`).

| tol. 25 cm | Portas certas | Portas erradas | Giradas | F1 portas | Janelas certas | Janelas erradas | F1 janelas |
|---|---|---|---|---|---|---|---|
| 11 antigas | 67 / 74 | 27 | 2 | 0,80 | 45 / 62 | 11 | 0,76 |
| 22 novas | 143 / 144 | 69 | 2 | 0,80 | 93 / 105 | 0 | 0,94 |
| 33 | 210 / 218 | 96 | 4 | 0,80 | 138 / 167 | 11 | 0,87 |

Nas 22 novas a revocação de portas é quase total e não há janela falsa; o
problema é a precisão das portas: 69 falsas, de 0 a 7 por planta (m02-d, m06-a,
m06-f, m04, m13-a com 6-7). Todas sem etiqueta P (o desenho de SJC não tem),
o que confirma o próximo alvo: folhas falsas em armário, louça e piso. Nas
janelas, os 12 erros são omissões.

## V3: arco da porta só na cor da folha (2026-10-03)

Diagnóstico das 96 portas falsas da V2 nas 33 plantas: 93 vinham do recurso
de "arco em polilinha" (contar fatias de 10° com pedaços curtos sobre o
círculo da folha), que aceitava pedaços de qualquer camada. Nas plantas de
SJC o piso (ciano) e os algarismos das cotas (roxo, em traços) enchiam o
círculo. O caminho do arco encadeado (Bézier) tinha 116 portas certas e 2
erradas.

Mudança: `Segment` e `page_curves` passam a carregar a cor do traço, e no
recurso de polilinha só contam pedaços da mesma cor da folha (cor
desconhecida casa com qualquer uma). Paredes iguais nas 33 plantas e
`schema.validate` sem erros.

| tol. 25 cm, 33 plantas | Portas certas | Portas erradas | F1 portas | Janelas certas | Janelas erradas | F1 janelas |
|---|---|---|---|---|---|---|
| V2 | 210 / 218 | 96 | 0,80 | 138 / 167 | 11 | 0,87 |
| V3 | 167 / 218 | 11 | 0,84 | 142 / 167 | 17 | 0,87 |

A queda de portas certas é quase toda do gabarito: das 43 portas que a V3
deixa de achar, revisadas no recorte, 39 não existem no desenho (parede de
banheiro sem abertura, ou janela "1.00X0.60"/"1.50X1.10"). São portas falsas
da V2 que ficaram nas correções de m11, m12, m13-b, m14-a, m14-b, m15-a e
m15-b. As outras 4 são portas reais com a folha vermelha e o arco laranja
(camadas diferentes), que só passavam antes pelo ruído do piso.

Testado e descartado:
- exigir peças tangentes ao círculo (|corda · raio| pequeno): com tolerância
  0,2 deixou 27 portas falsas mas perdeu 40 certas, e o arco das portas de
  SJC é uma Bézier só, não peças tangentes;
- aceitar também as cores das curvas longas da página, ou curvas de qualquer
  cor: as louças são curvas laranja, e as falsas voltaram a 95;
- dividir cada Bézier em 6 cordas: nenhuma porta a mais, 6 falsas a mais.

Conclusão para o projeto: a heurística muda de estilo para estilo de
desenho (camadas, cores, como o arco é traçado). O caminho principal passa a
ser o modelo treinado nas correções (E5 e o Qwen-VL); o ramo vetorial fica
como pré-anotação para o editor.

### Gabarito limpo (2026-10-03)

As 39 portas inexistentes foram removidas das correções no editor pela API
(backup das plantas alteradas em `data/backup_before_rm/`, fora do git). As
4 portas reais ficaram. Novo gabarito: 179 portas, 167 janelas.

| tol. 25 cm, 33 plantas | Portas certas | Portas erradas | F1 portas | Janelas certas | Janelas erradas | F1 janelas |
|---|---|---|---|---|---|---|
| V2 | 171 / 179 | 135 | 0,71 | 138 / 167 | 11 | 0,87 |
| V3 | 167 / 179 | 11 | 0,94 | 142 / 167 | 17 | 0,87 |

## E5: E4 + 33 correções limpas, 120 épocas, GPU local (2026-10-03)

Mesma receita do E4 (CubiCasa 1600/400, 768 px, MiT-B1, restyle 0,5) com as
33 plantas corrigidas no editor depois da limpeza das 39 portas falsas
(26 no treino x20, 7 na validação) e 120 épocas em vez de 40. 2 h 49 min
na RTX 5060 Ti. Melhor época: 115 (a curva quase não sobe depois da 60:
IoU de parede 0,724 / 0,757 / 0,762 / 0,761 / 0,765 / 0,770 nas épocas 20 a
120).

| IoU | parede | porta | janela | mIoU |
|---|---|---|---|---|
| E4, validação (CubiCasa + 2 BR) | 0,747 | 0,562 | 0,723 | 0,677 |
| E5, validação (CubiCasa + 7 BR) | 0,772 | 0,622 | 0,757 | 0,717 |
| E4 nas 7 BR de validação do E5 | 0,837 | 0,522 | 0,714 | 0,691 |
| E5 nas 7 BR de validação | 0,894 | 0,673 | 0,724 | 0,764 |

O E4 pode ter visto no treino algumas das 7 plantas (a divisão de 11 era
outra) e mesmo assim perde. Nos estilos redesenhados o E5 também sobe
(parede contorno 0,936, hachura 0,964). Publicado no editor
(`PUT /api/model`, sha256 `54466fea…`).

## Q1: Qwen3-VL-4B + LoRA, 2 épocas, GPU local (2026-10-03)

`wallextractor.vlm_data` (CubiCasa 1600 com restyle 0,5 + 26 brasileiras
x20, lado 1024) → `train_vlm` (LoRA r16 só no LLM, lr 1e-4, acumulação 8,
2 épocas, 2 h 29 min na RTX 5060 Ti, pico de 16,3 GB) → `eval_vlm`.

| | perda treino | perda validação |
|---|---|---|
| época 1 | 0,596 | 0,602 |
| época 2 | 0,446 | 0,586 |

F1 por tolerância (fração do lado maior da imagem):

| | JSON válido | parede @1,5% | porta @1,5% | janela @1,5% | parede @5% | porta @5% | janela @5% |
|---|---|---|---|---|---|---|---|
| Base sem treino, 7 BR | 0 / 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| Q1, 7 BR de validação | 7 / 7 | 0,32 | 0,34 | 0,27 | 0,68 | 0,80 | 0,70 |
| Q1, 30 CubiCasa de validação | 29 / 30 | 0,25 | 0,50 | 0,25 | 0,42 | 0,66 | 0,42 |

O modelo base ignora o formato e inventa coordenadas redondas. O Q1 acerta
a estrutura (número de paredes e aberturas quase igual ao gabarito, F1 de
0,7 a 0,8 a 5%), mas erra a posição fina: a 1,5% (cerca de 15 px) o F1 cai
para 0,3. É o limite conhecido da coordenada escrita token a token
(`docs/escolha-do-modelo.md`). A perda de validação quase não caiu na
segunda época enquanto a de treino caiu: as 26 brasileiras já foram vistas
40 vezes cada.

### Q1: ajuste pela máscara (descartado) e métrica por comprimento (2026-10-03)

Hipótese: o Qwen acerta a estrutura e erra a posição; mover cada parede, só
numa faixa estreita, para a parede da máscara do E5 recuperaria a precisão a
1,5%. Medido nas 7 brasileiras de validação:

| F1 de parede (pontas) | @1,5% | @5% |
|---|---|---|
| Q1 | 0,32 | 0,68 |
| + deslocamento perpendicular | 0,32 | 0,68 |
| + pontas na interseção com a parede perpendicular | 0,30 | 0,71 |
| + pontas no fim da parede na máscara | 0,24 | 0,74 |
| + as duas (faixa de 1,5% a 3,5%) | 0,31 a 0,33 | 0,70 a 0,74 |

Nada recupera a precisão. O erro perpendicular do Qwen já é pequeno
(mediana 0,1% do lado maior); o que falha a 1,5% são as pontas, e no
gabarito as paredes são trechos interrompidos em cada porta e janela, não de
canto a canto. Estender até a interseção cria trechos que não existem, e a
máscara do E5 é fraca nas paredes de contorno duplo oco de SJC.

A métrica de pontas exagera o erro: um trecho com uma ponta fora conta como
erro inteiro. Medida pelo comprimento (fração da linha de parede prevista a
menos da tolerância de uma parede do gabarito, e vice-versa), o Q1 fica em:

| tolerância | 0,5% | 1,0% | 1,5% | 5% |
|---|---|---|---|---|
| F1 por comprimento | 0,64 | 0,71 | 0,77 | 0,95 |

`eval_vlm` passa a reportar `wall_len` junto com o F1 de pontas.

## Correções com IA no portfólio de SJC: Claude x GPT (2026-10-03)

As IAs corrigiram plantas do portfólio pelo pacote do editor e devolveram
também os scripts e um relatório (`data/ferr`, fora do git).

- Claude (24_terreo) escreveu um detector completo (`pipeline_automatico`:
  espessura, paredes como pares de faces paralelas, junção, portas e janelas
  pela cor, corte nos vãos). Em plantas que não viu (25_terreo,
  25_superior) dá ~17 trechos em vez de 140-160, eixos internos a 1-2 px do
  gabarito, F1 de pontas a 1,5% de 0,47-0,57 (E5: 0,08-0,11). Erra de forma
  sistemática: toma a projeção da cobertura por face de parede (parede falsa
  no alto, parede esquerda 26 px para fora), mede a janela com os
  prolongamentos azul-claros (~20 px a mais de cada lado: F1 de janela 0-0,25)
  e tem as cores de porta/janela fixas. Não incorporado como rascunho.
- GPT (25_superior) corrigiu no olho e escreveu ferramentas de inspeção:
  recorte ampliado com grade nas coordenadas originais e validação
  (parede sobre vão, abertura solta, segmento torto). Incorporadas ao pacote
  (`recortes/`) e à importação (avisos).
- Divergência de convenção: o Claude não dividia a parede no T, o GPT
  dividia. As 33 correções anteriores dividem em todos os 126 encontros em T
  (nenhum atravessado), então o LEIA-ME passou a exigir a divisão em todo
  encontro, e a janela medida só pelo caixilho.
- Os gabaritos 24_terreo e 25_superior são as próprias correções das IAs,
  então comparar uma IA com eles mede concordância, não acerto.

### Lotes da IA: GPT (5 plantas) e Claude (5 plantas), detector v2 (2026-10-03)

- Lote do GPT (24_superior, 23, 22, 21b, 21a): correções boas e coerentes com
  as convenções (0 avisos, todos os T divididos), mas o relatório final veio
  genérico; o LEIA-ME do lote passou a trazer um modelo com tabelas.
- Lote do Claude (18c, 19a, 19b, 19c, 20): relatório completo no modelo. Ele
  rodou o próprio detector (`ferramentas.py` v2: pares de faces com
  tolerância de 15%, "miolo limpo" entre as faces para não casar face com
  cota, duas espessuras por planta, cores saturadas fora da máscara, soleira
  reconhecida, divisão nos encontros) e editou à mão só o que ele errou: 16
  edições em 5 plantas. Essas correções nasceram do detector, então
  carregam o viés dele.
- Detector v2 contra 7 plantas corrigidas por outra fonte (o lote do GPT,
  25_terreo e 25_superior), F1 a 1,5% do lado maior:

| rascunho | paredes por comprimento | paredes (pontas) | portas | janelas |
|---|---|---|---|---|
| E5 + vetorização (atual) | 0,94 | 0,10 | 0,55 | 0,80 |
| detector v1 do Claude | 0,92 | 0,53 | 0,70 | 0,64 |
| detector v2 do Claude | 0,91 | 0,72 | 0,75 | 0,67 |

  O v2 entrega 14-27 trechos em vez de 130-200 e acerta as pontas, mas perde
  as paredes internas quando elas são desenhadas mais finas que as externas
  (21a e 21b: ~12 px contra 20 px, fora da faixa de 0,7-1,5·t), e com elas as
  portas internas. As janelas da 25_terreo e 25_superior continuam medidas
  com os prolongamentos.

  Ajuste ao trazer o v2 para `wallextractor.faces`: espessuras secundárias a
  partir de 20% do peso da principal e de 0,6·t (antes 30% e 0,7·t). Na 21b
  as internas (14 px contra 20) passam a entrar: 28 paredes em vez de 18,
  6 de 6 portas, comprimento 0,84 -> 0,95; nas outras plantas nada piora. Na
  21a as internas pesam só 14% e continuam de fora; baixar o corte para 12%
  as recupera, mas cria paredes falsas no lote do Claude (precisão 1,00 ->
  0,96) e piora as portas, então ficou em 20%.
- Lote do GPT com o modelo de relatório (17a, 17b, 18a, 18b): o modelo
  funcionou, todas as tabelas vieram com números (631 trechos de parede do
  rascunho viraram 87; 218 paredes inexistentes; 4 tipos trocados, quase
  sempre a J1 entre banheiro e A.S.). As ferramentas dele seguem sendo de
  inspeção, sem detector. As sugestões para o rascunho coincidem com as do
  Claude (pares de faces, juntar antes de dividir, máscara para cota,
  piso e louça, abertura pelos batentes e caixilho).
- Detector de faces nessas 4 plantas de edícula (gabarito do GPT), F1 a 1,5%:
  pontas 0,32-0,45 (E5: 0,05-0,10), mas perde paredes (10-16 trechos contra
  20-25; comprimento 0,79-0,92 contra 0,84-0,95) e as aberturas variam
  (portas 0,25-1,00, janelas 0-0,80). Confirma que ele é um acelerador para
  o estilo em que foi escrito, não um rascunho geral.

## E6 e Q2: 50 plantas corrigidas (2026-10-05)

50 correções (4 escritório, 29 vetoriais de SJC, 17 do portfólio raster),
divididas em 40 de treino e 10 de validação (`annotations`, semente 0). A
divisão mudou em relação às 33 plantas: das 10 de validação, 4 estavam no
treino do E5/Q1, 1 na validação deles e 5 são novas (portfólio).

**E6** (receita do E5, 80 épocas, 2 h 03 min; melhor época 72). IoU nas 10
de validação:

| | parede | porta | janela | mIoU |
|---|---|---|---|---|
| E5 (viu 4 das 10 no treino) | 0,739 | 0,612 | 0,694 | 0,682 |
| E6 | 0,837 | 0,746 | 0,824 | 0,802 |

O E6 ganha mesmo com a vantagem do E5. Publicado no editor (`PUT /api/model`,
sha256 `0982eafb…`).

**Q2** (Q1 com as 50 plantas, repetição 10 em vez de 20, 2 épocas, 3 h 13 min;
perda de validação 0,593 -> 0,578). Na validação inteira o Q1 parecia melhor
(paredes a 1,5% 0,52 contra 0,24), mas o Q1 tirou 1,00 em tudo nas 4 plantas
que viu no treino: decorou (cada planta vista 40 vezes). Separando, F1 a 1,5%:

| | plantas | paredes (pontas) | paredes (comprimento) | portas | janelas |
|---|---|---|---|---|---|
| Q1 | 4 vistas no treino do Q1 | 1,00 | 1,00 | 1,00 | 1,00 |
| Q2 | as mesmas 4 (nunca vistas) | 0,33 | 0,73 | 0,55 | 0,20 |
| Q1 | 6 nunca vistas por nenhum | 0,11 | 0,69 | 0,27 | 0,29 |
| Q2 | as mesmas 6 | 0,18 | 0,73 | 0,18 | 0,41 |

Nas plantas novas o Q2 melhora paredes e janelas e piora portas; amostra de
6, quase todas do portfólio, estilo que os dois acham difícil. Q2 no
CubiCasa (30): paredes 0,30 / 0,85, portas 0,56, janelas 0,22.

Lições: comparar modelos só em plantas que nenhum viu (fixar a validação ao
crescer o conjunto, ou guardar um conjunto de teste à parte); repetição alta
das plantas brasileiras faz o Qwen decorar.

- Detector de faces nas 18 plantas de Bauru (escaneadas): falha em 10 (não
  acha nenhum par de faces: paredes preenchidas, como nas Redentor, ou
  traço colorido de cópia heliográfica, que a máscara descarta de
  propósito) e nas outras acha 0 a 26 trechos. Não foi adaptado (heurística
  por estilo); passou a recusar com explicação em vez de erro de Python.
- Paredes pintadas da cor do caixilho (Bauru: Vitória Régia e Nova Bauru,
  paredes preenchidas de azul): `medir_aberturas` via "janela" ao longo da
  parede inteira e `cortar_nos_vaos` trocava as paredes por janelas.
  Correção: uma janela só vale se as duas faces da parede forem
  interrompidas em pelo menos 40% do trecho (`faces_continuas`). Nas portas
  não se aplica: a soleira é desenhada como duas linhas no vão e, aplicada a
  elas, a regra derrubava as portas do portfólio de 0,76 para 0,33. Portfólio
  (17 plantas, F1 a 1,5%): portas 0,759 -> 0,759, janelas 0,706 -> 0,699.
  Vitória Régia e Nova Bauru: 0 janelas falsas (antes 8 e 13); as paredes
  externas pintadas de azul forte continuam de fora.

## Validação fixa, rascunho raster x Qwen na mesma métrica, Q3 e outros modelos (2026-10-07)

79 plantas corrigidas. A validação passa a ser fixa (`splits/val_editor.txt`,
`annotations.prepare --val-ids`): as 10 de validação do E6/Q2 ("antigas") e 7
novas sorteadas entre Bauru e Caraguatatuba ("novas", estilos escaneado e
colorido que nenhum modelo viu). As outras 62 são treino, e plantas novas
só entram no treino.

`scripts/eval_pipeline.py` mede o rascunho raster (SegFormer + `vectorize`,
como o editor faz) com a métrica do Qwen (`wallextractor.vlm_metrics`: mesmas
imagens de 1024 px, mesmo gabarito, F1 a 1,5% do lado maior). `--cleanup`
aplica geometria genérica do detector de faces: endireita trechos a menos de
6° do eixo, junta colineares, liga cantos, corta nos vãos e divide nos
encontros.

| grupo | rascunho | paredes (pontas) | paredes (comprimento) | portas | janelas |
|---|---|---|---|---|---|
| 7 novas | Q2 (Qwen 4B) | 0,22 | 0,81 | 0,17 | 0,31 |
| 7 novas | E5 + vetorização | 0,15 | 0,92 | 0,68 | 0,81 |
| 7 novas | E5 + vetorização + limpeza | 0,56 | 0,91 | 0,68 | 0,81 |
| 10 antigas | Q2 | 0,23 | 0,73 | 0,35 | 0,30 |
| 10 antigas | E5 + vetorização | 0,13 | 0,91 | 0,47 | 0,65 |
| 10 antigas | E5 + vetorização + limpeza | 0,47 | 0,90 | 0,47 | 0,65 |

O E5 viu 4 das 10 antigas no treino; nas 7 novas ninguém viu nada. O
rascunho raster já ganhava do Qwen em comprimento, portas e janelas e só
perdia nas pontas, por entregar a parede picotada; com a limpeza ganha em
tudo.

Em andamento no Pichau (`run_q3.sh`, `run_models.sh`): Q3 (Qwen3-VL-4B, 62
plantas, repetição 5 com giros e espelhos, `vlm_data --augment`), E7 (receita
do E6 na divisão fixa) e, com a receita do Q3, Qwen3-VL-2B e InternVL3.5-4B
(`train_vlm` agora acha a resposta pelo modelo de chat de qualquer modelo;
`--max-patches 6` no InternVL). O Qwen3-VL-8B em 4 bits precisa do
bitsandbytes, que a imagem `we-vlm` não tem.
