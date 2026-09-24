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

Próximos passos: (a) portas e janelas no ramo vetorial a partir da pena
intermediária e dos vãos entre paredes; (b) fechar os trechos curtos que o
pareamento perde nos cantos; (c) conjunto de teste: revisar os quatro JSONs
gerados e corrigir o que estiver errado, guardando fora do repositório.
