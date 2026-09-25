# Datasets para treinar o Qwen-VL na extração de paredes

Levantamento feito em 2026-09-24. Objetivo: escolher os datasets para fazer
fine-tuning de um Qwen-VL (2.5-VL ou 3-VL) que recebe a imagem de uma planta
baixa (renderizada a partir do PDF) e devolve as paredes como JSON estruturado
(segmentos com coordenadas, espessura, aberturas).

## Resumo executivo

| # | Dataset | Tamanho | Formato das paredes | Licença | Uso sugerido |
|---|---------|---------|---------------------|---------|--------------|
| 1 | **CubiCasa5K** | 5.000 plantas (raster + SVG) | Polígonos de parede no SVG, >80 classes | CC BY-NC 4.0 | Base principal de SFT (mesmo receita do FloorplanVLM). Só pesquisa/uso não comercial. |
| 2 | **FloorPlanCAD** | ~15.000 desenhos CAD (DWG→SVG) | Linhas vetoriais rotuladas, parede é classe "stuff" | Anotações CC BY-NC 4.0 | O mais parecido com PDF vetorial de projeto de engenharia. Ideal para o pipeline "PDF vetorial → SVG → JSON". |
| 3 | **ResPlan** | 17.000 plantas residenciais (só vetor) | Polígonos de parede em metros, espessura 10–40 cm | **CC BY 4.0** (dados) / MIT (código) | Único grande dataset com licença comercial. Precisa renderizar as imagens (o repositório tem utilitário de plot). |
| 4 | **MSD (Modified Swiss Dwellings)** | 5.300 plantas / 18.900 apartamentos | Imagens + geometria + grafos | CC BY (Kaggle/Zenodo/4TU) | Prédios multi-unidade, complementa o residencial unifamiliar. |
| 5 | **R2V + R3D (Rent3D)** | 815 + ~200 imagens | Máscaras pixel a pixel (parede/porta/janela/cômodo) | Acadêmica | Benchmark clássico. Pequeno; serve para avaliação, não para treinar o VLM. |
| 6 | **RPLAN** | 80.788 plantas 256×256 | Raster com paredes/cômodos rotulados | Acadêmica, mediante pedido | Muito grande, mas simples e de baixa resolução. Bom para pré-treino sintético. |
| 7 | **Structured3D** | 3.500 casas, 21.835 cômodos | Junções e planos de parede (3D), planta derivável | Termos próprios, formulário | Render limpo e geometria perfeita. Usado na fase 2 do FloorplanVLM. |
| 8 | **Floor Plan CIS** (Zenodo 17871079) | 500 plantas | Máscaras de parede | Ver Zenodo | Teste de domain shift (paredes hachuradas, estilo diferente do CubiCasa). |
| 9 | **PFP (Precision Floorplan)** | 1.554 plantas vetoriais | SVG | Site comercial, verificar | Vetorização de desenho técnico. |
| 10 | **CVC-FP / SESYD / FPLAN-POLY** | Centenas | Vetor / raster | Acadêmica (IAPR TC10) | Históricos, pequenos. Só para comparação. |
| 11 | **Roboflow Universe** (wall-floor, floor-plan-segmentation) | 148 a 8.880 imagens por projeto | Instance segmentation | CC BY 4.0 / MIT (varia) | Qualidade irregular. Útil como complemento barato. |

## Recomendação

Para chegar rápido a um protótipo:

1. **Fase 1 (SFT, replicar o que já funciona):** CubiCasa5K convertido de SVG
   para JSON. Há três projetos públicos que fizeram exatamente isso com
   Qwen2.5-VL-3B + LoRA e publicaram código e pesos (ver seção "Trabalhos
   prontos"). Dá para começar do adapter deles e validar o pipeline em dias.
2. **Fase 2 (dados limpos e mais volume):** ResPlan renderizado em vários
   estilos (traço fino, parede preenchida, hachura) + MSD. Isso também resolve
   a licença: ResPlan é CC BY 4.0, então o modelo final pode ser comercial se
   for treinado só nele e no MSD.
3. **Fase 3 (domínio de engenharia):** FloorPlanCAD, porque a origem é DWG,
   igual ao que gera os PDFs de projeto. Além de treinar o VLM, ele permite um
   caminho paralelo sem VLM: se o PDF for vetorial, extrair os paths com
   PyMuPDF e classificar linhas como parede.
4. **Avaliação:** R2V, R3D e Floor Plan CIS como conjuntos de teste
   fora da distribuição. O artigo de He Zhang (ago/2026) publicou anotações
   corrigidas do CubiCasa5K e o benchmark ResPlan-FP com 16.998 plantas,
   com métrica de custo de edição. Vale usar como referência de avaliação.
5. **Dados brasileiros:** não existe dataset público de plantas baixas
   brasileiras anotadas. Será preciso montar um conjunto próprio pequeno
   (100 a 300 plantas) para fine-tuning final e teste, anotando com uma
   ferramenta simples no próprio sistema web (isso é um bom módulo do produto).

## Ponto de atenção: licenças

CubiCasa5K e FloorPlanCAD são CC BY-NC. Um modelo treinado neles só pode ser
usado em pesquisa ou uso interno não comercial. Se o sistema web vai ser um
produto, o caminho seguro é: prototipar com CubiCasa5K, e treinar a versão
de produção com ResPlan + MSD + dados próprios.

## Trabalhos prontos que já treinaram Qwen-VL nisso

- **FloorplanVLM** (arXiv 2602.06507, fev/2026). Qwen2.5-VL reformulado como
  geração de sequência JSON. Três estágios: SFT, SFT em dados de alta
  qualidade, e GRPO com recompensa geométrica. Construíram Floorplan-2M e
  Floorplan-HQ-300K com um data engine próprio. GRPO subiu a validade de
  90,2% para 96,1%. Não confirmei se os dados foram liberados (arXiv está
  bloqueado no ambiente atual).
- **miladmirzazadeh/FloorPlanVLM** (GitHub). Reimplementação aberta da receita
  acima com Qwen2.5-VL-3B + LoRA + TRL/GRPO. Dados: CubiCasa5K (~5k), MSD
  (~4k), Structured3D (~3,5k) e renders sintéticos (até 10k). Suporta parede
  curva. Treinado em 1 A100 80 GB. Métricas: wall-IoU, room-IoU/F1, opening
  F1, MAE de contagem de paredes.
- **mudasir13cs/qwen25-vl-3b-floorplan-sft** e **-grpo** (Hugging Face).
  Pesos LoRA sobre Qwen2.5-VL-3B-Instruct treinados no CubiCasa5K convertido
  para JSON. Código e labels em **manitocross/floorplan-vlm-training**.
  Schema de saída: paredes com id, start, end, thickness, curvature e
  openings (portas/janelas).
- **He Zhang, "When Should a Network Emit Geometry, and When Should It Detect
  It?"** (arXiv 2608.25608, ago/2026). Compara VLM autoregressivo contra
  detecção por heatmap. Em scans reais (CubiCasa5K) a detecção ganha em
  todas as métricas de parede; em renders vetoriais limpos a sequência ganha
  por 5 a 8 pontos. Fundir os dois sobe o wall F1 em 7 pontos. Conclusão
  prática para nós: PDF vetorial renderizado limpo favorece o VLM; planta
  escaneada favorece um modelo de segmentação (U-Net/SegFormer) ou a fusão.

## Detalhes por dataset

### 1. CubiCasa5K
- 5.000 plantas finlandesas revisadas manualmente, de um lote de 15.000.
- Anotação por imagem em SVG com polígonos; classes de estrutura incluem
  Wall, Railing, portas e janelas; 12 classes de cômodo.
- Download: Zenodo record 2613548. O LMDB pré-processado do repositório
  oficial ocupa ~105 GB, mas o zip bruto é bem menor.
- Repositório: https://github.com/CubiCasa/CubiCasa5k
- Licença: CC BY-NC 4.0.

### 2. FloorPlanCAD
- >15.000 plantas reais (residencial, comercial, hospital, escola) convertidas
  de DWG para SVG, com anotação por linha em ~30 a 35 categorias.
- Parede, cortina de vidro, guarda-corpo etc. são classes "stuff"; portas,
  janelas e mobiliário são "things".
- Download: Google Drive (script no repositório VecFormer usa
  `gdown 1wsOQxIXjsqYzMlUpPNRjyQiMnwgVbtJG`); SymPoint traz
  `download_data.py` e `parse_svg.py` para gerar JSON.
- Também disponível como Voxel51/FloorPlanCAD no Hugging Face (split de teste).
- Site: https://floorplancad.github.io/
- Licença das anotações: CC BY-NC 4.0.

### 3. ResPlan
- 17.000 plantas derivadas de anúncios imobiliários; só geometria, sem imagem.
- Polígonos de paredes, portas, janelas, cômodos e varandas em metros;
  17 rótulos semânticos; grafos de conectividade com 4 tipos de aresta.
- Splits canônicos: 13.053 treino / 1.632 val / 1.632 teste (+683 aumentados).
- Repositório: https://github.com/m-agour/ResPlan (ResPlan.zip com pickle,
  split.json, resplan_utils.py). Também no Kaggle.
- Licença: CC BY 4.0 (dados), MIT (código).

### 4. MSD (Modified Swiss Dwellings)
- 5.300 plantas de conjuntos habitacionais suíços, 18.900 apartamentos.
- Entrega imagens, geometria e grafos (networkx / torch_geometric).
- Download: Kaggle "Modified Swiss Dwellings", Zenodo 17294451 (versão JSON),
  4TU.ResearchData.
- Repositório: https://github.com/caspervanengelenburg/msd
- Licença: Creative Commons Attribution.

### 5. R2V e R3D
- R2V: 815 imagens (715/100) do Raster-to-Vector (Liu et al., ICCV 2017).
  Repositório: https://github.com/art-programmer/FloorplanTransformation
- R3D: Rent3D com anotação pixel a pixel + 18 imagens.
  Fonte: http://www.cs.toronto.edu/~fidler/projects/rent3D.html
- Anotações usadas pelo DeepFloorplan: https://github.com/zlzeng/DeepFloorplan

### 6. RPLAN
- 80.788 plantas asiáticas, 256×256, 13 tipos de cômodo, tudo alinhado aos
  eixos e na mesma escala.
- Página do projeto: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
- Acesso mediante pedido aos autores.

### 7. Structured3D
- 3.500 projetos profissionais com junções de parede, planos e semântica.
- Download mediante formulário de termos de uso: https://structured3d-dataset.org/
- Repositório: https://github.com/bertjiazheng/Structured3D

### 8. Floor Plan CIS
- 500 plantas de anúncios da Rússia/CEI, publicado junto com o MitUNet
  (arXiv 2512.02413). Paredes estruturais em preto sólido, divisórias com
  hachura diagonal. Bom teste de domain shift.
- DOI: 10.5281/zenodo.17871079

### 9. PFP (Precision Floorplan)
- 1.554 plantas vetoriais reais (1.364 treino / 150 val), usadas em "Deep
  Vectorization of Technical Drawings" (ECCV 2020).
- Script: `dataset/precision_floorplan_download.py` em
  https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings
- Origem comercial; verificar termos antes de usar em produto.

### 10. CVC-FP, SESYD, FPLAN-POLY, Versailles-FP
- Listados em https://iapr-tc10.univ-lr.fr/?page_id=71 (IAPR TC10).
- Versailles-FP: plantas históricas escaneadas, detecção de parede.

### 11. Roboflow Universe
- "wall-floor" (part1: 8.880 imagens; x: 4.852; part2val: 5.062),
  "Floor-plan-segmentation" (IIITBangalore, 148 imagens, wall/door/window),
  "floor-wall-segmentation" (m10, 985 imagens).
- Busca: https://universe.roboflow.com/search?q=class%3Afloorplan

## Plantas brasileiras públicas em PDF vetorial (2026-09-25)

Não há dataset anotado, mas há PDFs vetoriais públicos que servem de teste
para o ramo vetorial e, corrigidos no editor, de gabarito:

| Fonte | O que tem | Estado |
|---|---|---|
| Prefeitura de São José dos Campos, Planta Popular (`sjc.sp.gov.br/servicos/habitacao-e-regularizacao-fundiaria/habitacao/planta-popular/`) | 36 modelos de casa de 43 a 100 m²; 29 vetoriais (m01 a m15-b), 13 do "portfólio" escaneados | 7 corrigidos no editor (V2 em `experimentos.md`). Sem etiquetas P/J nem quadro; janela = 4 linhas vermelhas no vão com anotação "L×A / peitoril" |
| Cadernos CAIXA, casa popular 42 m² (`caixa.gov.br/Downloads/banco-projetos-projetos-HIS/`) | 42 páginas A4, pranchas vetoriais pequenas | Extrai paredes, sem escala (sem rótulo nem cotas legíveis); bloqueia download por script |
| FNDE Proinfância (creches Tipo 1, 2, B, MEI) | 41 pranchas por tipo em PDF, DWG e IFC, de BIM, com quadro de esquadrias | Paredes hachuradas, não em pena grossa: o pareamento não acha paredes. Precisa de outro detector |
| Bauru (143 PDFs de núcleos habitacionais), AGEHAB-MS, COHAB-MG | Casas populares | Escaneados (só ramo raster) |

## Próximos passos propostos

1. Baixar CubiCasa5K e ResPlan; escrever o conversor SVG/pickle → JSON com o
   mesmo schema usado pelo manitocross/floorplan-vlm-training, para reaproveitar
   os adapters publicados como ponto de partida.
2. Definir o schema JSON de saída do WallExtractor (paredes como segmentos com
   espessura + aberturas + escala em metros) e o renderizador de ResPlan em
   múltiplos estilos.
3. Montar um conjunto de teste com 20 a 50 PDFs de projetos brasileiros reais,
   renderizados com PyMuPDF, para medir o modelo desde o primeiro dia.
