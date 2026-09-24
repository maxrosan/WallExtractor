# O Qwen é o melhor modelo aberto para extrair paredes de plantas baixas?

Estudo feito em 2026-09-24. Complementa `docs/datasets.md`.

Pergunta: dado que a tarefa é "imagem da planta (renderizada do PDF) → JSON
com segmentos de parede, espessura e aberturas", o Qwen-VL é a melhor escolha
entre os modelos abertos? E, antes disso, um VLM (modelo de visão e
linguagem) é a melhor abordagem?

## Resposta curta

1. **Entre os VLMs abertos, sim: a família Qwen é a escolha mais defensável
   hoje.** Todos os três trabalhos públicos que já fizeram exatamente esta
   tarefa (FloorplanVLM da Beike, a reimplementação miladmirzazadeh e os
   adapters mudasir13cs/manitocross) usaram Qwen2.5-VL. O ecossistema de
   fine-tuning (Unsloth, LLaMA-Factory, ms-swift, TRL/GRPO) é o mais maduro,
   a licença é Apache 2.0 em todos os tamanhos, e o grounding 2D (RefCOCO,
   ODinW, CountBench) é estado da arte entre abertos. O único concorrente
   direto que vale testar é o **GLM-4.6V-Flash (9B, MIT)**, que afirma bater
   o Qwen3-VL-8B em grounding.

2. **Mas um VLM sozinho provavelmente não é a melhor arquitetura para a
   geometria.** O artigo mais rigoroso sobre o assunto (He Zhang, ago/2026,
   `fpvec-lab`) mostra que, com dados e treino iguais, uma cabeça de
   **detecção por heatmap ganha da decodificação autoregressiva em plantas
   reais**, e a diferença cresce quando a tolerância aperta: +2,7 pontos de
   wall F1 a 5% e +5,1 pontos a 1,5%. A fusão dos dois ganha de ambos por
   ~7 pontos. Coordenadas emitidas token a token são probabilísticas; a
   geometria é exata. O VLM fine-tunado que chegou a 92,5% de IoU de parede
   (FloorplanVLM) precisou de 2 milhões de plantas proprietárias e 32 H200,
   fora do nosso alcance.

3. **Recomendação: arquitetura híbrida, com o Qwen como componente e não como
   sistema inteiro.**
   - PDF vetorial (caso mais comum em projeto de engenharia): extrair os
     paths com PyMuPDF e classificar linhas como parede com um modelo de
     primitivas vetoriais (linha do VecFormer/SymPoint). Coordenadas exatas,
     sem VLM.
   - PDF raster/escaneado: modelo de detecção (junções + linhas centrais,
     código MIT do `fpvec-lab`, ou MitUNet para máscara de parede) para a
     geometria, e Qwen-VL para semântica, topologia, leitura de cotas/escala
     e para montar o JSON final. Começar pelo protótipo VLM puro só porque é
     o caminho mais rápido de ter algo funcionando, medindo desde o dia 1 a
     tolerância apertada (F1@0.015).

## Por que o VLM puro tem limite: evidência

| Fonte | O que mediu | Resultado |
|---|---|---|
| He Zhang, *When Should a Network Emit Geometry, and When Should It Detect It?* (arXiv 2608.25608, ago/2026) | Mesma rede, mesmos dados, duas saídas: sequência autoregressiva vs. heatmap. CubiCasa5K com anotações corrigidas. | Detecção: wall F1 0,818 @0,05 / 0,787 @0,015. Sequência: 0,790 / 0,736. Fusão: 0,853 / 0,811. Em renders vetoriais limpos a sequência ganha por 5 a 8 pontos. |
| Mesmo artigo, baseline VLM de fronteira zero-shot (Gemini 3.1 Pro) em ResPlan-FP | Topologia correta, coordenadas grosseiras | Wall F1 0,811 @0,05 mas **0,472 @0,015** |
| FloorplanVLM (Beike, arXiv 2602.06507, fev/2026) | Qwen2.5-VL-3B, SFT + SFT-HQ + GRPO, JSON direto | IoU parede externa 92,5%, validade do JSON 96,1%. Custo: Floorplan-2M proprietário, 32×H200, sem comparação com baselines nem pesos liberados. |
| Raster2Seq (Cornell, SIGGRAPH 2026, MIT, pesos no HF) | Decoder autoregressivo especializado, 256/512 px | Room F1 88,7 no CubiCasa5K, SOTA. Foca em cômodos, não em paredes com espessura. |
| VecFormer (NeurIPS 2025) em FloorPlanCAD | Entrada vetorial (linhas do DWG/SVG) | PQ 91,1, muito acima de qualquer método raster. Para PDF vetorial, é o caminho. |

Leitura prática: o VLM entende bem *o que* está na planta e *como se
conecta*; erra *onde exatamente*. Numa planta de 1024 px, 1,5% são 15 px,
ou seja, a própria espessura de uma parede. Para o produto (área, quantitativo,
BIM) esse erro importa.

## Comparação dos VLMs abertos candidatos

Estado em setembro de 2026.

| Modelo | Tamanhos abertos | Licença | Saída de coordenadas | Pontos a favor | Pontos contra |
|---|---|---|---|---|---|
| **Qwen2.5-VL** (fev/2025) | 3B, 7B, 32B, 72B | Apache 2.0 | Pixel absoluto, patch 28 px | Três projetos prontos de planta→JSON; adapters LoRA públicos; receita validada | Geração anterior; superado pelo Qwen3-VL em grounding |
| **Qwen3-VL** (set–nov/2025) | 2B, 4B, 8B, 32B, 30B-A3B, 235B-A22B | Apache 2.0 | Relativo 0–1000 (recomendado), patch 32 px | SOTA aberto em RefCOCO/ODinW/CountBench; 256K de contexto; tooling completo | Nada específico contra; é a atualização natural do 2.5 |
| **Qwen3.5** (fev–mar/2026) | 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B | Apache 2.0 | Idem | Multimodal nativo (fusão precoce); em EPIC-Bench o 9B empata com Qwen3-VL-8B (35,45 vs 35,55); Unsloth suporta todos | Ganho de grounding sobre Qwen3-VL ainda não comprovado em benchmark independente |
| **Qwen3.8-27B** (ago/2026) | só 27B (e um 2.4T-A95B com licença própria) | Apache 2.0 | Idem | Melhor modelo denso aberto ~30B; OmniDocBench 91,1 | Só um tamanho; 27B exige QLoRA ou 2 GPUs; sem número de grounding publicado |
| **GLM-4.6V** (Z.ai, dez/2025) | 9B Flash, 106B-A12B | MIT | Bbox nativo | Afirma bater Qwen3-VL-8B "em quase tudo"; Ref-L4 87,7 (Flash); tool-calling nativo | Ecossistema de fine-tuning menor (ms-swift sim, Unsloth parcial); menos trabalho prévio em plantas |
| **InternVL3.5** (ago/2025) | 1B a 241B-A28B | Apache 2.0 | Bbox | Bom em documentos; muitos tamanhos | Tiling dinâmico em tiles de 448 px quebra a visão global da planta; grounding abaixo do Qwen3-VL |
| **Gemma 4** (Google, abr/2026) | E2B, E4B, 12B, 26B-A3.8B, 31B | Apache 2.0 (primeira vez na linha Gemma) | Bbox `[y1,x1,y2,x2]` | Licença finalmente limpa; bom em documento | Orçamento máximo de 1.120 tokens por imagem, pouco para planta grande; não é feito para coordenada precisa |
| **Molmo2** (Ai2, jan/2026, CVPR oral) | 4B, 8B, 7B-O | CC BY 4.0 (pesos) | **Pontos**, não bbox nem segmentos | Melhor aberto em pointing e contagem (PixMo-Count 88,5 vs 65,0 do Qwen3-VL-8B) | Não emite segmentos; base Qwen3 de qualquer forma |
| **Florence-2** (Microsoft) | 0.23B, 0.77B | MIT | 1.000 bins por eixo; suporta polígonos | Barato, treina em minutos, fine-tuning em detecção bem documentado (Roboflow) | Sem raciocínio; resolução 768 px; 1.000 bins = 1 px em 1024, ok |
| **PaliGemma 2** | 3B, 10B, 28B | Gemma (restritiva) | 1.024 bins | Receita de JSON estruturado pronta | Licença antiga da Gemma, sem motivo para preferir ao Gemma 4 |
| VLMs de OCR (PaddleOCR-VL, DeepSeek-OCR-2, dots.ocr, olmOCR 2) | 0.9B a 3B | Apache/MIT | Layout em bbox | Excelentes em documento; DeepSeek-OCR-2 foi fine-tunado para reconhecer estrutura molecular (grafo a partir de desenho), prova de que o conceito funciona | Não foram feitos para geometria; nicho |

Modelos fechados (GPT, Gemini, Claude) ficam fora por definição da pergunta,
mas registro que o SALI-FP (arXiv 2609.25615, set/2026), que usa gpt-image-2
e gpt-5.5 num pipeline de produção com 11.534 plantas, obteve apenas 0,33 de
IoU de parede visível no CubiCasa5K. Modelo de fronteira sem fine-tuning não
resolve a geometria.

## Qual Qwen usar

| Fase | Modelo | Motivo |
|---|---|---|
| Protótipo (semana 1–2) | **Qwen2.5-VL-3B** + adapter `mudasir13cs/qwen25-vl-3b-floorplan-sft` | Já existe; valida pipeline, schema e métricas sem treinar nada |
| Treino próprio | **Qwen3-VL-4B ou 8B** (ou Qwen3.5-4B/9B, empate técnico) | Melhor grounding que o 2.5; cabe em 1 GPU de 80 GB com LoRA; coordenadas 0–1000 |
| Se precisar de mais capacidade | Qwen3-VL-32B ou Qwen3.8-27B com QLoRA | Só se o 8B saturar; custo ×4 |

Detalhe de fine-tuning que importa: congelar o ViT e o merger, LoRA só no LLM,
como fazem os projetos públicos. Imagem com lado maior em 1024 px (mesma
escolha do FloorplanVLM). Reward do GRPO baseado em IoU/F1 geométrico, não
só em validade do JSON.

## Alternativas fora do VLM que devem entrar no sistema

| Componente | Repositório | Licença | Para quê |
|---|---|---|---|
| Detecção de junções/linhas centrais + fusão | github.com/Cyprinus12138/fpvec-lab | MIT (código), ResPlan-FP CC BY 4.0 | Geometria de parede em raster; já traz avaliador com F1 por tolerância e o benchmark ResPlan-FP (16.998 plantas) |
| Raster2Seq | github.com/Cornell-VAILab/Raster2Seq (pesos em HF `haopt/Raster2Seq`) | MIT | Polígonos de cômodo; útil para fechar topologia |
| MitUNet (Mix-Transformer + U-Net) | arXiv 2512.02413 | ver repo | Máscara de parede em plantas com hachura; publicou o Floor Plan CIS |
| VecFormer / SymPoint | github.com/WesKwong/VecFormer | ver repo | Classificação de primitivas vetoriais (PDF vetorial → parede/porta/janela) |
| Pipeline determinístico + agente | arXiv 2608.17237 (Univ. Alberta, ago/2026) | artigo | Plantas estruturais em PDF vetorial: recall/precisão de parede 1,000/1,000 com extração de primitivas + gramática de desenho + VLM só para correções |

O último item é o mais parecido com o nosso caso de uso (PDF de projeto de
engenharia) e não treinou nenhum modelo: extraiu as primitivas do PDF,
estimou a escala pelas cotas e usou o VLM apenas como revisor com regras de
admissão. É um forte indício de que, para PDF vetorial, o VLM é coadjuvante.

## Como decidir de forma objetiva: bake-off de uma semana

Rodar tudo no mesmo subconjunto (500 plantas de treino / 100 de teste do
CubiCasa5K com anotações corrigidas `skv4`, mais 100 do ResPlan-FP renderizado)
e medir com o avaliador do `fpvec-lab`: wall F1 @0,05 e @0,015, opening F1,
validade do JSON, tempo por imagem.

| Candidato | Custo estimado em 1 A100 80 GB |
|---|---|
| Qwen2.5-VL-3B com adapter público (sem treino) | horas |
| Qwen3-VL-4B LoRA, 2 épocas | ~4 h |
| Qwen3-VL-8B LoRA, 2 épocas | ~8 h |
| GLM-4.6V-Flash 9B LoRA, 2 épocas | ~8 h |
| fpvec-lab detecção (ResNet + deformable attention, 256 px) | ~3 h |
| Florence-2-large com polígonos | ~1 h |

Critério de decisão: se o melhor VLM ficar mais de 3 pontos abaixo da
detecção em F1@0,015, o VLM vira o módulo de semântica e o sistema segue
híbrido. Se empatar, o VLM puro é aceitável e mais simples de manter.

Nota sobre o RunPod: a conta está acessível, sem pods, com cerca de US$ 6 de
saldo. O bake-off inteiro custa perto de 25 horas de A100, ou seja, algo
entre US$ 40 e US$ 60 nos preços atuais. Será preciso adicionar crédito.

## Restrição de produção: easypanel-max com 6 a 8 GB de RAM e sem GPU

Adicionado em 2026-09-24 depois da definição do ambiente de produção.
O treino pode ir para o RunPod; a inferência precisa rodar no easypanel-max,
que tem entre 6 e 8 GB de RAM, sem GPU, compartilhados com o sistema, o
Docker do EasyPanel e os outros serviços (EVOAPI etc.). Orçamento realista
para o extrator: **3 a 4 GB de RAM em pico e poucos vCPUs.**

### O que cabe e o que não cabe

| Componente | Pico de RAM em CPU | Tempo por planta (estimativa, 4 vCPUs) | Veredito |
|---|---|---|---|
| PyMuPDF (extração de paths do PDF vetorial) + regras | < 300 MB | < 1 s | Cabe com folga |
| SegFormer-B2 / U-Net para máscara de parede (MitUNet, ~25M parâmetros), ONNX int8 | < 1 GB | 2 a 10 s em 1024 px | Cabe com folga |
| Detecção do fpvec-lab (ResNet + deformable attention, ~50M parâmetros) | ~1 GB | 2 a 5 s | Cabe; precisa retreinar em 512 px, os 256 px originais são pouco para planta grande |
| Florence-2-base (0,23B) / large (0,77B) | 1 a 3 GB | 5 a 30 s | Cabe; entrada fixa em 768 px, exige tiling |
| Qwen3-VL-2B em GGUF Q4 + projetor de visão (llama.cpp) | 3,5 a 4 GB | 3 a 6 min (prefill de ~3k tokens visuais + geração de ~2k tokens de JSON) | Cabe no limite; lento; a quantização em 4 bits degrada justamente as coordenadas |
| Qwen3.5-0.8B | ~2 GB | 1 a 3 min | Cabe, mas não há evidência de que 0,8B acerte geometria |
| Qwen3-VL-4B Q4 | 5 a 6 GB | 6 a 12 min | Não cabe com os outros serviços |
| Qwen3-VL-8B / GLM-4.6V-Flash 9B | > 6 GB mesmo em Q4 | dezenas de minutos | Não cabe |

A conclusão é que **o VLM não pode ser o núcleo do sistema no easypanel-max**.
Isso reforça a recomendação híbrida da seção anterior: a geometria fica com
componentes pequenos e determinísticos, que são justamente os que se saíram
melhor no artigo de He Zhang.

### Arquitetura recomendada para essa restrição

```
PDF ──► PyMuPDF ──► tem paths vetoriais?
                      │
        sim ──────────┴────────── não (raster / escaneado)
         │                              │
  classificador de linhas         renderiza 1024 px
  (regras + modelo leve,          ──► SegFormer/U-Net int8 (máscara de parede)
   estilo VecFormer)              ──► esqueletização + vetorização (RDP)
         │                              │
         └──────────┬───────────────────┘
                    ▼
        JSON de paredes com coordenadas exatas
                    │
                    ▼  (opcional, só quando precisar de semântica:
                        escala pelas cotas, nome de cômodo, tipo de abertura)
        VLM fora do servidor: RunPod Serverless com Qwen3-VL-8B + LoRA
        (escala a zero; paga só os segundos de uso)
```

Custo do VLM remoto: um worker serverless com L4 ou A10 custa na faixa de
US$ 0,0003 a 0,0005 por segundo. Uma planta leva de 10 a 30 s, ou seja,
**menos de US$ 0,02 por planta**. O saldo atual de US$ 6 cobre centenas de
plantas. Se o volume crescer, o mesmo endpoint escala sem mexer no
easypanel-max.

### O que treinar no RunPod

| Treino | Hardware | Tempo | Sai para o easypanel-max? |
|---|---|---|---|
| SegFormer-B2 em CubiCasa5K + ResPlan renderizado + dados próprios | 1 A100 40/80 GB | 3 a 6 h | Sim, exportado em ONNX int8 (~30 MB) |
| Detecção do fpvec-lab em 512 px | 1 A100 | 4 a 8 h | Sim, ONNX |
| Classificador de primitivas vetoriais em FloorPlanCAD | 1 A100 ou até CPU | 1 a 3 h | Sim, é um modelo pequeno |
| LoRA em Qwen3-VL-8B para semântica e JSON | 1 A100 80 GB | ~8 h | Não; fica no RunPod Serverless, mesclado e quantizado em AWQ |
| LoRA em Qwen3-VL-2B (só se quiser VLM local) | 1 A100 | ~4 h | GGUF Q4 via llama.cpp, aceitando 3 a 6 min por planta |

### Decisão

1. Núcleo no easypanel-max: PyMuPDF + SegFormer int8 + vetorização. Sem VLM
   local na primeira versão.
2. VLM (Qwen3-VL-8B com LoRA) no RunPod Serverless, chamado por HTTP só
   para semântica. Se o custo por planta ficar acima do aceitável, a
   alternativa local é Qwen3-VL-2B Q4, com a lentidão e a perda de precisão
   descritas acima.
3. O bake-off da seção anterior continua válido, mas o critério muda: o
   modelo de detecção precisa ser bom o bastante sozinho, porque é ele que
   roda no servidor. O VLM só é comparado no papel de módulo de semântica.

## Fontes

- FloorplanVLM: https://arxiv.org/abs/2602.06507
- He Zhang, emit vs detect: https://arxiv.org/abs/2608.25608 e https://github.com/Cyprinus12138/fpvec-lab
- Raster2Seq: https://arxiv.org/abs/2602.09016 e https://github.com/Cornell-VAILab/Raster2Seq
- HouseMind (tokenização de plantas, CVPR 2026, base Qwen3-0.6B): https://arxiv.org/abs/2603.11640
- SALI-FP: https://arxiv.org/abs/2609.25615
- Agentic CV para plantas estruturais: https://arxiv.org/abs/2608.17237
- MitUNet / Floor Plan CIS: https://arxiv.org/abs/2512.02413
- VecFormer: https://arxiv.org/abs/2505.23395
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
- Qwen3.5: https://qwen.ai/blog?id=qwen3.5
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- GLM-4.6V: https://venturebeat.com/ai/z-ai-debuts-open-source-glm-4-6v-a-native-tool-calling-vision-model-for
- InternVL3.5: https://arxiv.org/abs/2508.18265
- Gemma 4 para visão computacional: https://datature.io/blog/gemma-4-what-computer-vision-engineers-actually-need-to-know
- Molmo2: https://arxiv.org/abs/2601.10611
- Florence-2 fine-tuning: https://blog.roboflow.com/fine-tune-florence-2-object-detection/
- Reimplementação FloorPlanVLM: https://github.com/miladmirzazadeh/FloorPlanVLM
- Adapters públicos: https://huggingface.co/mudasir13cs/qwen25-vl-3b-floorplan-sft e https://huggingface.co/manitocross/floorplan-vlm-training
- Unsloth Qwen3.5: https://unsloth.ai/docs/models/qwen3.5/fine-tune
