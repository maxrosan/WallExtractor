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

Em andamento. Mesmo pod, batch 4, teto de 35 min.
