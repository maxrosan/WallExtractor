#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rascunho por pares de faces: paredes e aberturas direto da imagem da planta (Pillow + numpy).

Escrito pelo Claude durante as correções em lote do portfólio de SJC (relatório em
docs/experimentos.md, "Lotes da IA") e trazido para o projeto como opção de rascunho no editor
("Refazer rascunho com detector de faces"). É heurística de estilo: lê paredes como pares de linhas
paralelas a uma espessura estimada da própria imagem e portas/janelas pelas cores laranja/azul do
desenho de BIM; serve para acelerar a correção, não substitui a revisão. Mudança em relação ao
original: espessuras secundárias a partir de 20% do peso da principal e de 0,6·t (paredes internas
desenhadas mais finas que as externas).

Tudo trabalha em pixels da imagem; os parâmetros são relativos ao tamanho da imagem ou à
espessura da parede (`t`, em px, estimada da própria imagem). Uso direto:

  python -m wallextractor.faces auto planta.png > rascunho.json
  python -m wallextractor.faces conferir rascunho.json planta.png

Outros comandos: perfil, espessura, paredes, aberturas, limpar, comparar, editar, finalizar, lote,
estatisticas. Cada função tem um comentário dizendo que problema resolve.
"""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------------------------
# 0. Utilidades básicas
# --------------------------------------------------------------------------------------------


def carregar(path):
    """Abre o PNG e devolve (imagem RGB como array int, imagem em cinza como array)."""
    im = Image.open(path).convert("RGB")
    rgb = np.array(im).astype(int)
    gray = np.array(im.convert("L"))
    return rgb, gray


def mascara_escura(gray, limiar=128, rgb=None, saturacao_max=60):
    """Problema: separar linhas de desenho (preto/cinza) de fundo, hachura clara e CORES.
    `limiar` em 0-255: faces de parede ficam tipicamente < 130 (algumas saem cinza ~120 quando o
    editor exporta camadas com transparência). Magenta (louças) e azul escuro têm luminância baixa
    e entrariam como 'escuro': se `rgb` for dado, pixels com saturação (max-min dos canais) acima
    de `saturacao_max` são excluídos — linha de parede é cinza, não colorida."""
    m = gray < limiar
    if rgb is not None:
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        m &= sat <= saturacao_max
    return m


def runs(vetor, min_len=1):
    """Problema: saber onde uma linha começa e termina (e onde é interrompida por um vão).
    Devolve [(inicio, fim)] dos trechos True consecutivos com comprimento >= min_len."""
    out, s = [], None
    for i, v in enumerate(vetor):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                out.append((s, i - 1))
            s = None
    if s is not None and len(vetor) - s >= min_len:
        out.append((s, len(vetor) - 1))
    return out


def agrupar_consecutivos(indices, folga=0):
    """Agrupa índices consecutivos (ou separados por <= folga) em intervalos (a, b)."""
    out, s, p = [], None, None
    for i in indices:
        if s is None:
            s = i
        elif i > p + 1 + folga:
            out.append((s, p))
            s = i
        p = i
    if s is not None:
        out.append((s, p))
    return out


def dist_ponto_segmento(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L)) if L else 0.0
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def orientacao(w, tol_px=3):
    """'h', 'v' ou 'd' (diagonal) de uma parede/abertura."""
    dx = abs(w["end"][0] - w["start"][0])
    dy = abs(w["end"][1] - w["start"][1])
    if dy <= tol_px and dx > dy:
        return "h"
    if dx <= tol_px and dy > dx:
        return "v"
    return "d"


def comprimento(w):
    return math.dist(w["start"], w["end"])


# --------------------------------------------------------------------------------------------
# 1. Leitura da imagem: perfis, faces, espessura
# --------------------------------------------------------------------------------------------


def linhas_longas(mask, eixo="ambos", min_frac=0.15):
    """Problema: achar rapidamente as faces de parede (linhas retas longas) sem olhar a imagem.
    Para cada linha/coluna conta pixels escuros; devolve as que têm mais que `min_frac` do
    tamanho da imagem naquela direção. `min_frac` relativo (0.15 = 15% da largura/altura).
    Retorna {'rows': [(y, n)], 'cols': [(x, n)]}; linhas de 3 px aparecem como 3 índices vizinhos."""
    h, w = mask.shape
    out = {"rows": [], "cols": []}
    if eixo in ("ambos", "h"):
        rs = mask.sum(axis=1)
        out["rows"] = [(int(y), int(rs[y])) for y in range(h) if rs[y] >= min_frac * w]
    if eixo in ("ambos", "v"):
        cs = mask.sum(axis=0)
        out["cols"] = [(int(x), int(cs[x])) for x in range(w) if cs[x] >= min_frac * h]
    return out


def centros_de_linhas(indices_com_contagem, folga=1):
    """Problema: uma linha de 3 px vira 3 linhas vizinhas no perfil. Agrupa e devolve o centro
    de cada grupo: [(centro, contagem_max)]."""
    grupos = agrupar_consecutivos([i for i, _ in indices_com_contagem], folga)
    cont = dict(indices_com_contagem)
    return [((a + b) / 2.0, max(cont.get(i, 0) for i in range(a, b + 1))) for a, b in grupos]


def perfil_pixels(gray, y=None, x=None, de=0, ate=None, max_val=230):
    """Problema: distinguir face de parede (escura), linha de cota (cinza) e azulejo numa fatia.
    Lista (posição, valor) dos pixels < max_val ao longo da linha y (ou coluna x) entre `de` e `ate`."""
    if y is not None:
        v = gray[y, de:ate]
    else:
        v = gray[de:ate, x]
    return [(int(i + de), int(p)) for i, p in enumerate(v) if p < max_val]


def faces_do_perfil(soma, minimo, largura_max, salto=0.25):
    """Problema: duas linhas encostadas (parede + caixilho, parede + bancada) viram um grupo só
    no perfil e o centro cai no meio das duas. Agrupa índices consecutivos com contagem >= minimo,
    mas corta o grupo onde a contagem muda mais que `salto` (25%) de uma coluna para a vizinha —
    linhas distintas quase sempre têm comprimentos distintos. Um pedaço de até 2 px colado num
    grupo vizinho é anti-aliasing (ou linha coincidente) e volta a ser juntado a ele. Descarta
    grupos mais largos que largura_max (preenchimento, hachura). Devolve a lista de centros (int)."""
    idx = [i for i in range(len(soma)) if soma[i] >= minimo]
    grupos, s, p = [], None, None
    for i in idx:
        if s is None:
            s = i
        elif i != p + 1 or abs(soma[i] - soma[p]) > salto * max(soma[i], soma[p]):
            grupos.append((s, p))
            s = i
        p = i
    if s is not None:
        grupos.append((s, p))
    juntos = []
    for g in grupos:
        if juntos and g[0] == juntos[-1][1] + 1 and (g[1] - g[0] < 2 or juntos[-1][1] - juntos[-1][0] < 2):
            juntos[-1] = (juntos[-1][0], g[1])
        else:
            juntos.append(g)
    return [int(round((a + b) / 2)) for a, b in juntos if b - a + 1 <= largura_max]


def ruido_central(mask, eixo, a, b, s, e):
    """Problema: distinguir o miolo de uma parede (vazio) do espaço entre a face e a linha de cota
    (textos de cota, linhas de chamada). Fração do trecho [s,e] com algum pixel escuro no terço
    central entre as faces a < b. Parede limpa ~0; faixa de cota tipicamente > 0,2 %."""
    d = b - a
    lo, hi = a + max(2, d // 3), b - max(2, d // 3) + 1
    if hi <= lo:
        return 0.0
    miolo = mask[s:e + 1, lo:hi] if eixo == "v" else mask[lo:hi, s:e + 1]
    return float(miolo.any(axis=1 if eixo == "v" else 0).mean())


def estimar_espessura(mask, face_min_frac=0.08, t_min_frac=0.005, t_max_frac=0.05, ruido_max=0.002):
    """Problema: a espessura da parede (em px) muda de planta para planta (escala desconhecida).
    Pega as faces retas longas (trechos >= face_min_frac do menor lado), mede a distância entre
    pares de faces paralelas vizinhas e soma, para cada distância, o comprimento em que as duas
    faces coexistem lado a lado. A distância com maior comprimento acumulado é a espessura
    típica (paredes externas pesam muito; azulejo e cota, pouco). Restringe a
    [t_min_frac, t_max_frac] do menor lado (969 px -> 5..48 px). Devolve (t, histograma_ponderado)."""
    h, w = mask.shape
    lado = min(h, w)
    tmin, tmax = max(2, lado * t_min_frac), lado * t_max_frac
    face_min = int(lado * face_min_frac)
    hist = Counter()
    for eixo in ("h", "v"):
        longo = so_trechos_longos(mask, eixo, face_min)
        soma = longo.sum(axis=1) if eixo == "h" else longo.sum(axis=0)
        fs = faces_do_perfil(soma, face_min, tmax / 2)
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                d = b - a
                if d > tmax:
                    break
                if d < tmin:
                    continue
                la = longo[a] if eixo == "h" else longo[:, a]
                lb = longo[b] if eixo == "h" else longo[:, b]
                for s, e in runs(la & lb, face_min):
                    # só pares de miolo limpo (parede); entre face e cota há texto/linha de chamada
                    if ruido_central(mask, eixo, a, b, s, e) <= ruido_max:
                        hist[int(d)] += e - s + 1
    if not hist:
        return None, hist
    # junta vizinhos (±1 px) para não dividir a moda entre 25 e 26
    suav = Counter({d: hist[d - 1] + hist[d] + hist[d + 1] for d in hist})
    return suav.most_common(1)[0][0], hist


def espessuras_candidatas(hist, principal, min_frac=0.2, separacao=2, faixa=(0.6, 1.5)):
    """Problema: a mesma planta pode ter paredes de duas espessuras (ex.: 26 px nas internas e 30 px
    nas externas de um lado, por linhas mais grossas no desenho, ou internas mais finas). Devolve
    [principal] + picos secundários do histograma ponderado com peso >= min_frac do pico principal e
    afastados mais de `separacao` px dos já escolhidos, dentro de `faixa` × principal (fora disso é
    azulejo ou parede + cota, não uma segunda espessura de parede)."""
    if not hist:
        return [principal]
    suav = Counter({d: hist[d - 1] + hist[d] + hist[d + 1] for d in hist})
    base = suav[principal]
    out = [principal]
    for d, peso in suav.most_common():
        if peso < min_frac * base:
            break
        if faixa[0] * principal <= d <= faixa[1] * principal and all(abs(d - e) > separacao for e in out):
            out.append(d)
    return out


def so_trechos_longos(mask, eixo, min_len):
    """Problema: texto, hachura e mobiliário geram muitos pixels escuros que confundem o perfil.
    Mantém, em cada linha (eixo 'h') ou coluna ('v'), só os trechos escuros contínuos >= min_len."""
    out = np.zeros_like(mask)
    if eixo == "h":
        for y in range(mask.shape[0]):
            for s, e in runs(mask[y], min_len):
                out[y, s:e + 1] = True
    else:
        for x in range(mask.shape[1]):
            for s, e in runs(mask[:, x], min_len):
                out[s:e + 1, x] = True
    return out


def detectar_paredes(mask, t, face_min=None, sobreposicao_min=None, largura_max_linha=None, ruido_max=0.002, penal_ruido=0.25, tol_t=0.15):
    """Problema: achar paredes como PARES de faces paralelas (duas linhas a ~t px), em vez de
    tomar qualquer linha isolada (azulejo, cota, projeção) como parede.
    - t: espessura esperada em px (de estimar_espessura) ou lista [principal, secundárias...]
      (de espessuras_candidatas); aceita d dentro de ±tol_t de qualquer uma. 15 % é apertado de
      propósito: a linha de cota costuma ficar a ~1,2·t da face externa.
    - face_min: comprimento mínimo de um trecho reto para contar como face (padrão 2*t);
      trechos curtos (texto, hachura, louças) são descartados antes do perfil.
    - sobreposicao_min: comprimento mínimo de parede (padrão 0.5*t) — guarda os tocos entre
      porta e canto; ligar_cantos depois estica até o cruzamento.
    - largura_max_linha: grupos de colunas/linhas escuras mais largos que isso (padrão t/2)
      não são linha de desenho (preenchimento, hachura) e são ignorados.
    Para cada par de faces a ~t, a interseção dos trechos escuros das duas vira parede no eixo
    médio; onde uma das faces some (vão) o trecho é cortado sozinho."""
    ts = list(t) if isinstance(t, (list, tuple)) else [t]
    t = ts[0]
    face_min = int(2 * t) if face_min is None else int(face_min)
    min_len = 0.5 * t if sobreposicao_min is None else sobreposicao_min
    largura_max_linha = t / 2 if largura_max_linha is None else largura_max_linha
    candidatos = []
    for eixo in ("h", "v"):
        longo = so_trechos_longos(mask, eixo, face_min)
        soma = longo.sum(axis=1) if eixo == "h" else longo.sum(axis=0)
        fs = faces_do_perfil(soma, face_min, largura_max_linha)
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                d = b - a
                if d > max(ts) * (1 + tol_t):
                    break
                desvio = min(abs(d - te) for te in ts)
                if not any(te * (1 - tol_t) <= d <= te * (1 + tol_t) for te in ts):
                    continue
                # as FACES vêm do perfil filtrado (linhas longas); a EXTENSÃO vem da máscara crua,
                # para não perder os tocos curtos entre um vão e um canto
                la = mask[a] if eixo == "h" else mask[:, a]
                lb = mask[b] if eixo == "h" else mask[:, b]
                la = la | np.roll(la, 1) | np.roll(la, -1)
                lb = lb | np.roll(lb, 1) | np.roll(lb, -1)
                for s, e in runs(la & lb, int(min_len)):
                    # "ruído" entre as faces: fração do trecho com pixel escuro no miolo da parede.
                    # Miolo de parede é vazio; entre a face e a linha de cota há textos e tiques.
                    ruido = ruido_central(mask, eixo, a, b, s, e)
                    candidatos.append((desvio, eixo, a, b, d, s, e, ruido))
    # Uma linha é face de UMA parede só (não pode ser face de duas paredes encostadas, nem de
    # uma parede e de uma 'parede' feita com a linha de cota ao lado). Em caso de disputa pelo
    # mesmo trecho da mesma face, fica o par com espessura mais perto de t e, empatando, o mais
    # longo. Par com miolo "sujo" (ruido_central > ruido_max: texto de cota, linha de chamada entre
    # a face e a linha de cota) paga penal_ruido*t na fila — dentro de uma parede não há nada.
    usadas = defaultdict(list)  # (eixo, face) -> [(s, e)]
    paredes = []
    for _, eixo, a, b, d, s, e, ruido in sorted(candidatos, key=lambda c: (c[0] + (penal_ruido * t if c[7] > ruido_max else 0), -(c[6] - c[5]))):
        # faces a <= 2 px uma da outra são a mesma linha (anti-aliasing / texto encostado)
        vizinhas = [f for face in (a, b) for f in range(face - 2, face + 3)]
        if any(not (e < us or s > ue) for f in vizinhas for us, ue in usadas.get((eixo, f), ())):
            continue
        usadas[(eixo, a)].append((s, e))
        usadas[(eixo, b)].append((s, e))
        eixo_pos = (a + b) / 2.0
        if eixo == "h":
            paredes.append({"start": [s, eixo_pos], "end": [e, eixo_pos], "thickness": float(d)})
        else:
            paredes.append({"start": [eixo_pos, s], "end": [eixo_pos, e], "thickness": float(d)})
    return renumerar(paredes)


# --------------------------------------------------------------------------------------------
# 2. Limpeza da lista de paredes (serve para o planta.json da máquina)
# --------------------------------------------------------------------------------------------


def renumerar(paredes, prefixo="w"):
    for i, p in enumerate(paredes, 1):
        p["id"] = f"{prefixo}{i}"
    return paredes


def mesclar_colineares(paredes, t, tol_eixo=None, gap_max=None, tol_t=0.5):
    """Problema: parede real picotada em dezenas de trechos (ex.: 34 pedaços numa parede externa).
    Junta trechos com a mesma orientação, eixo a menos de `tol_eixo` (padrão t/2) e separados por
    até `gap_max` (padrão 1.5*t: cobre o buraco que um encontro em T deixa numa das faces;
    vãos de porta/janela são bem maiores que isso e ficam). A espessura vira a mediana.
    Trechos diagonais não são mesclados."""
    tol_eixo = t / 2 if tol_eixo is None else tol_eixo
    gap_max = 1.5 * t if gap_max is None else gap_max
    grupos = defaultdict(list)
    for p in paredes:
        o = orientacao(p)
        if o == "d":
            grupos[("d", id(p))].append(p)
            continue
        eixo = p["start"][1] if o == "h" else p["start"][0]
        chave = None
        for k in grupos:
            if k[0] == o and abs(k[1] - eixo) <= tol_eixo:
                chave = k
                break
        grupos[chave or (o, eixo)].append(p)
    saida = []
    for (o, eixo), ps in grupos.items():
        if o == "d":
            saida.extend(ps)
            continue
        i = 0 if o == "h" else 1
        segs = sorted(((min(p["start"][i], p["end"][i]), max(p["start"][i], p["end"][i]), p["thickness"]) for p in ps))
        eixo_med = float(np.median([(p["start"][1 - i] + p["end"][1 - i]) / 2 for p in ps]))
        atual = list(segs[0])
        ts = [segs[0][2]]
        for a, b, th in segs[1:]:
            if a - atual[1] <= gap_max:
                atual[1] = max(atual[1], b)
                ts.append(th)
            else:
                saida.append(_seg(o, eixo_med, atual[0], atual[1], float(np.median(ts))))
                atual, ts = [a, b, th], [th]
        saida.append(_seg(o, eixo_med, atual[0], atual[1], float(np.median(ts))))
    return renumerar(saida)


def _seg(o, eixo, a, b, t):
    if o == "h":
        return {"start": [a, eixo], "end": [b, eixo], "thickness": t}
    return {"start": [eixo, a], "end": [eixo, b], "thickness": t}


def filtrar_paredes(paredes, t, t_min=0.5, t_max=1.8, len_min=0.5, manter_diagonais=False):
    """Problema: 'paredes' que são linha simples (azulejo, cota, projeção, caixilho, degrau),
    trechos minúsculos ou espessuras absurdas (parede + linha de cota contadas como uma só).
    Mantém só espessura em [t_min*t, t_max*t] e comprimento >= len_min*t. Devolve (mantidas, removidas)."""
    ok, fora = [], []
    for p in paredes:
        th = p["thickness"]
        if not (t_min * t <= th <= t_max * t) or comprimento(p) < len_min * t or (orientacao(p) == "d" and not manter_diagonais):
            fora.append(p)
        else:
            ok.append(p)
    return renumerar(ok), fora


def remover_isoladas(paredes, t, len_max=4.0, alcance=1.5):
    """Problema: pares de linhas de mobiliário/louça (pia, tanque, bancada) com a mesma distância
    da parede viram 'paredinhas' soltas. Remove trechos curtos (< len_max*t) cuja nenhuma ponta
    está a menos de alcance*t de outra parede. Devolve (mantidas, removidas)."""
    ok, fora = [], []
    for p in paredes:
        if comprimento(p) >= len_max * t:
            ok.append(p)
            continue
        toca = any(q is not p and min(dist_ponto_segmento(p[k], q["start"], q["end"]) for k in ("start", "end")) <= alcance * t for q in paredes)
        (ok if toca else fora).append(p)
    return renumerar(ok), fora


def normalizar_espessura(paredes, t, tol=0.35):
    """Problema: espessuras variando 22..38 px na mesma parede de 15 cm. Quem está a menos de
    `tol*t` de um valor típico (t ou lista de espessuras) recebe esse valor; o resto fica."""
    ts = list(t) if isinstance(t, (list, tuple)) else [t]
    for p in paredes:
        te = min(ts, key=lambda v: abs(p["thickness"] - v))
        if abs(p["thickness"] - te) <= tol * ts[0]:
            p["thickness"] = float(te)
    return paredes


def ligar_cantos(paredes, t, alcance=None):
    """Problema: pontas de parede que param antes do encontro ou passam dele ('cantos abertos').
    Para cada ponta, se há uma parede perpendicular cujo eixo passa a menos de `alcance`
    (padrão 1.5*t), move a ponta até o cruzamento dos eixos — desde que o cruzamento esteja do lado
    de fora dessa ponta ou a menos de t/2 dela (a ponta que fica na borda de um vão perto do canto
    não é puxada até o canto)."""
    alcance = 1.5 * t if alcance is None else alcance
    for p in paredes:
        o = orientacao(p)
        if o == "d":
            continue
        for chave in ("start", "end"):
            x, y = p[chave]
            for q in paredes:
                if q is p or orientacao(q) != ("v" if o == "h" else "h"):
                    continue
                outro = p["end"] if chave == "start" else p["start"]
                if o == "h":
                    ex = q["start"][0]
                    ya, yb = sorted((q["start"][1], q["end"][1]))
                    novo = [ex, y]
                    perto = abs(x - ex) <= alcance and ya - alcance <= y <= yb + alcance
                else:
                    ey = q["start"][1]
                    xa, xb = sorted((q["start"][0], q["end"][0]))
                    novo = [x, ey]
                    perto = abs(y - ey) <= alcance and xa - alcance <= x <= xb + alcance
                # o cruzamento tem de estar do lado de fora desta ponta (prolongar) ou, se estiver
                # dentro do trecho, a menos de t/2 dela — nunca do lado da outra ponta
                k = 0 if o == "h" else 1
                c, pk, qk = novo[k], p[chave][k], outro[k]
                fora_da_ponta = (c - pk) * (pk - qk) >= 0
                if perto and (fora_da_ponta or abs(c - pk) <= t / 2):
                    p[chave] = novo
                    break
    return paredes


def dividir_nos_encontros(paredes, t, alcance=None, aberturas=()):
    """Problema: o editor quer a parede DIVIDIDA em todo encontro (L, T, cruz): num T a parede que
    continua vira dois trechos terminando no eixo da que chega. Para cada parede, acha os eixos das
    paredes perpendiculares que a tocam (ponta da outra a <= alcance, padrão t, do eixo desta) em
    pontos interiores e corta ali. Aberturas contam como parte da parede delas (uma parede que
    chega num vão encostado no canto ainda é um encontro). Trechos menores que t/2 depois do corte
    são descartados."""
    alcance = t if alcance is None else alcance
    elementos = list(paredes) + list(aberturas)
    saida = []
    for p in paredes:
        o = orientacao(p)
        if o == "d":
            saida.append(p)
            continue
        i = 0 if o == "h" else 1
        a, b = sorted((p["start"][i], p["end"][i]))
        eixo = p["start"][1 - i]
        cortes = set()
        for q in elementos:
            if q is p or orientacao(q) != ("v" if o == "h" else "h"):
                continue
            pos = q["start"][i]            # coordenada do eixo de q ao longo de p
            if not (a + alcance < pos < b - alcance):
                continue
            # q tem de chegar até p: alguma ponta de q a <= alcance do eixo de p, ou q cruza p
            qa, qb = sorted((q["start"][1 - i], q["end"][1 - i]))
            if qa - alcance <= eixo <= qb + alcance:
                cortes.add(pos)
        pontos = [a] + sorted(cortes) + [b]
        for s, e in zip(pontos, pontos[1:]):
            if e - s >= t / 2:
                saida.append(_seg(o, eixo, s, e, p["thickness"]))
    return renumerar(saida)


def remover_degeneradas(paredes, t, len_min=0.25):
    """Problema: depois de ligar cantos e cortar, sobram trechos de comprimento ~0 (toco entre o
    eixo da parede e a borda de um vão encostado nela). Remove os menores que len_min*t."""
    return renumerar([p for p in paredes if comprimento(p) >= len_min * t])


def arredondar(objs, casas=0):
    for o in objs:
        o["start"] = [round(v, casas) if casas else int(round(v)) for v in o["start"]]
        o["end"] = [round(v, casas) if casas else int(round(v)) for v in o["end"]]
        if "thickness" in o:
            o["thickness"] = round(o["thickness"], 1)
    return objs


# --------------------------------------------------------------------------------------------
# 3. Aberturas: marcos de porta (cor) e caixilhos de janela (cor) sobre o eixo da parede
# --------------------------------------------------------------------------------------------

# Cores típicas de CAD exportado: porta/marco em laranja, janela em azul. Ajuste se a planta
# usar outras cores (intervalos inclusivos em RGB 0-255).
COR_PORTA = {"r": (200, 255), "g": (120, 200), "b": (0, 140)}
COR_JANELA = {"r": (0, 150), "g": (100, 255), "b": (200, 255)}


def mascara_cor(rgb, cor):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return ((r >= cor["r"][0]) & (r <= cor["r"][1]) & (g >= cor["g"][0]) & (g <= cor["g"][1])
            & (b >= cor["b"][0]) & (b <= cor["b"][1]))


def faces_continuas(escuro, o, eixo, espessura, s, e, raio=2):
    """Problema: em plantas com a parede PINTADA da cor do caixilho (ex.: cópias com paredes
    preenchidas de azul), a cor sozinha acha 'janela' ao longo da parede inteira. Num vão de
    verdade as duas faces da parede são interrompidas (param nos batentes e só o caixilho passa).
    Devolve a fração de [s, e] em que as DUAS faces (eixo ± espessura/2) têm traço escuro."""
    h, w = escuro.shape
    lim = w if o == "h" else h
    s, e = max(0, int(s)), min(lim - 1, int(e))
    if e <= s:
        return 0.0
    presentes = []
    for f in (eixo - espessura / 2, eixo + espessura / 2):
        lo, hi = max(0, int(round(f)) - raio), min((h if o == "h" else w) - 1, int(round(f)) + raio)
        faixa = escuro[lo:hi + 1, s:e + 1].any(axis=0) if o == "h" else escuro[s:e + 1, lo:hi + 1].any(axis=1)
        presentes.append(faixa)
    return float((presentes[0] & presentes[1]).mean())


def medir_aberturas(rgb, paredes, t, cor_porta=COR_PORTA, cor_janela=COR_JANELA,
                    faixa=0.7, folga=None, largura_min=1.5, faces_max=0.6):
    """Problema: achar portas e janelas e a largura exata do vão, e não confundir uma com a outra.
    Varre uma faixa de ±faixa*t em volta do eixo de cada parede, ao longo de toda a imagem,
    procurando pixels da cor de porta (marcos) e de janela (caixilho); guarda só os vãos que
    encostam na parede (a parede detectada já vem cortada no vão, o vão fica entre dois trechos). Grupos de colunas/linhas coloridas separados por menos
    de `folga` (padrão 2.5*t) formam um vão; porta = do começo do primeiro marco ao fim do último
    (borda externa dos marcos = onde a linha da parede é interrompida); janela = extensão do
    caixilho. Vãos menores que largura_min*t são ignorados (etiquetas, folhas de porta soltas).
    Uma janela em que as duas faces da parede continuam em mais de `faces_max` do comprimento é a
    própria parede colorida, não abertura (`faces_continuas`); nas portas não se aplica, porque a
    soleira é desenhada como duas linhas atravessando o vão.
    Devolve lista de openings no eixo da parede."""
    folga = 2.5 * t if folga is None else folga
    mp, mj = mascara_cor(rgb, cor_porta), mascara_cor(rgb, cor_janela)
    escuro = mascara_escura(rgb.mean(axis=2), 128, rgb)
    h, w = mp.shape
    aberturas = []
    for p in paredes:
        o = orientacao(p)
        if o == "d":
            continue
        i = 0 if o == "h" else 1
        a, b = sorted((p["start"][i], p["end"][i]))
        eixo = p["start"][1 - i]
        lo, hi = int(max(0, eixo - faixa * t)), int(min((h if o == "h" else w) - 1, eixo + faixa * t))
        for tipo, m in (("door", mp), ("window", mj)):
            if o == "h":
                col = m[lo:hi + 1, :].any(axis=0)
            else:
                col = m[:, lo:hi + 1].any(axis=1)
            idx = [k for k in range(len(col)) if col[k]]
            for s, e in agrupar_consecutivos(idx, int(folga)):
                if e - s < largura_min * t:
                    continue
                # o vão tem de encostar numa ponta da parede (ou sobrepô-la): o resto é de outra parede
                if e < a - 1.5 * t or s > b + 1.5 * t:
                    continue
                # windows only: a door's threshold is drawn as two lines across the opening, like faces
                if tipo == "window" and faces_continuas(escuro, o, eixo, p["thickness"], s, e) > faces_max:
                    continue  # the wall runs on through the coloured stretch: painted wall, not a window
                st = [s, eixo] if o == "h" else [eixo, s]
                en = [e, eixo] if o == "h" else [eixo, e]
                aberturas.append({"type": tipo, "start": st, "end": en, "wall_id": p["id"]})
    # remove duplicatas (mesma abertura vista de duas paredes vizinhas)
    unicas = []
    for ab in aberturas:
        if not any(ab["type"] == u["type"] and math.dist(ab["start"], u["start"]) < t and math.dist(ab["end"], u["end"]) < t for u in unicas):
            unicas.append(ab)
    for k, ab in enumerate(unicas, 1):
        ab["id"] = f"o{k}"
    return unicas


def atribuir_wall_id(aberturas, paredes, t):
    """Problema: depois de cortar/renumerar paredes, o wall_id das aberturas fica órfão.
    Dá a cada abertura o id da parede colinear mais próxima de uma das suas pontas."""
    for ab in aberturas:
        o = orientacao(ab)
        cands = [p for p in paredes if orientacao(p) == o]
        if cands:
            melhor = min(cands, key=lambda p: min(dist_ponto_segmento(ab[k], p["start"], p["end"]) for k in ("start", "end")))
            ab["wall_id"] = melhor["id"]
    return aberturas


def remover_paredes_em_vaos(paredes, aberturas, t, alcance=1.5, fracao_min=0.5):
    """Problema: soleira de porta (duas linhas na espessura da parede atravessando o vão) ou a linha
    de cota ao lado do vão formam uma 'parede' paralela à verdadeira, deslocada ~t, bem no vão.
    Remove paredes paralelas a uma abertura, com eixo a <= alcance*t do eixo dela, cuja sobreposição
    com o vão cobre >= fracao_min do próprio comprimento da parede."""
    saida = []
    for p in paredes:
        o = orientacao(p)
        i = 0 if o == "h" else 1
        pa, pb = sorted((p["start"][i], p["end"][i]))
        lixo = False
        for ab in aberturas:
            if orientacao(ab) != o or abs(ab["start"][1 - i] - p["start"][1 - i]) > alcance * t:
                continue
            if abs(ab["start"][1 - i] - p["start"][1 - i]) < 1:
                continue  # mesmo eixo: cortar_nos_vaos cuida
            aa, bb = sorted((ab["start"][i], ab["end"][i]))
            sobre = min(pb, bb) - max(pa, aa)
            if sobre > 0 and sobre >= fracao_min * (pb - pa):
                lixo = True
                break
        if not lixo:
            saida.append(p)
    return renumerar(saida)


def remover_aberturas_orfas(aberturas, paredes, t, alcance=1.0):
    """Problema: abertura medida numa 'parede' que depois foi removida (soleira, linha de cota) fica
    órfã e duplica a abertura verdadeira. Mantém só aberturas com uma parede colinear (mesmo eixo,
    a <= t/2) encostando numa das pontas a <= alcance*t."""
    saida = []
    for ab in aberturas:
        o = orientacao(ab)
        i = 0 if o == "h" else 1
        ok = False
        for p in paredes:
            if orientacao(p) != o or abs(p["start"][1 - i] - ab["start"][1 - i]) > t / 2:
                continue
            if min(dist_ponto_segmento(ab[k], p["start"], p["end"]) for k in ("start", "end")) <= alcance * t:
                ok = True
                break
        if ok:
            saida.append(ab)
    for k, ab in enumerate(saida, 1):
        ab["id"] = f"o{k}"
    return saida


def cortar_nos_vaos(paredes, aberturas, t):
    """Problema: parede contínua por cima de porta/janela (o editor espera a parede interrompida
    em cada vão). Para cada abertura sobre o eixo de uma parede, divide a parede em dois trechos,
    um terminando numa borda do vão e outro começando na outra."""
    saida = list(paredes)
    for ab in aberturas:
        o = orientacao(ab)
        i = 0 if o == "h" else 1
        a, b = sorted((ab["start"][i], ab["end"][i]))
        eixo = ab["start"][1 - i]
        novos = []
        for p in saida:
            if orientacao(p) != o or abs(p["start"][1 - i] - eixo) > t / 2:
                novos.append(p)
                continue
            pa, pb = sorted((p["start"][i], p["end"][i]))
            if b <= pa or a >= pb:
                novos.append(p)
                continue
            if pa < a:
                novos.append(_seg(o, p["start"][1 - i], pa, a, p["thickness"]))
            if b < pb:
                novos.append(_seg(o, p["start"][1 - i], b, pb, p["thickness"]))
        saida = novos
    return renumerar(saida)


# --------------------------------------------------------------------------------------------
# 4. Avaliação: comparar rascunho x correção, escala, conferência visual
# --------------------------------------------------------------------------------------------


def comparar_rascunho(rascunho, correcao, t, tol_frag=None):
    """Problema: saber o que o rascunho automático errou e quanto (para calibrar o gerador).
    Classifica cada parede do rascunho contra a correção: 'fragmento' (eixo a <= tol_frag da parede
    correta, padrão t/2), 'fragmento com eixo deslocado' (até 1*t: pegou a face errada, p.ex. linha
    de cota), 'inexistente' (longe de qualquer parede). Conta também espessura fora de ±30% e
    quantos fragmentos por parede final."""
    tol_frag = t / 2 if tol_frag is None else tol_frag
    res = {"fragmento": 0, "fragmento com eixo deslocado": 0, "inexistente": 0,
           "espessura errada (>30%)": 0, "diagonais": 0, "por_parede_final": Counter(), "detalhe": []}
    for w in rascunho["walls"]:
        m = ((w["start"][0] + w["end"][0]) / 2, (w["start"][1] + w["end"][1]) / 2)
        best = min(correcao["walls"], key=lambda c: dist_ponto_segmento(m, c["start"], c["end"]))
        d = dist_ponto_segmento(m, best["start"], best["end"])
        if d <= tol_frag:
            cat = "fragmento"
        elif d <= t:
            cat = "fragmento com eixo deslocado"
        else:
            cat = "inexistente"
        res[cat] += 1
        if cat != "inexistente":
            res["por_parede_final"][best["id"]] += 1
            if abs(w["thickness"] - best["thickness"]) > 0.3 * best["thickness"]:
                res["espessura errada (>30%)"] += 1
        if orientacao(w) == "d":
            res["diagonais"] += 1
        res["detalhe"].append((w["id"], cat, round(comprimento(w)), w["thickness"], best["id"], round(d, 1)))
    res["por_parede_final"] = dict(res["por_parede_final"])
    return res


def escala_px_por_m(px, metros):
    """Problema: escala desconhecida. Dada uma cota legível (ex.: largura total 5,00 m) e a
    distância em px entre as faces correspondentes, devolve px/m. Daí: parede de 15 cm = 0.15*escala."""
    return px / metros


def conferir(correcao, png, saida="conferencia.png"):
    """Problema: ver o resultado por cima da planta (mesmo desenho do conferir.py do pacote)."""
    img = Image.open(png).convert("RGB")
    img = Image.blend(img, Image.new("RGB", img.size, "white"), 0.35)
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 500))
    for w in correcao.get("walls", []):
        d.line([tuple(w["start"]), tuple(w["end"])], fill=(200, 30, 30), width=lw)
    for o in correcao.get("openings", []):
        col = (20, 140, 70) if o.get("type") == "door" else (40, 90, 220)
        d.line([tuple(o["start"]), tuple(o["end"])], fill=col, width=lw * 3)
    for x in correcao.get("walls", []) + correcao.get("openings", []):
        d.text(((x["start"][0] + x["end"][0]) / 2 + 4, (x["start"][1] + x["end"][1]) / 2 + 4), str(x.get("id", "")), fill=(0, 0, 0))
    img.save(saida)
    return saida


def recortar_ampliado(png, box, fator=3, saida="recorte.png"):
    """Problema: a imagem é reduzida para o modelo ver; detalhes (marco, caixilho, canto) somem.
    Recorta `box` = (x0, y0, x1, y1) em pixels originais e amplia `fator` vezes sem suavizar."""
    im = Image.open(png).convert("RGB").crop(box)
    im = im.resize((im.width * fator, im.height * fator), Image.NEAREST)
    im.save(saida)
    return saida



# --------------------------------------------------------------------------------------------
# 6. Edição manual, gravação e lote
# --------------------------------------------------------------------------------------------


def aplicar_edicoes(plan, edicoes, t=None):
    """Problema: o automático acerta 80-90 % e o resto é mais rápido descrever do que redesenhar.
    `edicoes` = {"remover": [ids], "alterar": {"w3": {"end": [x, y]}}, "adicionar_paredes": [...],
    "adicionar_aberturas": [...]}. Depois renumera, redivide nos encontros (inclusive nos novos),
    descarta degenerados e refaz o wall_id das aberturas pela geometria."""
    t = t or float(np.median([w["thickness"] for w in plan["walls"]]))
    rem = set(edicoes.get("remover", []))
    plan["walls"] = [w for w in plan["walls"] if w["id"] not in rem]
    plan["openings"] = [o for o in plan["openings"] if o["id"] not in rem]
    for oid, mud in edicoes.get("alterar", {}).items():
        for obj in plan["walls"] + plan["openings"]:
            if obj["id"] == oid:
                obj.update(mud)
    for w in edicoes.get("adicionar_paredes", []):
        w.setdefault("thickness", t)
        plan["walls"].append(w)
    for o in edicoes.get("adicionar_aberturas", []):
        plan["openings"].append(o)
    renumerar(plan["walls"])
    plan["walls"] = dividir_nos_encontros(plan["walls"], t, aberturas=plan["openings"])
    plan["walls"] = remover_degeneradas(plan["walls"], t)
    for k, o in enumerate(plan["openings"], 1):
        o["id"] = f"o{k}"
    atribuir_wall_id(plan["openings"], plan["walls"], t)
    arredondar(plan["walls"])
    arredondar(plan["openings"])
    return plan


def validar(plan, original):
    """Problema: entregar JSON que o editor rejeita. Confere plan_id/image iguais ao planta.json,
    coordenadas dentro da imagem, ids únicos e wall_id existente. Devolve lista de problemas."""
    erros = []
    if plan.get("plan_id") != original.get("plan_id"):
        erros.append("plan_id diferente")
    if plan.get("image") != original.get("image"):
        erros.append("image diferente")
    W, H = original["image"]["width"], original["image"]["height"]
    for x in plan["walls"] + plan["openings"]:
        for px, py in (x["start"], x["end"]):
            if not (0 <= px <= W and 0 <= py <= H):
                erros.append(f"{x.get('id')} fora da imagem")
    if len({w["id"] for w in plan["walls"]}) != len(plan["walls"]):
        erros.append("ids de parede repetidos")
    if len({o["id"] for o in plan["openings"]}) != len(plan["openings"]):
        erros.append("ids de abertura repetidos")
    wids = {w["id"] for w in plan["walls"]}
    for o in plan["openings"]:
        if o.get("wall_id") and o["wall_id"] not in wids:
            erros.append(f"{o['id']} aponta para parede inexistente")
    return erros


def finalizar(pasta, notas, destino="correcoes"):
    """Problema: gravar correcoes/<plan_id>.json no formato do LEIA-ME e anotar o progresso.
    Usa <pasta>/final.json se existir (edição manual), senão <pasta>/auto.json."""
    import os
    orig = json.load(open(os.path.join(pasta, "planta.json"), encoding="utf-8"))
    src = os.path.join(pasta, "final.json")
    if not os.path.exists(src):
        src = os.path.join(pasta, "auto.json")
    d = json.load(open(src, encoding="utf-8"))
    out = {"plan_id": orig["plan_id"], "image": orig["image"], "walls": d["walls"],
           "openings": d["openings"], "notas": notas}
    erros = validar(out, orig)
    if erros:
        raise SystemExit("não gravado: " + "; ".join(erros))
    os.makedirs(destino, exist_ok=True)
    json.dump(out, open(os.path.join(destino, f"{orig['plan_id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(destino, "progresso.txt"), "a", encoding="utf-8") as fp:
        fp.write(f"{orig['plan_id']} ok\n")
    return out


def consolidar_lote(destino="correcoes", saida="correcoes.json"):
    """Problema: o editor importa um único correcoes.json com uma chave por plan_id."""
    import glob
    out = {}
    for p in sorted(glob.glob(f"{destino}/*.json")):
        d = json.load(open(p, encoding="utf-8"))
        out[d["plan_id"]] = d
    json.dump(out, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def estatisticas_rascunho(original, correcao):
    """Problema: medir, por planta, o que o rascunho automático errou (para o relatório e para
    calibrar o gerador). Junta comparar_rascunho com contagens de trechos curtos/finos/grossos e a
    classificação das aberturas do rascunho (aproveitável, tipo errado, lixo, inexistente, faltando)."""
    t = float(np.median([w["thickness"] for w in correcao["walls"]]))
    r = comparar_rascunho(original, correcao, t)
    r.pop("detalhe", None)
    r["curtos (<1.3t)"] = sum(1 for w in original["walls"] if comprimento(w) < 1.3 * t)
    r["finos (<0.5t)"] = sum(1 for w in original["walls"] if w["thickness"] < 0.5 * t)
    r["grossos (>1.6t)"] = sum(1 for w in original["walls"] if w["thickness"] > 1.6 * t)
    ab = {"aproveitavel": 0, "tipo_errado": 0, "lixo": 0, "inexistente": 0, "faltando": 0}
    for a in original["openings"]:
        m = ((a["start"][0] + a["end"][0]) / 2, (a["start"][1] + a["end"][1]) / 2)
        cand = [x for x in correcao["openings"] if dist_ponto_segmento(m, x["start"], x["end"]) <= t]
        if comprimento(a) < 1.2 * t or orientacao(a) == "d":
            ab["lixo"] += 1
        elif not cand:
            ab["inexistente"] += 1
        elif any(x["type"] == a["type"] for x in cand):
            ab["aproveitavel"] += 1
        else:
            ab["tipo_errado"] += 1
    for x in correcao["openings"]:
        m = ((x["start"][0] + x["end"][0]) / 2, (x["start"][1] + x["end"][1]) / 2)
        if not any(dist_ponto_segmento(m, a["start"], a["end"]) <= t and comprimento(a) >= 1.2 * t for a in original["openings"]):
            ab["faltando"] += 1
    r["aberturas"] = ab
    r["paredes rascunho/correcao"] = (len(original["walls"]), len(correcao["walls"]))
    r["aberturas rascunho/correcao"] = (len(original["openings"]), len(correcao["openings"]))
    return r


# --------------------------------------------------------------------------------------------
# 5. Pipeline completo (automático) e linha de comando
# --------------------------------------------------------------------------------------------


class DesenhoNaoSuportado(ValueError):
    """The drawing has no walls as pairs of parallel lines, so this detector does not apply."""


def pipeline_automatico(png, t=None, limiar=128, plan_id=None):
    """Encadeia tudo: espessura -> paredes por pares de faces -> mescla/filtra -> aberturas por cor
    -> corta nos vãos -> liga cantos -> divide nos encontros (L/T/cruz). É um rascunho melhor, não dispensa a revisão visual."""
    rgb, gray = carregar(png)
    mask = mascara_escura(gray, limiar, rgb)
    t_est, hist = estimar_espessura(mask)
    if t is None and t_est is None:
        raise DesenhoNaoSuportado(
            "não há paredes desenhadas como duas linhas paralelas nesta planta (paredes preenchidas, "
            "cópia colorida ou desbotada); o detector de faces não se aplica a este estilo, use o rascunho da máquina")
    ts = espessuras_candidatas(hist, t_est) if t is None else [t]
    t = ts[0]
    paredes = detectar_paredes(mask, ts)
    paredes = mesclar_colineares(paredes, t)
    paredes, _ = filtrar_paredes(paredes, t)
    paredes, _ = remover_isoladas(paredes, t)
    paredes = normalizar_espessura(paredes, ts)
    paredes = ligar_cantos(paredes, t)
    aberturas = medir_aberturas(rgb, paredes, t)
    paredes = cortar_nos_vaos(paredes, aberturas, t)
    paredes = remover_paredes_em_vaos(paredes, aberturas, t)
    aberturas = remover_aberturas_orfas(aberturas, paredes, t)
    paredes = ligar_cantos(paredes, t)
    paredes = dividir_nos_encontros(paredes, t, aberturas=aberturas)
    paredes = remover_degeneradas(paredes, t)
    atribuir_wall_id(aberturas, paredes, t)
    arredondar(paredes)
    arredondar(aberturas)
    h, w = gray.shape
    return {"plan_id": plan_id, "image": {"width": int(w), "height": int(h)},
            "walls": paredes, "openings": aberturas, "notas": f"pipeline automático, t={t} px (espessuras aceitas: {ts})"}


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return
    cmd, args = argv[1], argv[2:]
    if cmd == "perfil":
        rgb, gray = carregar(args[0])
        ll = linhas_longas(mascara_escura(gray, rgb=rgb), min_frac=float(args[1]) if len(args) > 1 else 0.15)
        print("rows:", centros_de_linhas(ll["rows"]))
        print("cols:", centros_de_linhas(ll["cols"]))
    elif cmd == "espessura":
        rgb, gray = carregar(args[0])
        t, hist = estimar_espessura(mascara_escura(gray, rgb=rgb))
        print("espessura estimada:", t, "px; histograma:", dict(hist))
    elif cmd == "paredes":
        rgb, gray = carregar(args[0])
        mask = mascara_escura(gray, rgb=rgb)
        t = float(args[1]) if len(args) > 1 else estimar_espessura(mask)[0]
        ps = ligar_cantos(normalizar_espessura(remover_isoladas(filtrar_paredes(mesclar_colineares(detectar_paredes(mask, t), t), t)[0], t)[0], t), t)
        print(json.dumps({"walls": arredondar(ps)}, ensure_ascii=False))
    elif cmd == "aberturas":
        rgb, gray = carregar(args[0])
        plan = json.load(open(args[1], encoding="utf-8"))
        t = float(args[2]) if len(args) > 2 else estimar_espessura(mascara_escura(gray, rgb=rgb))[0]
        print(json.dumps({"openings": arredondar(medir_aberturas(rgb, plan["walls"], t))}, ensure_ascii=False))
    elif cmd == "limpar":
        plan = json.load(open(args[0], encoding="utf-8"))
        t = float(args[1]) if len(args) > 1 else float(np.median([w["thickness"] for w in plan["walls"]]))
        ps = ligar_cantos(normalizar_espessura(remover_isoladas(filtrar_paredes(mesclar_colineares(plan["walls"], t), t)[0], t)[0], t), t)
        plan["walls"] = arredondar(ps)
        print(json.dumps(plan, ensure_ascii=False, indent=1))
    elif cmd == "auto":
        plan = pipeline_automatico(args[0], float(args[1]) if len(args) > 1 else None)
        print(json.dumps(plan, ensure_ascii=False, indent=1))
    elif cmd == "comparar":
        r = json.load(open(args[0], encoding="utf-8"))
        c = json.load(open(args[1], encoding="utf-8"))
        t = float(np.median([w["thickness"] for w in c["walls"]]))
        res = comparar_rascunho(r, c, t)
        det = res.pop("detalhe")
        print(json.dumps(res, ensure_ascii=False, indent=1))
        for linha in det:
            if linha[1] != "fragmento":
                print(*linha)
    elif cmd == "conferir":
        print(conferir(json.load(open(args[0], encoding="utf-8")), args[1] if len(args) > 1 else "planta.png"))
    elif cmd == "editar":
        plan = json.load(open(args[0], encoding="utf-8"))
        t = None
        if "t=" in plan.get("notas", ""):
            t = float(plan["notas"].split("t=")[1].split()[0])
        plan = aplicar_edicoes(plan, json.loads(args[2]), t)
        json.dump(plan, open(args[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(args[1], len(plan["walls"]), "paredes", len(plan["openings"]), "aberturas")
    elif cmd == "finalizar":
        out = finalizar(args[0], args[1] if len(args) > 1 else "")
        print(out["plan_id"], len(out["walls"]), "paredes", len(out["openings"]), "aberturas")
    elif cmd == "lote":
        out = consolidar_lote()
        print({k: (len(v["walls"]), len(v["openings"])) for k, v in out.items()})
    elif cmd == "estatisticas":
        o = json.load(open(args[0], encoding="utf-8"))
        c = json.load(open(args[1], encoding="utf-8"))
        print(json.dumps(estatisticas_rascunho(o, c), ensure_ascii=False, indent=1))
    else:
        print("comando desconhecido:", cmd)
        print(__doc__)


if __name__ == "__main__":
    _main(sys.argv)
