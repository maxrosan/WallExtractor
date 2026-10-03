"""LoRA fine-tuning of Qwen3-VL on plan image -> walls/doors/windows JSON (see ``wallextractor.vlm_data``).

The vision tower and merger stay frozen; LoRA goes on the language model's attention and MLP projections,
as in the public floor-plan VLM projects. Loss only on the answer tokens. One image per step (sizes
vary), gradient accumulation for the effective batch. Metrics go to ``<out>/metrics.jsonl``; the adapter
of every epoch goes to ``<out>/epoch<N>`` and the one with the lowest validation loss to ``<out>/best``.

Usage (GPU, 16 GB is enough for the 4B in bf16 with gradient checkpointing):
  python -m wallextractor.train_vlm --data data/vlm --out runs/vlm4b --model Qwen/Qwen3-VL-4B-Instruct \
      --epochs 2 --grad-accum 8 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List

import torch
from PIL import Image

from .vlm_data import PROMPT

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_rows(path: str, limit: int = 0, unique: bool = False) -> List[Dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if unique:
        seen, out = set(), []
        for r in rows:
            if r["image"] not in seen:
                seen.add(r["image"])
                out.append(r)
        rows = out
    return rows[:limit] if limit else rows


def messages(target: str = None) -> List[Dict]:
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if target is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return m


class Encoder:
    """Row -> model inputs with labels on the answer only."""

    def __init__(self, processor, data_dir: str, max_len: int):
        self.p = processor
        self.dir = data_dir
        self.max_len = max_len
        tok = processor.tokenizer
        self.header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, row: Dict):
        img = Image.open(os.path.join(self.dir, row["image"])).convert("RGB")
        text = self.p.apply_chat_template(messages(row["target"]), tokenize=False)
        enc = self.p(text=[text], images=[img], return_tensors="pt")
        ids = enc["input_ids"][0]
        labels = torch.full_like(ids, -100)
        h, n = self.header, len(self.header)
        start = None
        lst = ids.tolist()
        for i in range(len(lst) - n, -1, -1):
            if lst[i:i + n] == h:
                start = i + n
                break
        if start is None:
            raise ValueError("assistant header not found")
        labels[start:] = ids[start:]
        if len(lst) > self.max_len:
            return None  # an image + answer that long would be truncated mid-answer: skip it
        enc["labels"] = labels[None]
        return enc


def build_model(name: str, r: int, alpha: int, dropout: float, grad_ckpt: bool = True):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    for p in model.parameters():
        p.requires_grad_(False)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model, processor


@torch.no_grad()
def val_loss(model, enc_fn, rows, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for r in rows:
        b = enc_fn(r)
        if b is None:
            continue
        b = {k: v.to(device) for k, v in b.items()}
        tot += float(model(**b).loss)
        n += 1
    model.train()
    return tot / max(1, n)


def train(a) -> Dict:
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    model, processor = build_model(a.model, a.lora_r, a.lora_alpha, a.lora_dropout)
    model.to(device)
    model.print_trainable_parameters()
    enc = Encoder(processor, a.data, a.max_len)
    train_rows = load_rows(os.path.join(a.data, "train.jsonl"), a.limit_train)
    val_rows = load_rows(os.path.join(a.data, "val.jsonl"), a.limit_val, unique=True)
    print(f"[vlm] train rows {len(train_rows)}, val plans {len(val_rows)}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train_rows) * a.epochs / a.grad_accum)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    t0 = time.time()
    best = float("inf")
    step = skipped = 0
    stop = False
    model.train()
    mf = open(os.path.join(a.out, "metrics.jsonl"), "a", encoding="utf-8")
    for epoch in range(1, a.epochs + 1):
        rows = train_rows[:]
        random.shuffle(rows)
        running, seen = 0.0, 0
        for i, r in enumerate(rows):
            b = enc(r)
            if b is None:
                skipped += 1
                continue
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss / a.grad_accum
            loss.backward()
            running += float(loss) * a.grad_accum
            seen += 1
            if seen % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 2 ** 30
                    print(f"[vlm] epoch {epoch} step {step}/{steps} loss {running / seen:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} mem {mem:.1f}G {time.time() - t0:.0f}s", flush=True)
            if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
                print("[vlm] time limit reached", flush=True)
                stop = True
                break
        vl = val_loss(model, enc, val_rows, device)
        rec = {"epoch": epoch, "train_loss": running / max(1, seen), "val_loss": vl, "steps": step,
               "skipped_long": skipped, "elapsed_s": round(time.time() - t0)}
        print(f"[vlm] {json.dumps(rec)}", flush=True)
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        model.save_pretrained(os.path.join(a.out, f"epoch{epoch}"))
        if vl < best:
            best = vl
            model.save_pretrained(os.path.join(a.out, "best"))
            processor.save_pretrained(os.path.join(a.out, "best"))
        if stop:
            break
    summary = {"best_val_loss": best, "args": vars(a), "elapsed_s": round(time.time() - t0)}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder written by wallextractor.vlm_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
