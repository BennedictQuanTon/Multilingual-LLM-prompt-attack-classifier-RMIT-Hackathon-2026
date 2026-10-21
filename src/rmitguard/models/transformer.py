"""Fine-tune one approved backbone as a 3-way sequence classifier.

- kind == "encoder": full fine-tuning, fp16, both T4s through DataParallel.
- kind == "llm": 4-bit NF4 base + LoRA on all linear layers + a classification head, one GPU
  (the same recipe as the Qwen2.5-3B classifier that was our best single model in 2025).
"""

from __future__ import annotations

import gc
import glob
import inspect
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from packaging.version import Version
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ..config import ExperimentConfig
from ..constants import L2I, LABELS
from ..metrics import macro_f1

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
ID2LABEL = dict(enumerate(LABELS))
DTYPE_KW = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
TRAINING_ARG_NAMES = set(inspect.signature(TrainingArguments.__init__).parameters)
# Batches of similar length = far less padding. The argument was renamed in transformers 5.
LENGTH_GROUPING = (
    {"train_sampling_strategy": "group_by_length"}
    if "train_sampling_strategy" in TRAINING_ARG_NAMES
    else {"group_by_length": True}
)


def resolve_model_path(cfg: ExperimentConfig) -> str:
    """Kaggle mounts approved models under /kaggle/input/...; locally fall back to the Hugging Face id."""
    slug = cfg.kaggle_model.split("/")[1]
    hits = [
        os.path.dirname(p)
        for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
        if f"/{slug}/" in p.lower()
    ]
    if hits:
        return sorted(hits, key=len)[0]
    if os.path.isdir("/kaggle/input"):
        print(f"⚠️ {cfg.kaggle_model} is not attached as an input; trying the Hub ({cfg.hf_id}).")
    return cfg.hf_id


# ---------------------------------------------------------------- tokenisation
def _special_token_wrap(tokenizer) -> tuple[list[int], list[int]]:
    probe = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(probe) + 1):
        if full[i : i + len(probe)] == probe:
            return full[:i], full[i + len(probe) :]
    raise ValueError(f"cannot locate special tokens of {tokenizer.__class__.__name__}: {full}")


def head_tail(ids: list[int], body_len: int, head_len: int) -> list[int]:
    if len(ids) <= body_len:
        return ids
    return ids[:head_len] + ids[len(ids) - (body_len - head_len) :]


def encode(tokenizer, df: pd.DataFrame, cfg: ExperimentConfig) -> list[list[int]]:
    texts = df["text"].tolist()
    if cfg.prepend_language:
        texts = [f"[{lang}] {t}" for lang, t in zip(df["language"], texts)]
    prefix, suffix = _special_token_wrap(tokenizer)
    body_len = cfg.max_len - len(prefix) - len(suffix)
    head_len = min(cfg.head_len, body_len)
    ids = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    return [prefix + head_tail(x, body_len, head_len) + suffix for x in ids]


def to_dataset(ids: list[list[int]], labels: np.ndarray | None = None) -> Dataset:
    data = {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}
    if labels is not None:
        data["labels"] = [int(v) for v in labels]
    return Dataset.from_dict(data)


# ---------------------------------------------------------------- models
class LastTokenClassifier(torch.nn.Module):
    """Classification head on the last non-pad token, for LLMs without a *ForSequenceClassification class."""

    is_parallelizable = True  # tells Trainer not to move / DataParallel-wrap the 4-bit model
    model_parallel = True

    def __init__(self, base, hidden_size: int, num_labels: int):
        super().__init__()
        self.base = base
        self.config = base.config
        self.score = torch.nn.Linear(hidden_size, num_labels, bias=False).to(base.device)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1  # right padding
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return SequenceClassifierOutput(logits=self.score(pooled.float()))


def load_encoder(path: str):
    config = AutoConfig.from_pretrained(path, num_labels=len(LABELS), id2label=ID2LABEL, label2id=L2I)
    if hasattr(config, "reference_compile"):  # ModernBERT / mmBERT: torch.compile breaks on T4 + DataParallel
        config.reference_compile = False
    # ignore_mismatched_sizes: the safety-tuned checkpoints ship a 2-class head; it is re-initialised for 3 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        path, config=config, ignore_mismatched_sizes=True
    )
    return model.float()


def load_llm(path: str, cfg: ExperimentConfig, tokenizer):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    common = {"quantization_config": bnb, DTYPE_KW: torch.float16, "device_map": {"": 0}}
    lora_kw = dict(
        r=cfg.lora["r"],
        lora_alpha=cfg.lora["alpha"],
        lora_dropout=cfg.lora["dropout"],
        target_modules="all-linear",
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=len(LABELS), id2label=ID2LABEL, label2id=L2I, **common
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", **lora_kw))
    except ValueError as exc:  # architecture has no *ForSequenceClassification
        print(f"No sequence-classification class ({exc.__class__.__name__}); using last-token head.")
        base = AutoModel.from_pretrained(path, **common)
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base = get_peft_model(base, LoraConfig(task_type="FEATURE_EXTRACTION", **lora_kw))
        model = LastTokenClassifier(base, base.config.hidden_size, len(LABELS))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M")
    return model


# ---------------------------------------------------------------- training
def class_weights(y: np.ndarray, mode: str) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    w = np.ones(len(LABELS)) if mode == "none" else len(y) / (len(LABELS) * np.maximum(counts, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        # our loss is a per-batch mean, so let Trainer divide by gradient_accumulation_steps
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = F.cross_entropy(
            logits.float(),
            labels,
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


def make_training_args(**kwargs) -> TrainingArguments:
    unsupported = sorted(k for k in kwargs if k not in TRAINING_ARG_NAMES)
    if unsupported:
        print(f"⚠️ transformers {transformers.__version__} does not support {unsupported}; ignoring them.")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in TRAINING_ARG_NAMES})


def _softmax(logits) -> np.ndarray:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return torch.softmax(torch.tensor(np.asarray(logits), dtype=torch.float32), dim=-1).numpy()


def predict_probs(trainer: Trainer, ids: list[list[int]]) -> np.ndarray:
    """Predict in length order (much less padding on a long-tailed length distribution), then restore order."""
    order = np.argsort([len(x) for x in ids], kind="stable")
    probs = _softmax(trainer.predict(to_dataset([ids[i] for i in order])).predictions)
    out = np.empty_like(probs)
    out[order] = probs
    return out


def train_fold(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    test_df: pd.DataFrame,
    seed: int,
):
    t0 = time.time()
    is_llm = cfg.kind == "llm"
    path = resolve_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_llm:
        tokenizer.padding_side = "right"

    tr_ids, va_ids, te_ids = (encode(tokenizer, df, cfg) for df in (train_df, val_df, test_df))
    n_tok = np.array([len(x) for x in tr_ids])
    truncated = float(np.mean(n_tok >= cfg.max_len))
    print(f"[{cfg.name}] model={path} | mean tokens {n_tok.mean():.0f} | truncated {truncated:.1%}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = load_llm(path, cfg, tokenizer) if is_llm else load_encoder(path)
    n_gpu = torch.cuda.device_count()
    devices = 1 if is_llm else max(n_gpu, 1)
    steps_per_epoch = math.ceil(len(tr_ids) / (cfg.batch_size * cfg.grad_accum * devices))
    out_dir = os.path.join(WORK_DIR, f"tmp_{cfg.name}_{seed}")
    args = make_training_args(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * (2 if is_llm else 4),
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        warmup_steps=math.ceil(cfg.warmup_ratio * steps_per_epoch * cfg.epochs),
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine" if is_llm else "linear",
        fp16=cfg.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.grad_ckpt and not is_llm,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.grad_ckpt and not is_llm else None,
        optim="paged_adamw_8bit" if is_llm else "adamw_torch",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=2,
        **LENGTH_GROUPING,
        label_names=["labels"],
    )
    if is_llm and args.n_gpu > 1:
        args._n_gpu = 1
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=to_dataset(tr_ids, y_train),
        eval_dataset=to_dataset(va_ids, y_val),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {"macro_f1": macro_f1(p.label_ids, _softmax(p.predictions).argmax(1))},
        class_weights=class_weights(y_train, cfg.class_weight),
        label_smoothing=cfg.label_smoothing,
    )
    trainer.train()
    val_probs, test_probs = predict_probs(trainer, va_ids), predict_probs(trainer, te_ids)
    info = dict(
        model_path=path,
        minutes=(time.time() - t0) / 60,
        mean_tokens=float(n_tok.mean()),
        truncated=truncated,
        steps=trainer.state.global_step,
        devices=devices,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        history=trainer.state.log_history,
    )

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return val_probs, test_probs, info
