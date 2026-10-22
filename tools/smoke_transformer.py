"""CPU smoke test of the full transformer path (tokenise -> weighted Trainer -> predict -> artifacts).

Builds a 2-layer XLM-R with the real XLM-R tokenizer, so it exercises exactly the code that runs on Kaggle
in about a minute, without a GPU. Used by CI; run locally with:
    uv run --with torch --with transformers --with datasets --with accelerate python tools/smoke_transformer.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification  # noqa: E402

import rmitguard.models.transformer as T  # noqa: E402
from conftest import synth  # noqa: E402
from rmitguard.config import ExperimentConfig  # noqa: E402
from rmitguard.runner import run_experiment  # noqa: E402


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="smoke_"))
    synth(300, 0, True, "r").to_csv(work / "train.csv", index=False)
    synth(80, 1, False, "t").to_csv(work / "test.csv", index=False)

    tiny = work / "tiny-xlmr"
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    cfg = XLMRobertaConfig(
        vocab_size=tok.vocab_size + 2,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=130,
        num_labels=2,
    )
    XLMRobertaForSequenceClassification(cfg).save_pretrained(tiny)  # 2-class head: checks the 3-class re-init
    tok.save_pretrained(tiny)
    T.resolve_model_path = lambda _cfg: str(tiny)

    for prepend in (False, True):
        exp = ExperimentConfig(
            name=f"smoke-lang{int(prepend)}",
            model="xlmr-base",
            n_folds=3,
            folds=[0, 1],
            epochs=1,
            batch_size=8,
            max_len=48,
            head_len=32,
            fp16=False,
            prepend_language=prepend,
            class_weight="balanced",
            n_gpu=1 if prepend else None,
            freeze_embeddings=prepend,
        )
        run = run_experiment(exp, str(work), work / "artifacts")
        meta = json.loads((run / "meta.json").read_text())
        assert meta["folds_done"] == [0, 1], meta["folds_done"]
        assert (run / "submission.csv").exists()
        print(
            f"OK {exp.name}: folds {meta['folds_done']}, macro F1 tuned {meta['scores']['macro_f1_tuned']:.3f}"
        )


if __name__ == "__main__":
    main()
