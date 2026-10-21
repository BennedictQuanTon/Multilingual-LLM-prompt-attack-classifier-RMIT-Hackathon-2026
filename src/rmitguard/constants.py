"""Competition facts. Everything here comes from the competition pages (fetched 2026-10-21)."""

COMPETITION = "rmit-hackathon-2026"
DEADLINE_UTC = "2026-10-22T11:00:00Z"  # 18:00 Vietnam time
MAX_DAILY_SUBMISSIONS = 6

LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
L2I = {label: i for i, label in enumerate(LABELS)}
LANGUAGES = ("pt", "fr", "it", "en")
N_TEST_ROWS = 6000

# The ONLY backbones the rules allow. key -> (Hugging Face id, Kaggle model handle, kind)
# Kaggle handles are attached to a notebook via `model_sources` and mounted under /kaggle/input.
APPROVED_MODELS = {
    "distilbert-mbert": (
        "distilbert/distilbert-base-multilingual-cased",
        "aisuko/distilbert-base-multilingual-cased/Transformers/default/1",
        "encoder",
    ),
    "mmbert-small": ("jhu-clsp/mmBERT-small", "aisuko/mmbert-small/Transformers/default/1", "encoder"),
    "mmbert-base": ("jhu-clsp/mmBERT-base", "aisuko/mmbert-base/Transformers/default/1", "encoder"),
    "xlmr-base": ("FacebookAI/xlm-roberta-base", "aisuko/xlm-roberta-base/Transformers/default/1", "encoder"),
    "xlmr-large": (
        "FacebookAI/xlm-roberta-large",
        "aisuko/xlm-roberta-large/Transformers/default/1",
        "encoder",
    ),
    "rembert": ("google/rembert", "aisuko/rembert/Transformers/default/1", "encoder"),
    "deberta-pi-v2": (
        "protectai/deberta-v3-base-prompt-injection-v2",
        "aisuko/deberta-v3-base-prompt-injection-v2/Transformers/default/1",
        "encoder",
    ),
    "wolf-defender": (
        "patronus-studio/wolf-defender-prompt-injection",
        "aisuko/wolf-defender-prompt-injection/Transformers/default/1",
        "encoder",
    ),
    "granite-2b": (
        "ibm-granite/granite-3.3-2b-instruct",
        "aisuko/granite-3.3-2b-instruct/Transformers/default/1",
        "llm",
    ),
    "mistral-7b": (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "aisuko/mistral-7b-instruct-v0.3/Transformers/default/1",
        "llm",
    ),
}
