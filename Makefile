# Team GenData — RMIT Hackathon 2026. `make help` lists everything.
SHELL := /bin/bash
COMP := rmit-hackathon-2026
RUN  := uv run
CONFIG ?= configs/xlmr-large.yaml
FOLDS ?=
KERNEL ?=
VERSION ?=
MSG ?=
SOURCES ?=

.DEFAULT_GOAL := help
.PHONY: help setup check lint fmt test smoke data eda baseline build build-final push push-ensemble status output ensemble validate submit-dry submit

help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

# ----------------------------------------------------------------- dev
setup: ## install env + git hooks
	uv sync --extra dev
	$(RUN) pre-commit install

check: lint test ## everything CI runs (except the transformer smoke test)

lint: ## ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt: ## auto-fix lint + format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test: ## unit + end-to-end tests on synthetic data
	$(RUN) pytest

smoke: ## real HF Trainer path with a tiny XLM-R on CPU (~1 min)
	uv run --with torch --with "transformers>=4.48" --with datasets --with accelerate --with sentencepiece \
		python tools/smoke_transformer.py

# ----------------------------------------------------------------- data + local runs
data: ## download competition data to data/ (join the competition on Kaggle first)
	mkdir -p data
	$(RUN) kaggle competitions download $(COMP) -p data
	cd data && unzip -o -q $(COMP).zip && rm -f $(COMP).zip
	@ls -la data

eda: ## reports/eda.md
	$(RUN) rmitguard eda

baseline: ## TF-IDF 5-fold CV on CPU: sanity check + CV floor (never submitted)
	$(RUN) rmitguard train configs/tfidf.yaml

# ----------------------------------------------------------------- Kaggle
build: ## build notebook: make build CONFIG=configs/xlmr-large.yaml [FOLDS=0,1]
	$(RUN) python tools/build_kernel.py train $(CONFIG) $(if $(FOLDS),--folds $(FOLDS))

build-final: ## build the official RMIT_SGS_GenData_C1 notebook from CONFIG
	$(RUN) python tools/build_kernel.py train $(CONFIG) --final

push: ## build + push + start on Kaggle (T4 x2): make push CONFIG=... [FOLDS=0]
	@dir=$$($(RUN) python tools/build_kernel.py train $(CONFIG) $(if $(FOLDS),--folds $(FOLDS)) | awk '/^built/{print $$2}'); \
	echo "pushing $$dir"; $(RUN) kaggle kernels push -p "$$dir"

push-ensemble: ## merge teammates' runs on Kaggle: make push-ensemble SOURCES=user/a,user/b
	@dir=$$($(RUN) python tools/build_kernel.py ensemble --sources $(SOURCES) | awk '/^built/{print $$2}'); \
	$(RUN) kaggle kernels push -p "$$dir"

status: ## make status KERNEL=user/slug
	$(RUN) kaggle kernels status $(KERNEL)

output: ## download a kernel's output to outputs/<slug>: make output KERNEL=user/slug
	mkdir -p outputs/$(notdir $(KERNEL))
	$(RUN) kaggle kernels output $(KERNEL) -p outputs/$(notdir $(KERNEL))

ensemble: ## merge local run dirs: make ensemble SOURCES="outputs/a outputs/b"
	$(RUN) rmitguard ensemble $(SOURCES)

validate: ## make validate FILE=path/to/submission.csv
	$(RUN) rmitguard validate $(FILE)

submit-dry: ## dry run: make submit-dry KERNEL=user/slug [VERSION=n] MSG="..."
	$(RUN) python tools/submit.py --kernel $(KERNEL) $(if $(VERSION),--version $(VERSION)) -m "$(MSG)"

submit: ## spends 1 of 6 daily submissions: make submit KERNEL=user/slug VERSION=n MSG="..."
	$(RUN) python tools/submit.py --kernel $(KERNEL) $(if $(VERSION),--version $(VERSION)) -m "$(MSG)" --yes
