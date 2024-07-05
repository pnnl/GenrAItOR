PROJECT="genraitor"
USER="jens829"
BASE_MODEL="meta-llama/Meta-Llama-3-8B"
PYTHON=".venv/bin/python3"

all:

.PHONY: tune
tune:
	$(PYTHON) -m $(PROJECT) raft:tune \
		--model_name "$(BASE_MODEL)"\
		--output_name data/finetuned

align:
	$(PYTHON) -m $(PROJECT) eval \
		--adapter_path data/finetuned \
		--base_model "$(BASE_MODEL)"\
		--raft_path "data/training/raft_outputs/raw.jsonl" \
		--save_path "data/align_scores.parquet"
web:
	cd genraitor/web && npm run dev

.PHONY: cluster-submit
cluster-submit:
	salloc -A $(PROJECT) -p dl_shared --gres=gpu:1

.PHONY: cluster-check
cluster-check:
	squeue -u $(USER)

ollama-serve:
	TMPDIR=/scratch TEMP=/scratch OLLAMA_DEBUG=1 OLLAMA_TMPDIR=/scratch OLLAMA_MODELS=/scratch/ollama \
	ollama serve

ollama-create:
	TMPDIR=/scratch TEMP=/scratch \
	ollama create genraitor -f ./Modelfile

ollama-download:
	wget https://github.com/ollama/ollama/releases/download/v0.3.8/ollama-linux-amd64.tgz

ollama-extract:
	tar -xvzf ollama-linux-amd64.tgz -C ./ollama-linux-amd64

ollama-install:
	cp ./ollama-linux-amd64/bin/ollama ~/.local/bin/ollama
align-checkpoint:
	wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt > data/alignscore/AlignScore-base.ckpt && \
	mv AlignScore-base.ckpt data/alignscore/
