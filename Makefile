PROJECT="genraitor"
USER="jens829"
BASE_MODEL="meta-llama/Meta-Llama-3-8B"
PYTHON=".venv/bin/python3"
POOL?="dl_shared"

all:

.PHONY: paper
paper:
	docker run --rm -it \
		-e JOURNAL="joss" \
		-v ${PWD}:/data \
		-u $(id -u):$(id -g) \
		openjournals/inara \
		-o pdf,crossref \
		paper/paper.md

.PHONY: train
train:
	$(PYTHON) -m $(PROJECT) train:raft \
		--model_name "$(BASE_MODEL)"\
		--output_name data/finetuned

train-xl:
	$(PYTHON) -m $(PROJECT) train:raft \
		-n data/finetuned-xl \
		-m "$(BASE_MODEL)"\
		-t data/training/hf/xlarge.hf

evaluate:
	$(PYTHON) -m $(PROJECT) eval \
		--adapter_path data/finetuned \
		--base_model "$(BASE_MODEL)"\
		--raft_path "data/training/hf/large.hf" \
		--save_path "data/align_scores.parquet" \
		--batch_size 25 
	#--raft_path "data/training/raft_outputs/raw.jsonl" \

web:
	cd genraitor/web && npm run dev

.PHONY: batch-train-xl
batch-train-xl:
	sbatch examples/train-xl.sh

.PHONY: cluster-request
cluster-request:
	salloc -A $(PROJECT) -p $(POOL) --gres=gpu:1

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
