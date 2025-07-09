"""
Adapted from https://github.com/automateyournetwork/fine_tune_example

RAG + Fine Tuning = RAFT
Using llama3 and multiomics data

"""

import gc
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from peft.peft_model import PeftConfig, PeftModel, PeftType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, ORPOTrainer, SFTConfig, SFTTrainer, setup_chat_format

from ..conf import env, log
from .strategies import TrainingStrategy

attn_implementation = env.training.attention

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # only works with GPU
    bnb_4bit_quant_type=env.model.quantization_type,
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)


def main(
    training_path: Path,
    base_model: str = env.model.name,
    new_model: str = env.model.output_name,
    strategy: TrainingStrategy = TrainingStrategy.ORPO,
    use_bnb: bool = False
):
    dataset = prepare_dataset(training_path)

    # Load tokenizer
    log.info(f"loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load model
    log.info(f"loading model: {base_model}")

    model_kwargs = {
        "device_map": "auto",
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True
    }

    if use_bnb:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **model_kwargs
    )
    model, tokenizer = setup_chat_format(model, tokenizer)

    if use_bnb:
        model = prepare_model_for_kbit_training(model)

    log.info(f"configuring trainer: {strategy}")
    _, trainer = _configure_trainer(strategy, model, tokenizer, dataset)
    log.info("training model")
    trainer.train()
    log.info(f"saving model {new_model}")
    trainer.save_model(new_model)

    # Flush memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def _configure_trainer(strategy, model, tokenizer, dataset):
    # QLoRA config

    match strategy:
        case TrainingStrategy.SFT:
            args = SFTConfig(
                learning_rate=1e-4,
                lr_scheduler_type="linear",
                per_device_train_batch_size=env.training.batch_size,
                per_device_eval_batch_size=env.training.batch_size,
                max_seq_length=env.training.max_seq_len,
                gradient_accumulation_steps=4,
                optim="paged_adamw_8bit",
                num_train_epochs=3,
                eval_strategy="steps",
                eval_steps=100,
                logging_steps=100,
                warmup_steps=10,
                output_dir="./results/",
            )
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                peft_config=peft_config,
                tokenizer=tokenizer,
            )
        case TrainingStrategy.OPRO:
            args = ORPOConfig(
                learning_rate=1e-4,
                beta=0.1,
                lr_scheduler_type="linear",
                max_length=1024,
                max_prompt_length=512,
                per_device_train_batch_size=env.training.batch_size,
                per_device_eval_batch_size=env.training.batch_size,
                gradient_accumulation_steps=4,
                optim="paged_adamw_8bit",
                num_train_epochs=3,
                eval_strategy="steps",
                eval_steps=100,
                logging_steps=100,
                warmup_steps=10,
                # report_to="wandb",
                output_dir="./results/",
            )
            trainer = ORPOTrainer(
                model=model,
                args=args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                peft_config=peft_config,
                tokenizer=tokenizer,
            )
        case _:
            raise NotImplementedError(f"unknown trainer: {strategy}")
    return args, trainer


def prepare_dataset(training_path):
    log.info(f"loading dataset {training_path}")

    def format_chat_template(row):
        role = "You are an expert on multiomics and pathogen metobolic pathways"
        row["chosen"] = f'{role} {row["chosen"]}'
        row["rejected"] = f'{role} {row["rejected"]}'
        row["role"] = role
        return row

    def format_instruction(row):
        row["prompt"] = row["instruction"]
        row["completion"] = row["cot_answer"]
        return row

    match training_path.suffix:
        case ".jsonl":
            dataset = load_dataset(
                "json",
                data_files={"train": str(training_path)},
                split="all",
            )

            log.info("mapping dataset")
            dataset = dataset.map(
                format_chat_template,
                num_proc=os.cpu_count() // 2,
                batched=False,
            )
        case _:
            from datasets import load_from_disk

            dataset = load_from_disk(training_path)

            log.info("mapping dataset")
            dataset = dataset.rename_column("instruction", "prompt")
            dataset = dataset.rename_column("cot_answer", "completion")

    log.info("shuffling dataset")
    dataset = dataset.shuffle(seed=42)

    log.info("splitting dataset")
    dataset = dataset.train_test_split(test_size=0.01)
    return dataset


def load(base_model, adapter_path):
    """Load and merge the adapter without saving."""
    # Reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=env.model.device_map,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model, tokenizer = setup_chat_format(model, tokenizer)

    if adapter_path is not None:
        adapter = PeftModel.from_pretrained(model, adapter_path, config=peft_config)
        log.info("merge and unload model")
        model = adapter.merge_and_unload()
    else:
        log.info("loading base model only")
    return tokenizer, model
