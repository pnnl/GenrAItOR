import gc
from pathlib import Path

import pandas as pd
import torch
from alignscore import AlignScore

from ..conf import env, log


def evaluate(tokenizer, model, data, batch_size=15) -> pd.DataFrame:
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    device = model.device
    log.info(f"eval {len(data)} records")
    results = []
    answers = []
    with torch.no_grad():
        for chunk in batch_dataframe(data, batch_size):

            claims = [i[: env.eval.max_tokens] for i in chunk["claim"].to_list()]
            contexts = [i[: env.eval.max_tokens] for i in chunk["context"].to_list()]

            inputs = tokenizer(contexts, return_tensors="pt", padding=True)
            ids = inputs["input_ids"].to(device)
            del inputs
            log.debug(f"{len(ids)} inputs for inference")
            outputs = model.generate(
                input_ids=ids,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
            )
            del ids
            out = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )
            answers.extend(out)
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    data["y_pred"] = answers

    scorer = AlignScore(
        model="roberta-base",
        batch_size=32,
        device="cuda",
        ckpt_path=str(Path(env.paths.app) / "data/alignscore/AlignScore-base.ckpt"),
        evaluation_mode="nli_sp",
    )
    pred_claims = data["y_pred"]
    contexts = data["context"].to_list()
    scores = scorer.score(contexts=contexts, claims=pred_claims)
    pred_scores = pd.DataFrame(scores, columns=["align_score"])
    pred_scores["dataset"] = "pred"
    pred_scores["context"] = contexts
    pred_scores["claim"] = pred_claims
    results.append(pred_scores)

    claims = data["claim"].to_list()
    contexts = data["context"].to_list()
    scores = scorer.score(contexts=contexts, claims=claims)
    eval_scores = pd.DataFrame(scores, columns=["align_score"])
    eval_scores["context"] = contexts
    eval_scores["claim"] = claims
    eval_scores["dataset"] = "eval"
    results.append(eval_scores)

    return pd.concat(results)


def load(raft_path: Path):
    import duckdb

    match raft_path.suffix:
        case ".jsonl":
            with duckdb.connect(":memory:") as conn:
                data = conn.sql(
                    f"""
                    SELECT
                        context || instruction || question as context
                        ,cot_answer as claim
                    FROM read_json("{raft_path}")
                """
                ).to_df()
        case _:
            from datasets import load_from_disk

            data = pd.DataFrame(load_from_disk(raft_path).to_dict())
            with duckdb.connect(":memory:") as conn:
                data = conn.sql(
                    """
                    SELECT
                        instruction as context
                        ,cot_answer as claim
                    FROM data
                """
                ).to_df()
                
    return data


def batch_dataframe(df, n):
    """
    Splits a dataframe into chunks of size n.

    :param df: The input dataframe
    :param n: The size of each chunk
    :return: A list of dataframes, each of size n (last one might be smaller)
    """
    # Determine the number of chunks
    num_chunks = len(df) // n + (1 if len(df) % n != 0 else 0)

    # Split the dataframe into chunks
    chunks = [df.iloc[i * n : (i + 1) * n] for i in range(num_chunks)]

    return chunks
