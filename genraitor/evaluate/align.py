from pathlib import Path

import pandas as pd
from alignscore import AlignScore

from ..conf import env, log


def evaluate(tokenizer, model, data, batch_size = 15) -> pd.DataFrame:

    tokenizer.pad_token = tokenizer.eos_token
    device = model.device
    scorer = AlignScore(
        model="roberta-base",
        batch_size=32,
        device=device,
        ckpt_path=str(Path(env.paths.app) / "data/alignscore/AlignScore-base.ckpt"),
        evaluation_mode="nli_sp",
    )
    log.info(f"eval {len(data)} records")
    results = []
    for _, row in data.iterrows():
        claims = [row["claim"]]
        contexts = [row["context"]]

        inputs = tokenizer(contexts, return_tensors="pt", padding=True)
        ids = inputs["input_ids"].to(device)

        outputs = model.generate(input_ids=ids, max_new_tokens=150, pad_token_id=tokenizer.pad_token_id)
        pred_claims = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)


        scores = scorer.score(contexts=contexts, claims=pred_claims)
        pred_scores = pd.DataFrame(scores, columns=["align_score"])
        pred_scores["dataset"] = "pred"
        # pred_scores["context"] = row["context"]
        # pred_scores["claim"] = row["claim"]
        results.append(pred_scores)

        scores = scorer.score(contexts=contexts, claims=claims)
        eval_scores = pd.DataFrame(scores, columns=["align_score"])
        # eval_scores["context"] = row["context"]
        # eval_scores["claim"] = row["claim"]
        eval_scores["dataset"] = "eval"
        results.append(eval_scores)

    return pd.concat(results)
