"""Exercise 1 evaluation: strict JSON exact match + per-field F1.

Runs the base SmolLM-135M and the LoRA-tuned variant on (a) the held-out test
split and (b) the full 10-row dataset, parsing the model's first-completion as
strict JSON. Anything that fails to parse OR contains extra prose around the
JSON object scores 0 for that row. Saves predictions + a small error table.
"""
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ex1_lora_finetune import MODEL_NAME, PROMPT_TEMPLATE, build_prompt

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "ex1_medical.json"
SPLIT_PATH = ROOT / "outputs" / "ex1_split.json"
ADAPTER_DIR = ROOT / "outputs" / "lora_adapter"
RESULTS_PATH = ROOT / "outputs" / "ex1_results.json"
ERRORS_PATH = ROOT / "outputs" / "ex1_errors.json"

FIELDS = ("Drug", "Dosage", "Adverse_Effect")


def parse_strict_json(text: str):
    """Return parsed dict if the *entire* completion (after stripping whitespace)
    is a JSON object with exactly the three required keys; else None."""
    s = text.strip()
    # Cut at first newline so trailing tokens after the JSON are tolerated only
    # when the model emits the JSON on one line followed by a newline -- we are
    # being slightly lenient: extra content after a newline is still failure
    # under "strict", so we DO require the full string to be valid JSON.
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != set(FIELDS):
        return None
    return obj


def normalize(v):
    if v is None:
        return ""
    return str(v).strip().lower()


def per_field_match(pred, gold):
    return {f: int(normalize(pred.get(f)) == normalize(gold.get(f))) for f in FIELDS}


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def evaluate(model, tokenizer, rows, label):
    preds = []
    em_count = 0
    field_correct = {f: 0 for f in FIELDS}
    field_total = len(rows)
    for r in rows:
        prompt = build_prompt(r["text"])
        raw = generate(model, tokenizer, prompt)
        gold = json.loads(r["output"])
        parsed = parse_strict_json(raw)
        # Only score field-level matches when JSON parses; otherwise zero out
        if parsed is None:
            row_em = 0
            field_match = {f: 0 for f in FIELDS}
        else:
            field_match = per_field_match(parsed, gold)
            row_em = int(all(field_match[f] == 1 for f in FIELDS))
        em_count += row_em
        for f in FIELDS:
            field_correct[f] += field_match[f]
        preds.append({
            "text": r["text"],
            "gold": gold,
            "raw_completion": raw,
            "parsed": parsed,
            "exact_match": row_em,
            "field_match": field_match,
        })

    em = em_count / len(rows)
    field_f1 = {f: field_correct[f] / field_total for f in FIELDS}
    macro = sum(field_f1.values()) / len(FIELDS)
    print(f"\n[{label}] Strict EM = {em:.3f}  ({em_count}/{len(rows)})")
    for f in FIELDS:
        print(f"   {f:>16s} acc = {field_f1[f]:.3f}")
    print(f"   macro field acc = {macro:.3f}")
    return {
        "label": label,
        "n": len(rows),
        "exact_match": em,
        "field_acc": field_f1,
        "macro_field_acc": macro,
        "predictions": preds,
    }


def load_data():
    with open(DATA_PATH) as f:
        full = json.load(f)
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    return full, split["train"], split["test"]


def main():
    full, train, test = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading BASE model")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    base.eval()
    if torch.cuda.is_available():
        base = base.cuda()

    base_full = evaluate(base, tokenizer, full, "BASE / full-10")
    base_test = evaluate(base, tokenizer, test, "BASE / test-2")

    del base
    torch.cuda.empty_cache()

    print("\n>>> Loading LoRA-tuned model")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    tuned = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    tuned.eval()
    if torch.cuda.is_available():
        tuned = tuned.cuda()

    tuned_full = evaluate(tuned, tokenizer, full, "LoRA / full-10")
    tuned_test = evaluate(tuned, tokenizer, test, "LoRA / test-2")
    tuned_train = evaluate(tuned, tokenizer, train, "LoRA / train-8")

    results = {
        "base_full": base_full,
        "base_test": base_test,
        "lora_full": tuned_full,
        "lora_test": tuned_test,
        "lora_train": tuned_train,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # Surface failures from BOTH models for the error analysis
    failures = []
    for run in (base_full, tuned_full):
        for p in run["predictions"]:
            if p["exact_match"] == 0:
                failures.append({
                    "model": run["label"],
                    "text": p["text"],
                    "gold": p["gold"],
                    "raw_completion": p["raw_completion"],
                    "parsed": p["parsed"],
                    "field_match": p["field_match"],
                })
    ERRORS_PATH.write_text(json.dumps(failures, indent=2))
    print(f"\nSaved {RESULTS_PATH.name} and {ERRORS_PATH.name} ({len(failures)} failures)")


if __name__ == "__main__":
    main()
