"""Exercise 2: Probe NLI by reading next-token logits over Yes/No.

For each example we run two prompts through the model -- the original premise
and the perturbed premise -- against a fixed hypothesis. We pull the logits for
the very last position of the prompt and softmax-normalize the two logits
{Yes, No} only, giving P(Yes) and P(No) restricted to that binary head. We then
log-difference the two passes to see whether the perturbation actually moves
the model's belief.

We deliberately use GPT-2 (a small, *old* causal LM) following the hint in the
homework: a SOTA model would saturate this task and there'd be nothing to
report. We also rerun on SmolLM-135M for an "old small vs. modern small"
contrast.
"""
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "ex2_nli.json"
RESULTS_PATH = ROOT / "outputs" / "ex2_results.json"
SUMMARY_PATH = ROOT / "outputs" / "ex2_summary.json"

MODELS = ["gpt2", "HuggingFaceTB/SmolLM-135M"]

PROMPT_TEMPLATE = (
    "Read the premise and hypothesis. Does the premise entail the hypothesis? "
    "Answer strictly 'Yes' or 'No'.\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer:"
)


def yes_no_token_ids(tokenizer):
    """Return (yes_id, no_id) for whichever surface form the tokenizer uses
    immediately after a space (i.e. continuing the prompt 'Answer:')."""
    candidates_yes = [" Yes", "Yes", " yes", "yes"]
    candidates_no = [" No", "No", " no", "no"]

    def first_single_token(cands):
        for c in cands:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                return c, ids[0]
        # fallback: use first token of the first candidate
        c = cands[0]
        ids = tokenizer.encode(c, add_special_tokens=False)
        return c, ids[0]

    yes_form, yes_id = first_single_token(candidates_yes)
    no_form, no_id = first_single_token(candidates_no)
    return yes_form, yes_id, no_form, no_id


@torch.inference_mode()
def yes_logprobs(model, tokenizer, prompt, yes_id, no_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits  # (1, T, V)
    last = logits[0, -1, :]
    pair = torch.tensor([last[yes_id].item(), last[no_id].item()],
                        device=last.device)
    log_softmax = torch.log_softmax(pair, dim=-1)
    log_p_yes = log_softmax[0].item()
    log_p_no = log_softmax[1].item()
    return log_p_yes, log_p_no


def run_model(model_name, examples):
    print(f"\n>>> {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    yes_form, yes_id, no_form, no_id = yes_no_token_ids(tokenizer)
    print(f"   Yes token: {yes_form!r} (id={yes_id})  "
          f"No token: {no_form!r} (id={no_id})")

    rows = []
    for ex in examples:
        prompt_orig = PROMPT_TEMPLATE.format(premise=ex["premise"],
                                             hypothesis=ex["hypothesis"])
        prompt_pert = PROMPT_TEMPLATE.format(premise=ex["perturbed_premise"],
                                             hypothesis=ex["hypothesis"])

        lp_yes_o, lp_no_o = yes_logprobs(model, tokenizer, prompt_orig, yes_id, no_id)
        lp_yes_p, lp_no_p = yes_logprobs(model, tokenizer, prompt_pert, yes_id, no_id)

        p_yes_o = math.exp(lp_yes_o)
        p_yes_p = math.exp(lp_yes_p)

        # Negative delta means perturbed prompt assigns LESS probability to
        # "Yes" -- the desired direction whenever the perturbation removes
        # entailment.
        delta_logp_yes = lp_yes_p - lp_yes_o
        delta_p_yes = p_yes_p - p_yes_o

        pred_orig = "Yes" if lp_yes_o > lp_no_o else "No"
        pred_pert = "Yes" if lp_yes_p > lp_no_p else "No"

        rows.append({
            "perturbation_type": ex["perturbation_type"],
            "premise": ex["premise"],
            "perturbed_premise": ex["perturbed_premise"],
            "hypothesis": ex["hypothesis"],
            "true_label": ex["true_label"],
            "P_yes_original": p_yes_o,
            "P_yes_perturbed": p_yes_p,
            "logp_yes_original": lp_yes_o,
            "logp_yes_perturbed": lp_yes_p,
            "delta_logp_yes": delta_logp_yes,
            "delta_p_yes": delta_p_yes,
            "pred_original": pred_orig,
            "pred_perturbed": pred_pert,
        })

        print(f"   [{ex['perturbation_type']:<28s}] "
              f"P(Yes) {p_yes_o:.3f} -> {p_yes_p:.3f}   "
              f"Δlogp = {delta_logp_yes:+.3f}   "
              f"pred {pred_orig}->{pred_pert}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows, label):
    """Aggregate stats per perturbation type."""
    by_type = {}
    for r in rows:
        by_type.setdefault(r["perturbation_type"], []).append(r)

    summary = {}
    for ptype, rs in by_type.items():
        deltas = [r["delta_logp_yes"] for r in rs]
        flips_to_no = sum(
            1 for r in rs if r["pred_original"] == "Yes" and r["pred_perturbed"] == "No"
        )
        avg_delta = sum(deltas) / len(deltas)
        summary[ptype] = {
            "n": len(rs),
            "avg_delta_logp_yes": avg_delta,
            "flips_yes_to_no": flips_to_no,
        }

    correct_orig = sum(1 for r in rows if r["pred_original"] == r["true_label"])
    summary["__overall__"] = {
        "model": label,
        "n": len(rows),
        "orig_accuracy_vs_true_label": correct_orig / len(rows),
        "mean_delta_logp_yes": sum(r["delta_logp_yes"] for r in rows) / len(rows),
    }
    return summary


def main():
    with open(DATA_PATH) as f:
        examples = json.load(f)

    all_results = {}
    all_summary = {}
    for m in MODELS:
        rows = run_model(m, examples)
        all_results[m] = rows
        all_summary[m] = summarize(rows, m)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    SUMMARY_PATH.write_text(json.dumps(all_summary, indent=2))
    print(f"\nSaved {RESULTS_PATH.name} and {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
