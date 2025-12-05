#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Dict, List

from finalProject_NilayKumar import solveQuestion

DEV_PATH = Path("cse476_final_project_dev_data.json")


def load_dev(path: Path) -> List[Dict[str, Any]]:
    # Explicit UTF-8 to avoid Windows encoding issues
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Dev file must contain a list of question objects.")
    return data


def main() -> None:
    questions = load_dev(DEV_PATH)

    # If you want to test on a subset first, set MAX_QUESTIONS to e.g. 50 or 200
    MAX_QUESTIONS = 200  # or e.g. 200
    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]

    total = len(questions)
    correct = 0

    # per-domain stats
    per_domain: Dict[str, Dict[str, int]] = {}
    mistakes: List[Dict[str, Any]] = []

    print(f"Evaluating on {total} dev questions...\n")

    for idx, q in enumerate(questions, start=1):
        gold = str(q["output"]).strip()
        pred = solveQuestion(q).strip()
        domain = q.get("domain", "unknown")

        # update domain stats
        if domain not in per_domain:
            per_domain[domain] = {"total": 0, "correct": 0}
        per_domain[domain]["total"] += 1

        if pred == gold:
            correct += 1
            per_domain[domain]["correct"] += 1
        else:
            # keep a few example mistakes
            if len(mistakes) < 20:
                mistakes.append(
                    {
                        "idx": idx,
                        "domain": domain,
                        "input": q["input"],
                        "gold": gold,
                        "pred": pred,
                    }
                )

        if idx % 25 == 0:
            print(f"[Progress] {idx}/{total} done...")

    overall_acc = correct / total if total > 0 else 0.0
    print("\n=== RESULTS ===")
    print(f"Overall accuracy: {correct}/{total} = {overall_acc:.3%}\n")

    print("Per-domain accuracy:")
    for domain, stats in per_domain.items():
        d_total = stats["total"]
        d_correct = stats["correct"]
        acc = d_correct / d_total if d_total > 0 else 0.0
        print(f"  {domain:10s}: {d_correct}/{d_total} = {acc:.3%}")

    if mistakes:
        print("\nExample mistakes (up to 20):")
        for m in mistakes:
            print("\n-----------------------------")
            print(f"Index:  {m['idx']}  (domain: {m['domain']})")
            print(f"Gold:   {m['gold']}")
            print(f"Pred:   {m['pred']}")
            print("Input:")
            print(m["input"][:400], "..." if len(m["input"]) > 400 else "")


if __name__ == "__main__":
    main()
