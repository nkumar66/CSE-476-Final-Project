#!/usr/bin/env python3
"""
Generate an answer file that matches the expected auto-grader format.

Now supports:
- Incremental saving (writes progress every N questions)
- Resuming from an existing answers file if it exists

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from finalProject_NilayKumar import solveQuestion


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def load_existing_answers(path: Path) -> List[Dict[str, Any]]:
    """
    If an answers file already exists, load it so we can resume.
    Otherwise return an empty list.
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, list):
        raise ValueError("Existing answers file must be a list.")
    # basic sanity check
    for idx, ans in enumerate(data):
        if "output" not in ans:
            raise ValueError(f"Existing answers missing 'output' at index {idx}")

    return data


def save_answers(path: Path, answers: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)


def build_answers(questions: List[Dict[str, Any]], output_path: Path) -> List[Dict[str, str]]:
    answers = []

    # If resume file exists, load it
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as fp:
            answers = json.load(fp)
        print(f"[Resume] Loaded {len(answers)} existing answers.")
    else:
        print("[Start] No existing answers found, starting fresh.")

    total = len(questions)
    print(f"[Info] Total questions: {total}")

    # Continue from where we left off
    for idx in range(len(answers), total):
        q = questions[idx]

        print(f"\n[Processing] Question {idx+1}/{total}")

        try:
            result = solveQuestion(q)
        except Exception as e:
            print(f"[Error] solveQuestion crashed: {e}")
            result = ""  # write empty so the pipeline continues safely

        answers.append({"output": str(result)})

        # SAVE AFTER EVERY QUESTION
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(answers, fp, ensure_ascii=False, indent=2)

        print(f"[Saved] {idx+1}/{total} written to disk.")

    return answers



def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)

    # Pass output_path to ensure correct behavior
    answers = build_answers(questions, OUTPUT_PATH)

    # Final validate
    validate_results(questions, answers)

    print(f"\n[Done] Wrote {len(answers)} answers to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
