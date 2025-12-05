import json
import random

QUESTIONS_PATH = "cse_476_final_project_test_data.json"  # or dev data
ANSWERS_PATH   = "cse_476_final_project_answers.json"

N = 5   # number of random samples to inspect

# Load data
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    qs = json.load(f)

with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
    ans = json.load(f)

# Safety checks
assert isinstance(qs, list)
assert isinstance(ans, list)

num_answers = len(ans)
print(f"Loaded {len(qs)} questions and {num_answers} answers.")
print("Sampling only from completed answers.\n")

# Random sample only from available answers
indices = random.sample(range(num_answers), min(N, num_answers))

for i in indices:
    print("=" * 60)
    print(f"Index: {i}")
    print(f"QUESTION:\n{qs[i]['input']}\n")
    print(f"MODEL ANSWER: {ans[i]['output']}\n")

    # For dev data: show gold if exists
    if "output" in qs[i] and qs[i] is not ans[i]:  
        print(f"GOLD ANSWER: {qs[i]['output']}")
    else:
        print("(Test data or no gold answer available.)")
