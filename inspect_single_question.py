import json

QUESTION_INDEX = 2498

with open("cse_476_final_project_test_data.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("cse_476_final_project_answers.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

print("=== QUESTION ===")
print(questions[QUESTION_INDEX]["input"])

print("\n=== CURRENT ANSWER ===")
print(answers[QUESTION_INDEX]["output"])
