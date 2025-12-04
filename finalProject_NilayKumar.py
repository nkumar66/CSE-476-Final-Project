import os, json, textwrap, re, time
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              


#trying to implement chain of thought, so the main change I've made here is increasing the maximum tokens so it can actually go through the chain of thought. The end
#algorithm I have in mind is to implement self-consistency, so multiple COTs are ran at the same time and then majority vote on an answer. Ideally seems to be the 
#best way to get an accurate answer, though may not be very fast.

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

#I want to classify the type of question with the first prompt to act as a router between different solving techniques/ inference time algorithms:
def ClassifyQuestionType(question: str) -> str:
    #first prompt response from model should be question type
    system = (
        "You are a classifier. "
        "Your job is to look at a question and decide if it is a math competition "
        "style problem that requires calculations / equations, or something else."
    )

    prompt = f"""Classify the following question as one of two labels:

    - math
    - other

    Rules:
    - Respond with exactly one word: either math or other.
    - 'math' means contest-style math, numeric answers, equations, geometry, counting, etc.
    - Everything else is 'other'.

    Question:
    {question}
    """
    print("TEST2")
    result = call_model_chat_completions(
        prompt=prompt,
        system=system,
        temperature=0.0,
    )
    print("TEST3")
    raw = (result["text"] or "").strip().lower()

    if "math" in raw:
        return "math"
    return "other"

#chain of thought implementation, my idea to get chain of thoughts is to prompt the model to do the problem in steps, so that with each subproblem, 
#forms a chain of thought

def ChainOfThought(question, temperature=0.7):
    system = (
        "You are an expert contest math solver. "
        "Always follow the user's formatting instructions exactly."
    )
    prompt = f"""Solve the problem step by step in plain text.
    Use at most 4 short steps, each on its own line.
    After the steps, on the very last line, write EXACTLY:

    Final answer: <integer>

    Replace <integer> with the final numeric answer and nothing else.

    Problem:
    {question}
    """

    result = call_model_chat_completions(
        prompt=prompt,
        system=system,
        temperature=temperature,
    )
    return result["text"]


def extract_integer_final(text: str):
    #first check if the answer is shown at the "Final Answer" text
    final = re.search(r"Final answer:\s*([\-]?\d+)", text)

    if final:
        return final.group(1).strip()
    
    #it was not at the final answer text, so just last number that was printed out
    final = re.findall(r"[\-]?\d+", text)
    return final[-1] if final else None


#now to implement the actual self consistency, so I want to run the chain of thought multiple times, and keep a list of all the final answers, and then simulate voting
#in the sense of taking the most common answer in the list, if all the answers are different, maybe I should keep running until I come across an answer that already
#exists in the array and return that

def SelfConsistency(question, attempts=5,temperature=0.7):
    finalAnswers = []
    frequency = {}
    for i in range(attempts):
        textVersion = ChainOfThought(question, temperature=temperature)
        number = extract_integer_final(textVersion)

        if number is None:
            continue
        
        finalAnswers.append(number)
        frequency[number] = frequency.get(number, 0) + 1

    if not frequency:
        return None
    
    #now we have all our finalAnswers and frequency table filled out, we can pick the highest frequency number
    bestAnswer = None
    bestCount = 0

    for value, count in frequency.items():
        if count > bestCount:
            bestCount = count
            bestAnswer = value

    return bestAnswer


#basic algorithm for direct answering or zero shot prompting / few shot prompting
def DirectAnswer(question: str, temperature: float = 0.0) -> str:
    system = "You are a helpful assistant. Reply with only the final answer, no explanation."

    result = call_model_chat_completions(
        prompt=question,
        system=system,
        temperature=temperature,
    )

    return (result["text"] or "").strip()


#MAIN FUNCTION CALL
#this function will act as the main entrypoint for a question from the given JSON, it will return a string with only the final answer
def solveQuestion(question: dict) -> str:
    text = question["input"]
    #classify the problem first
    print("TEST1")
    problemType = ClassifyQuestionType(text)
    print("TEST")

    #based on that answer, route to appropriate inference time algorithm
    if problemType == "math":
        return SolveMath(text)
    #direct answer for now if it's not math, will work on other techniques
    answer = DirectAnswer(text, temperature=0.0)
    return str(answer)

#the solver was taking too long so I decided to implement a function to determine whether problems are shorter or longer
#because we don't need the self consistency on easier math problems or chain of thought if it's really easy
def SolveMath(question: str) -> str:
    # Heuristic: short problems are likely easier
    if len(question) < 100:
        #faster path
        cotText = ChainOfThought(question, temperature=0.5)
        answer = extract_integer_final(cotText)
        if answer is None:
            #if it doesn't work just do direct answer
            answer = DirectAnswer(question, temperature=0.0)
        return str(answer)

    # Heavy path: self-consistency with fewer samples
    answer = SelfConsistency(question, attempts=3, temperature=0.3)
    if answer is None:
        answer = DirectAnswer(question, temperature=0.0)
    return str(answer)