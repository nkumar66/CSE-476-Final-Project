import os, json, textwrap, re, time
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              


#trying to implement chain of thought, so the main change I've made here is increasing the maximum tokens so it can actually go through the chain of thought. The end
#algorithm I have in mind is to implement self-consistency, so multiple COTs are ran at the same time and then majority vote on an answer. Ideally seems to be the 
#best way to get an accurate answer, though may not be very fast.

#result, increasing maximum tokens increased model accuracy yes, but runtime started to take far too long, multiple hours
#to create the dev data answers, so changing it to modify token limits for how its called, so in chain of thought
#it'll be called for max tokens of 512, and 128 in other scenarios

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60,
                                max_tokens: int = 128) -> dict:
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
        "max_tokens": max_tokens,
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
        "Your job is to look at a question and assign exactly one category. "
    )

    prompt = f"""Classify the following question as one of the following:

    - math
    - coding
    - futurePrediction
    - planning
    - commonSense


    Rules:
    - Respond with exactly one word. 
    - 'math' means contest-style math, numeric answers, equations, geometry, counting, etc.
    - 'coding' means programming questions, code interpretation, debugging, writing functions, etc.
    - 'futurePlanning' means the question has the words "the event to be predicted:"
    - 'planning' means the sentence starts with "I am" in a planning context.
    - 'commonSense' means the question is really simple and straightforward, and not the other ones.

    Question:
    {question}
    """
    #print("TEST2")
    result = call_model_chat_completions(
        prompt=prompt,
        system=system,
        temperature=0.0,
        max_tokens=17
    )
    #print("TEST3")
    raw = (result["text"] or "").strip().lower()
    return raw

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
        max_tokens=512
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

#changed to 1 because it would genuinely take like 2 hours if i wanted to run all the dev data, meaning it would take 
#12 hours if i wanted to run the test data, if i have time then use it
def SelfConsistency(question, attempts=3,temperature=0.7):
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
    #print("TEST1")
    problemType = ClassifyQuestionType(text)
    #print("TEST")

    #based on that answer, route to appropriate inference time algorithm
    if problemType == "math":
        return SolveMath(text)
    elif problemType == "coding":
        return SolveCoding(text)
    elif problemType == "futurePrediction":
        return SolveFuturePrediction(text)
    elif problemType == "planning":
        return SolvePlanning(text)
    elif problemType == "commonSense":
        return DomainDirectAnswer(text)
    
    #if it looks like MCQ, solve with MCQ solver
    if MCQLikeQuestion(text):
        answer = FewShotMC(text, temperature=0.0)
        return answer
    #direct answer for now if it's not math, will work on other techniques
    answer = DomainDirectAnswer(text, temperature=0.0)
    return str(answer)

#the solver was taking too long so I decided to implement a function to determine whether problems are shorter or longer
#because we don't need the self consistency on easier math problems or chain of thought if it's really easy
def SolveMath(question: str) -> str:
    #short problems are likely easier
    if len(question) < 100:
        #faster path
        cotText = ChainOfThought(question, temperature=0.5)
        answer = extract_integer_final(cotText)
        if answer is None:
            #if it doesn't work just do direct answer
            answer = DirectAnswer(question, temperature=0.0)
        return str(answer)

    #self-consistency if the question is a hard math problem
    answer = SelfConsistency(question, attempts=3, temperature=0.3)
    if answer is None:
        answer = DirectAnswer(question, temperature=0.0)
    return str(answer)


#if I help detect multiple choice questions, then we can use few shot prompting to increase model accuracy for MCQ questions
def MCQLikeQuestion(question: str) -> bool:
    q = question.lower()

    patterns = [" a.", " b.", " c.", " d.",
                " a)", " b)", " c)", " d)",
                "\na.", "\nb.", "\nc.", "\nd."]
    
    return any(p in q for p in patterns)

#use few shot prompting for the mcq, to help MCQ accuracy
def FewShotMC(question_text: str, temperature: float = 0.0) -> str:

    system = (
        "You answer questions using the patterns shown in the examples. "
        "Reply with only the final answer (a single letter like A, B, C, or D, "
        "or a short phrase), no explanation."
    )

    examples = """
    Q: A student walks to school one morning and notices the grass is wet. Which process most likely caused the grass to be wet?
        A. condensation  B. erosion  C. evaporation  D. precipitation
    Answer: D

    Q: Which part of the plant is mainly responsible for photosynthesis?
        A. roots  B. stems  C. leaves  D. flowers
    Answer: C

    Q: Water turns into water vapor in which process?
        A. freezing  B. melting  C. evaporation  D. condensation
    Answer: C
    """

    prompt = f"""{examples}

    Now answer this question in the same format:

    Q: {question_text}
    Answer:"""

    result = call_model_chat_completions(
        prompt=prompt,
        system=system,
        temperature=temperature,
    )
    return (result["text"] or "").strip()

# better response system for when the problem is not math and not MCQ
def DomainDirectAnswer(question: str, temperature: float = 0.0) -> str:
    system = (
        "You are a concise, reliable question-answering assistant. "
        "Use any necessary reasoning internally, but reply with only the final answer, "
        "as a short phrase or sentence, no explanation."
    )

    result = call_model_chat_completions(
        prompt=question,
        system=system,
        temperature=temperature,
    )
    return (result["text"] or "").strip()

def SolveCoding(question: str, temperature: float = 0.0) -> str:
    
    system = (
        "You are a professional coding agent. "
        "You analyze programs, debug code, and compute outputs. "
        "Think like a programmer and return ONLY the final answer. "
        "Do not explain. Do not include imports unless required by the question."
    )

    result = call_model_chat_completions(
        prompt=question,
        system=system,
        temperature=temperature,
        max_tokens=256
    )
    return (result["text"] or "").strip()

def SolveFuturePrediction(question: str, temperature: float = 0.0) -> str:
    
    system = (
        "You are a future prediction assistant. "
        "Your job is to read a scenario and give the predicted outcome. "
        "Output ONLY what the question asks for. No explanation."
    )

    result = call_model_chat_completions(
        prompt=question,
        system=system,
        temperature=temperature,
        max_tokens=128
    )
    return (result["text"] or "").strip()


def SolvePlanning(question: str, temperature: float = 0.0) -> str:
    
    system = (
        "You are a planning and decision-making assistant. "
        "You help determine the next steps, decisions, or actions. "
        "Output only the final recommended answer with no explanation."
    )

    result = call_model_chat_completions(
        prompt=question,
        system=system,
        temperature=temperature,
        max_tokens=128
    )
    return (result["text"] or "").strip()