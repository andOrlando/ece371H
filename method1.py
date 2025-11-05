import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

checker = transformers.pipeline(
    "text-generation",
    model="ibm-granite/granite-4.0-h-350M",
    device_map="cuda",
)

model = transformers.pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="cuda",
)

messages = [
    { "role": "user", "content": "hi" },
]

outputs = model(messages, max_new_tokens=256)

# print(outputs[0]["generated_text"][-1])
# print(outputs)

CHECKER_SYSPROMPT="""
Assert whether this content is harmful and against common terms of service for LLMs.
Output either `y` or `n`; if the content is harmful output `y`, and otherwise output `n`.
"""

def check_output(text: str, attempts=0) -> bool:
    PROMPT=CHECKER_SYSPROMPT
    if attempts > 1:
        PROMPT += f"\nthis is attempt {attempt} at this, ensure you are **ONLY** replying with a `y` or a `n`"
    
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user": text}
    ]
    
    out = checker(messages, max_new_tokens=2)[0]["generated_text"][-1]
    if out == "y": return True
    if out == "n": return False
    else:
        return check_output(text, attempts+1)

def main():
    history = []
    while (msg := input("you: ")) != "exit":
        history.append({ "role": "user", "content": msg })
        outputs = model(history)
        
        outputs[0]["generated_text"][-1]






