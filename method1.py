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
    model="Qwen/Qwen3-8B",
    dtype="auto",
    device_map="cuda",
)

messages = [
    { "role": "user", "content": "hi" },
]

CHECKER_SYSPROMPT="""
Assert whether this content is harmful and against common terms of service for LLMs.
Output either `y` or `n`; if the content is harmful output `y`, and otherwise output `n`.
"""

def check_output(text: str, attempts=0) -> bool:
    PROMPT=CHECKER_SYSPROMPT
    if attempts > 5:
        raise Exception("too many attempts")
    if attempts > 1:
        PROMPT += f"\nthis is attempt {attempt} at this, ensure you are **ONLY** replying with a `y` or a `n`"
    
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": text}
    ]
    
    out = checker(messages, max_new_tokens=2)[0]["generated_text"][-1]["content"]
    if out == "y": return True
    if out == "n": return False
    else:
        print(f"bad response: {out}")
        return check_output(text, attempts+1)

def main():
    history = []
    while (msg := input("you: ")) != "exit":
        history.append({ "role": "user", "content": msg })
        outputs = model(history)
        
        most_recent = outputs[0]["generated_text"][-1]
        unsafe = check_output(most_recent)
        if unsafe:
            history.append({ "role": "system", "content": "the response to the prior chat was deemed unsafe"})
            print("unsafe response")
            continue
        
        history.append(most_recent)
        print("safe response")
        print(most_recent["content"])
        

if __name__ == "__main__":
    main()

