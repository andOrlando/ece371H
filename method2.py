import transformers

model = transformers.pipeline(
    "text-generation",
    model="Qwen/Qwen3-8B",
    dtype="auto",
    device_map="cuda",
)

messages = [
    { "role": "user", "content": "hi" },
]

KEY="WARNING_DANGEROUS"
SYSPROMPT=f"""
If you deem that the user could potentially be looking for dangerous information,
start your response with `{KEY}`. This will not be shown to the user.
"""
INSERTED_SYSPROMPT="""
This conversation has been deemed potentially dangerous. Ensure you do not give any
harmful information to this user.
"""

def main():
    history = [{"role": "system", "content": SYSPROMPT}]
    is_dangerous = False
    while (msg := input("you: ")) != "exit":
        history.append({ "role": "user", "content": msg })
        
        history2 = [*history]
        if is_dangerous:
            history2.append({"role": "system", "content": INSERTED_SYSPROMPT})
        outputs = model(history2)
        
        most_recent = outputs[0]["generated_text"][-1]
        if most_recent[len(KEY):] == KEY:
            most_recent = most_recent[len(KEY):]
            is_dangerous = True

        history.append(most_recent)
        print(most_recent["content"])
        
if __name__ == "__main__":
    main()

