"""Prompts. Kept in one place so every model and baseline sees exactly the same wording."""

ANSWER_SYSTEM = "Answer the question with only the answer, in as few words as possible. No explanation."

CONFIDENCE_SYSTEM = (
    "You will see a question and a proposed answer. Estimate the probability that the proposed answer "
    "is correct. Reply with a single integer from 0 to 100 and nothing else."
)


def answer_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": question}]


def confidence_messages(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": CONFIDENCE_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nProposed answer: {answer}\nProbability correct (0-100):"},
    ]


def render(tokenizer, messages: list[dict]) -> str:
    """Use the model's chat template when it has one; plain text otherwise (tiny test models)."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return text + "\nassistant:"
