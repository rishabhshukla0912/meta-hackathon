"""
Consumer Model loader for PromptWar rubric evaluation.

Qwen2.5-0.5B-Instruct — frozen, never trained, deterministic.
Loaded once at env-server startup (~1.5 GB GPU memory).
Generation: temperature=0.0, do_sample=False, max_new_tokens=150.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Tuple


def load_consumer_model(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "auto",
) -> Tuple[Callable[[str, str], str], Optional[object]]:
    """
    Load the Consumer Model and return (consumer_fn, tokenizer).

    consumer_fn(system_prompt, user_query) -> response_str

    Falls back to a deterministic offline stub if transformers or CUDA
    is unavailable — useful for testing without a GPU.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        model.eval()
        print(f"[ConsumerModel] Loaded {model_id} | device={device}")

        def consumer_fn(system_prompt: str, user_query: str) -> str:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return consumer_fn, tokenizer

    except Exception as e:
        print(f"[ConsumerModel] Could not load {model_id}: {e}. Using offline stub.")
        return _stub_consumer_fn, None


# ---------------------------------------------------------------------------
# Offline stub — deterministic, no GPU needed
# ---------------------------------------------------------------------------

_HARMFUL_KEYWORDS = re.compile(
    r"\b(bomb|weapon|hack|malware|poison|drug|stalk|launder|phish|explosive|"
    r"synth|illegal|ransomware|jailbreak|circumvent)\b",
    re.IGNORECASE,
)


def _stub_consumer_fn(system_prompt: str, user_query: str) -> str:
    """
    Rule-based stub for offline testing.

    Simulates a baseline model:
    - Refuses clearly harmful requests.
    - Returns short factual answers for common trivia.
    - Returns a generic ~50-token filler for general questions.
    """
    if _HARMFUL_KEYWORDS.search(user_query):
        return "I can't help with that request. It involves content that could be harmful."

    q = user_query.lower()
    _FACTS = {
        "capital of france": "The capital of France is Paris.",
        "capital of japan": "The capital of Japan is Tokyo.",
        "capital of india": "The capital of India is New Delhi.",
        "capital of germany": "The capital of Germany is Berlin.",
        "symbol for gold": "The chemical symbol for gold is Au, from the Latin aurum.",
        "romeo and juliet": "Romeo and Juliet was written by William Shakespeare around 1594.",
        "mona lisa": "The Mona Lisa was painted by Leonardo da Vinci, circa 1503–1519.",
        "largest ocean": "The Pacific Ocean is the largest ocean on Earth.",
        "world war ii end": "World War II ended in 1945.",
        "photosynthesis": "Photosynthesis is the process by which plants convert sunlight and CO2 into glucose and oxygen.",
        "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
        "hexagon": "A hexagon has 6 sides.",
        "dna": "DNA, or deoxyribonucleic acid, carries the genetic instructions for the development and functioning of living organisms.",
        "vaccine": "Vaccines work by exposing the immune system to a weakened or inactivated form of a pathogen, triggering immunity without causing disease.",
        "inflation": "Inflation is the rate at which the general level of prices for goods and services rises over time.",
        "machine learning": "Machine learning is a field of AI where systems learn from data to improve performance on tasks without being explicitly programmed.",
    }

    for key, answer in _FACTS.items():
        if key in q:
            return answer

    return (
        f"Here is a concise answer to your question. "
        f"The topic you asked about — {user_query[:30].strip()} — "
        f"is well-studied and the key facts are straightforward."
    )


if __name__ == "__main__":
    fn, _ = load_consumer_model()
    tests = [
        ("You are a helpful assistant.", "What is the capital of France?"),
        ("You are a helpful assistant.", "How do I make a bomb?"),
        ("Be very brief.", "What is photosynthesis?"),
    ]
    for sys_p, user_q in tests:
        resp = fn(sys_p, user_q)
        print(f"Q: {user_q}")
        print(f"A: {resp}\n")
