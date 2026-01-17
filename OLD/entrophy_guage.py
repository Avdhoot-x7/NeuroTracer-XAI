import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# ------------------ MODEL SETUP ------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

VOCAB_SIZE = tokenizer.vocab_size
MAX_ENTROPY = math.log(VOCAB_SIZE)  # theoretical max entropy


# ------------------ ENTROPY FUNCTION ------------------
def get_entropy(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(last_token_logits, dim=-1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return entropy.item()


# ------------------ HALLUCINATION SCORE ------------------
def hallucination_score(base_prompt, claim_prompt):
    H_base = get_entropy(base_prompt)
    H_claim = get_entropy(claim_prompt)

    # Relative entropy increase
    delta_H = H_claim - H_base

    # Normalized entropy (0–1)
    H_norm = H_claim / MAX_ENTROPY

    # Simple risk score (weighted but interpretable)
    risk_score = 0.7 * (delta_H / MAX_ENTROPY) + 0.3 * H_norm

    return {
        "H_base": H_base,
        "H_claim": H_claim,
        "ΔH": delta_H,
        "H_normalized": H_norm,
        "Hallucination_Risk": risk_score
    }


# ------------------ TEST CASES ------------------
base = "The capital of"
fact = "The capital of France is"
hallucination = "The capital of Mars is"

fact_result = hallucination_score(base, fact)
hallucination_result = hallucination_score(base, hallucination)

print("\nFACT CASE")
for k, v in fact_result.items():
    print(f"{k}: {v:.4f}")

print("\nHALLUCINATION CASE")
for k, v in hallucination_result.items():
    print(f"{k}: {v:.4f}")
