import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from typing import Tuple

class NeuroTracer:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        self.max_entropy = math.log(self.vocab_size)

    def get_metrics(self, text: str, top_k: int = 50) -> Tuple[float, Tuple[int, int]]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the Logits for the final token
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1) # Fixed: dim=-1 for 1D tensor
        
        top_k_probs, _ = torch.topk(probs, top_k)
        top_k_probs = top_k_probs / top_k_probs.sum()
        entropy = -torch.sum(top_k_probs * torch.log(top_k_probs + 1e-10)).item()

        # Dynamic Expert Detection
        attentions = outputs.attentions
        max_score = -1.0
        expert_coor = (0, 0)

        # Scan the last 3 layers
        for l_idx in range(len(attentions) - 3, len(attentions)):
            layer = attentions[l_idx]
            for h_idx in range(layer.shape[1]):
                score = layer[0, h_idx, -1, :].mean().item()
                if score > max_score:
                    max_score = score
                    expert_coor = (l_idx, h_idx)

        return entropy, expert_coor

    def analyze_claim(self, base: str, claim: str) -> dict:
        h_base, _ = self.get_metrics(base)
        h_claim, expert = self.get_metrics(claim)

        # Normalized Risk Score Logic
        delta_h = (h_claim - h_base) / self.max_entropy
        h_norm = h_claim / self.max_entropy
        risk = (0.7 * delta_h) + (0.3 * h_norm)

        return {
            "base_entropy": h_base,
            "claim_entropy": h_claim,
            "expert_location": expert,
            "risk_score": risk,
            "is_hallucination": risk > 0.4
        }