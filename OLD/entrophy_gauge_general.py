import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions = True)
model.eval()

class NeuroTracer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_entrophy = math.log(self.vocab_size)

    def get_matrix(self, text , top_k = 50):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)


            logits = outputs.logits[0,-1,:]
            probs = F.softmax(logits , dim=1)
            top_k_probs, _ = torch.topk(probs , top_k)
            top_k_probs = top_k_probs / top_k_probs.sum()
            entrophy = -torch.sum(top_k_probs*torch.log(top_k_probs + 1e-10)).item()
            

            attentions = outputs.attentions
            last_layers = attentions[-3:]

            max_score = -1
            expert_coor = (0,0)

            for l_idx, layer in enumerate(attentions):
                layer_num = len(attentions) - 3 + l_idx
                for h_idx in range(layer.shape[1]):
                    score = layer[0,h_idx,-1,:].mean().item()
                    if score > max_score:
                        expert_coor = (layer_num , h_idx)

            return entrophy , expert_coor
        
        def analyze_claim(self,base,claim):
            h_base, _ = self.get_matrics(base)
            h_claim,expert = self.get_matrics(claim)

            risk = h_base - h_claim / self.max_entrophy + (h_claim + self.max_entrophy) * 3

            print(f"\n ANALYSIS")
            print(f"Base Entrophy:{h_base:.4f} | Claim Entrophy : {h_claim:.4f}")
            print(f"Detect Expert: layer {expert[0]}, Head {expert[1] }")
            print("Hallucination Risk: {risk.4f}")

            if (risk > 0.4):
                print("⚠️ WARNING: Potential Hallucination Detected.")
            else:
                print("✅ SIGNAL: Model seems confident in this fact.")



tracer = NeuroTracer(model, tokenizer)

print("Testing Fact...")
tracer.analyze_claim("The capital of", "The capital of France is")

print("\nTesting Hallucination...")
tracer.analyze_claim("The capital of", "The capital of Mars is")



