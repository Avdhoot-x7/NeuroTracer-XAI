import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

# 2. Process sentence
text = "The capital of France is Paris"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 3. Extract Attention (Layer 11, Head 0 - usually very semantic)
# Shape: [batch, heads, seq_len, seq_len]
attention = outputs.attentions[11][0, 0].detach().cpu().numpy()

# 4. Get the word names (Tokens)
#tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
# GPT-2 tokens often have a leading space (Ġ), so we search for the substring
france_idx = next(i for i, t in enumerate(tokens) if "France" in t)
paris_idx = next(i for i, t in enumerate(tokens) if "Paris" in t)

print(f"Searching Layer 11 for the Geography Expert...")

# 2. Loop through all 12 heads
for head_id in range(12):
    # Extract attention for THIS specific head
    attn_matrix = outputs.attentions[11][0, head_id].detach().cpu().numpy()
    
    # Check how much 'Paris' (row) looks at 'France' (col)
    score = attn_matrix[paris_idx, france_idx]
    print(f"Head {head_id}: Attention Score = {score:.4f}")
    
    # If this head is paying more attention, let's visualize it instead of the default Head 0
    if score > attention[paris_idx, france_idx]:
        attention = attn_matrix

# 5. Plotting the "Brain Map"
plt.figure(figsize=(8, 6))
sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, annot=True, cmap="YlGnBu")
plt.title("XAI Trace: Where is the model looking?")
plt.xlabel("Words the model looks AT")
plt.ylabel("Word the model is processing")

plt.show()
