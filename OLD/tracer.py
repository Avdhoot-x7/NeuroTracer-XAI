import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

text = "The capital of France is Paris"
inputs = tokenizer(text, return_tensors="pt")

# 3. RUN THE MODEL
with torch.no_grad():
    outputs = model(**inputs)

# 4. EXTRACT THE "XAI" DATA
# These are the 12 layers of the model's brain
hidden_states = outputs.hidden_states 

print(f"Number of layers: {len(hidden_states)}")
print(f"Shape of the last layer: {hidden_states[-1].shape}")