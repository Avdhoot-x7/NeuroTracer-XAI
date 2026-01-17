# from fastapi import FastAPI, HTTPException
# from engine import NeuroTracer
# from schemas import TraceRequest, TraceResponse

# app = FastAPI(title="NeuroTracer API")

# # Initialize Engine
# tracer = NeuroTracer()

# @app.post("/trace", response_model=TraceResponse)
# async def run_trace(request: TraceRequest):
#     try:
#         results = tracer.analyze_claim(request.base_prompt, request.claim_prompt)
#         return results
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8001)

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine import NeuroTracer # Assuming your current code is in engine.py

app = FastAPI()

# Load model once
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tracer = NeuroTracer(model, tokenizer)

class TraceRequest(BaseModel):
    base: str
    claim: str

@app.post("/analyze")
async def analyze(req: TraceRequest):
    # This calls your exact function
    result = tracer.analyze_claim(req.base, req.claim)
    return {"status": "success", "input": req.claim}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)