# NeuroTracer: XAI-Powered Hallucination Detection

NeuroTracer is a production-grade interpretability engine designed to quantify LLM uncertainty. Unlike standard black-box models, NeuroTracer inspects internal attention heads and logit entropy to detect potential hallucinations in real-time.

## 🚀 Key Features
- **Mechanistic Interpretability:** Scans specific Transformer attention heads to identify "Expert Circuits."
- **Uncertainty Quantification:** Implements Shannon Entropy metrics to calculate a real-time "Risk Score."
- **Production-Ready API:** Built with **FastAPI** for high-concurrency asynchronous inference.
- **Enterprise DevOps:** Fully containerized with **Docker** and automated via **GitHub Actions CI/CD**.

## 🛠️ Tech Stack
- **AI:** PyTorch, Hugging Face Transformers
- **Backend:** FastAPI, Pydantic
- **Ops:** Docker, GitHub Actions, Pytest
