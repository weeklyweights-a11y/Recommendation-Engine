# Resume2Vec — Encoder Comparison for Resume-to-JD Matching

## Links

- **Paper:** https://www.mdpi.com/2079-9292/14/4/794
- **DOI:** https://doi.org/10.3390/electronics14040794
- **Journal:** *Electronics* (MDPI), February 2025

## Summary

Resume2Vec systematically compares encoder models for resume-to-job-description matching using cosine similarity in embedding space. It benchmarks classical transformers (BERT, RoBERTa, DistilBERT) against larger models (GPT, Gemini, Llama) on resume–JD retrieval quality, latency, and resource cost.

## Key Architecture Details

| Model family | Role in study |
|--------------|---------------|
| **BERT / RoBERTa** | Strong baselines for semantic similarity; RoBERTa often edges BERT on downstream matching |
| **DistilBERT** | ~40% smaller/faster than BERT with modest quality tradeoff — good MVP candidate |
| **GPT / Gemini / Llama** | API or local LLM embeddings; higher quality potential, higher cost/latency |
| **Matching method** | Encode resume and JD independently (bi-encoder), cosine similarity for ranking |
| **Evaluation** | Pairwise resume–JD relevance; compares precision/recall across encoders |

## Key Findings (high level)

- **Bi-encoder + cosine similarity** is a strong, simple baseline for resume–JD matching.
- **Smaller distilled models** (DistilBERT, MiniLM-class) offer the best latency/quality tradeoff for production.
- **Larger LLM embeddings** can improve match quality but add API cost and slower indexing — best reserved for reranking or explanation, not bulk ANN indexing.
- Domain-specific fine-tuning (even light) typically beats zero-shot general encoders for HR text.

## What We Borrow

1. **Benchmark understanding** — Informs our `models_comparison.md` encoder shootout.
2. **Bi-encoder pattern** — Independent resume/JD encoding + FAISS ANN is validated by this line of work.
3. **Encoder tiering** — Fast model for indexing (MiniLM), stronger model for reranking (BGE reranker or cross-encoder).

## Recommended MVP Encoder Shortlist (derived)

| Tier | Model | Use |
|------|-------|-----|
| Fast index | `all-MiniLM-L6-v2` | FAISS bulk encoding (JobMatchAI choice) |
| Quality index | `BAAI/bge-large-en-v1.5` | Higher-recall semantic channel |
| Rerank | `BAAI/bge-reranker-large` | Stage-2 cross-encoder rerank |
