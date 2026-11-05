# Day 18 — Advanced RAG Document Assistant

## Features
- Upload PDF, DOCX, TXT and Markdown files, or paste text.
- Extract, clean and split documents into overlapping word chunks.
- Build local semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Build keyword search with BM25.
- Combine rankings using Reciprocal Rank Fusion (RRF).
- Rerank candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Send only the top reranked chunks to Gemini for grounded answers.
- Inspect hybrid scores, reranker scores and the final context.
- Test both answerable and unanswerable questions.

## Setup (Windows PowerShell, using uv)
Python 3.11 or 3.12 is recommended.

```powershell
cd path\to\Day_18_Advanced_RAG
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
copy .env.example .env
```

Add your Gemini API key to `.env`, or paste it in the application sidebar. Never commit `.env`.

Run:

```powershell
uv run streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Test workflow
1. Upload `sample_knowledge_base.txt` in the Knowledge Base tab.
2. Click Process documents.
3. Ask: `What is Reciprocal Rank Fusion and why is it used?`
4. Inspect the hybrid candidates, reranking scores and context sent to Gemini.
5. Ask: `What is the price of the laptop?`
6. Check whether the application says it could not find the information in the uploaded documents.

## Concepts
- **Semantic search:** embedding similarity retrieves passages with related meaning.
- **BM25:** keyword ranking is useful for exact words, acronyms, names and identifiers.
- **Hybrid search:** combines semantic and keyword result lists.
- **RRF:** `score(d) = sum(1 / (60 + rank_i(d)))`; combines ranks rather than directly adding incomparable raw scores.
- **Reranking:** a cross-encoder evaluates query/passage pairs and reorders the candidate set.
- **Grounded generation:** Gemini is instructed to answer only from selected chunks and cite source/chunk identifiers.

## Important limitations
This is a learning/demo implementation. The index is held in Streamlit session state, not persisted in a vector database. The first run downloads local model weights and may take time on an 8 GB RAM computer; CPU inference is supported but can be slow. Scanned PDFs need OCR. Prompt-based abstention is not a guarantee, so test it with multiple unsupported questions. For production, add persistent storage, access controls, robust evaluation, observability and stronger evidence/abstention checks.

## Screenshot checklist
Capture real screenshots from your running app:
1. Project files in VS Code.
2. Knowledge base after processing, including chunk count and previews.
3. Application running in the browser.
4. Hybrid retrieval candidate list with semantic, BM25 and RRF scores.
5. Reranking output with cross-encoder scores.
6. Selected context sent to Gemini.
7. A successful answer with source/chunk citation.
8. An unsupported question and the not-found response.
9. Evaluation tab showing the test summary.

Do not show your API key in screenshots.
