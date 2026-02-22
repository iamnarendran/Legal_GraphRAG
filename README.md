# Legal_GraphRAG
A working GraphRAG that lets users ask legal questions in natural language, answered by traversing a knowledge graph of SC judgments, judges, cited acts, and legal concepts.

---

## Tech Stack (All Free Tier)

| Component | Tool | Why |
|---|---|---|
| Notebooks | Google Colab (free GPU) | Development environment |
| PDF Parsing | PyMuPDF (`fitz`) | Fast, handles legal PDFs well |
| Entity Extraction | OpenRouter → Gemini Flash | Cheapest LLM, ~$0.0001/call |
| Graph Database | Neo4j AuraDB Free (200MB cloud) | No local setup, Cypher-ready |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Free, runs on Colab, 384-dim |
| Vector Index | Neo4j built-in vector index | Everything in one DB, no Chroma/Qdrant needed |
| RAG Answer LLM | OpenRouter → Gemini 1.5 Pro | Free credits, good reasoning |
| UI | Streamlit (share via tunnel) | Fast to build, looks professional |
| Agent (Stretch) | LangGraph + OpenRouter | Showcases agentic AI skill directly |

---
