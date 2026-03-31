# ⚖️ [LegalGraph AI](https://legalgraphrag.streamlit.app/)
### Indian Supreme Court Judgment Intelligence Platform

> A hybrid GraphRAG system that combines Neo4j knowledge graph traversal with semantic vector search to enable intelligent legal research over Indian Supreme Court judgments.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI_GPT_OSS_20B-412991?style=flat&logo=openai&logoColor=white)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 What is this?

Traditional legal research relies on keyword-based search you search "Article 21" and get documents containing that exact phrase. **LegalGraph AI thinks differently.**

It models the relationships between cases, judges, acts, sections, and legal concepts as a **knowledge graph**. When you ask *"Cases decided by M. R. Shah on land acquisition after 2020"*, the system:

1. Understands you're asking about a **judge**, an **act**, and a **year range**
2. Traverses the graph  *Judge → Cases → Acts* to find precisely linked judgments
3. Simultaneously runs semantic vector search to catch conceptually related cases
4. Synthesizes a cited, structured answer using an LLM

The result is legal research that understands *meaning*, not just keywords.

---

## 🏗️ Architecture

<img width="4710" height="3770" alt="Mermaid Chart - Create complex, visual diagrams with text -2026-02-23-150654" src="https://github.com/user-attachments/assets/aa6a0029-3062-4da0-99f4-275f2536941b" />

---

## 🧠 How GraphRAG Differs from Plain RAG

| Feature | Plain RAG | LegalGraph AI |
|---|---|---|
| Search method | Text similarity only | Graph traversal + semantic search |
| Query: "M. R. Shah cases" | Finds documents mentioning his name | Traverses `Judge → Cases` relationship |
| Query: "Article 21 cases" | Keyword match | Finds all cases with `REFERENCES_SECTION → Article 21` |
| Multi-hop reasoning | ❌ | ✅ Cases → Acts → Sections → related Cases |
| Citation network | ❌ | ✅ `CITES_CASE` relationships between judgments |
| Precision | Medium | High — explicit relationships, not approximate matches |

---

## 📂 Project Structure

```
legal-graphrag/
│
├── 01_pdf_extraction.ipynb          # Extract text from SC judgment PDFs
├── 02_entity_extraction_and_graph.ipynb  # LLM entity extraction + Neo4j loading
├── 03_graphrag_engine.ipynb         # Hybrid retrieval pipeline + validation
├── 04_launch_streamlit.ipynb        # Launch Streamlit app via ngrok (Colab)
│
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

### Core
| Component | Technology | Purpose |
|---|---|---|
| **Graph Database** | Neo4j AuraDB | Store case knowledge graph + vector index |
| **Query Language** | Cypher | Graph traversal for entity-linked retrieval |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) | 384-dim semantic embeddings |
| **Vector Index** | Neo4j built-in vector index | Cosine similarity search |
| **LLM** | GPT-OSS-20B via OpenRouter | Entity extraction + answer synthesis |
| **PDF Parsing** | PyMuPDF (fitz) | Text extraction from SC judgment PDFs |

### Application
| Component | Technology |
|---|---|
| **Web UI** | Streamlit |
| **LLM Gateway** | OpenRouter API |
| **Runtime** | Google Colab (T4 GPU) / Streamlit Cloud |
| **Language** | Python 3.10+ |

### Document Processing Pipeline
```
PDF → PyMuPDF → Text Cleaning → Chunking
                                    ↓
                           LLM Entity Extraction
                           (Judges, Acts, Sections, Concepts, Summary)
                                    ↓
                    Sentence Transformers Embeddings (384-dim)
                                    ↓
                    Neo4j AuraDB (Nodes + Relationships + Vector Index)
```

---

## 🕸️ Knowledge Graph Schema

```
Nodes:
  (:Case)         — id, title, citation, year, summary, embedding[384]
  (:Judge)        — name
  (:Act)          — name
  (:Section)      — id, number, act_name
  (:LegalConcept) — name

Relationships:
  (Case)-[:DECIDED_BY]---------->(Judge)
  (Case)-[:CITES_ACT]-----------(Act)
  (Case)-[:REFERENCES_SECTION]->(Section)-[:PART_OF]->(Act)
  (Case)-[:INVOLVES_CONCEPT]-->(LegalConcept)
  (Case)-[:CITES_CASE]---------->(Case)
```

---

## 🚀 Getting Started

### Prerequisites
- Google Colab account (free tier works)
- Neo4j AuraDB free instance → [Create here](https://neo4j.com/cloud/aura-free)
- OpenRouter API key → [Get here](https://openrouter.ai/keys)
- Indian SC judgment PDFs (sourced from [Kaggle](https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments))

### Setup

**1. Clone the repo and open in Colab**
```bash
git clone https://github.com/yourusername/legal-graphrag.git
```

**2. Run notebooks in order**
```
01_pdf_extraction.ipynb
    ↓ outputs: extracted_cases.jsonl
02_entity_extraction_and_graph.ipynb
    ↓ outputs: enriched_cases.jsonl + populated Neo4j graph
03_graphrag_engine.ipynb
    ↓ validates retrieval pipeline
04_launch_streamlit.ipynb
    ↓ outputs: public URL via ngrok
```

**3. Configure credentials**

In each notebook, update the config cell:
```python
NEO4J_URI          = 'neo4j+s://xxxxxxxx.databases.neo4j.io'
NEO4J_USER         = 'neo4j'
NEO4J_PASSWORD     = 'your-password'
OPENROUTER_API_KEY = 'sk-or-v1-...'
```

---

## 🔍 Example Queries

| Query | Retrieval Path |
|---|---|
| `"Cases decided by M. R. Shah on land acquisition"` | Graph: `Judge → Cases → Acts` |
| `"Section 302 IPC murder cases"` | Graph: `Section → Cases` + Vector |
| `"Natural justice in administrative law"` | Graph: `Concept → Cases` + Vector |
| `"Trade tax exemption and diversification"` | Vector similarity |
| `"Constitutional validity cases after 2020"` | Graph: year filter + `Concept → Cases` |

---

## 📊 Retrieval Strategy

```python
def graphrag_answer(query):
    # Step 1: Extract structured entities from natural language
    entities = extract_query_entities(query)   # LLM call

    # Step 2: Graph traversal — finds cases via explicit relationships
    graph_results = graph_retrieval(entities)  # Cypher queries

    # Step 3: Semantic search — finds conceptually similar cases
    vector_results = vector_retrieval(query)   # Neo4j vector index

    # Step 4: Merge (graph results prioritized) + LLM synthesis
    merged = merge_results(graph_results, vector_results)
    return synthesize_answer(query, merged)
```

---

## 💰 Cost Estimate

| Task | Model | Cost per 100 cases |
|---|---|---|
| Entity extraction | GPT-OSS-20B (OpenRouter) | ~$0.002 |
| Answer synthesis | GPT-OSS-20B (OpenRouter) | ~$0.05 per 10 queries |
| Embeddings | `all-MiniLM-L6-v2` (self-hosted) | **Free** |
| Vector + Graph DB | Neo4j AuraDB free tier | **Free** |
| **Total setup cost** | | **< $0.01** |

---

## 🗺️ Roadmap

- [ ] LangGraph agent with tool-use (graph search, vector search, citation trace as separate tools)
- [ ] Multi-language support — query in Hindi, Tamil, Bengali
- [ ] Citation network visualization (interactive graph in UI)
- [ ] Temporal legal research — track how interpretations evolve across decades
- [ ] Scale to full SC judgment corpus with sharded Neo4j instances

---

## 🙏 Data Source

Indian Supreme Court judgments sourced from the [Supreme Court of India Records](https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments) dataset on Kaggle.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
