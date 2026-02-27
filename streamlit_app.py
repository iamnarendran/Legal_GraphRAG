"""
LegalGraph AI — Streamlit Demo
Indian Supreme Court Knowledge Graph + GraphRAG

Streamlit Cloud deployment:
1. Push app.py + requirements.txt to GitHub
2. share.streamlit.io → New app → select repo
3. Settings → Secrets → add your credentials as TOML
"""

import streamlit as st
import json, re, os, time
from openai import OpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="LegalGraph AI", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")

def get_secret(key):
    try:    return st.secrets[key]
    except: return os.getenv(key, "")

NEO4J_URI          = get_secret("NEO4J_URI")
NEO4J_USER         = get_secret("NEO4J_USER")
NEO4J_PASSWORD     = get_secret("NEO4J_PASSWORD")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")

@st.cache_resource
def init_clients():
    llm      = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    driver   = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return llm, driver, embedder

llm, driver, embedder = init_clients()

def run_cypher(query, params={}):
    with driver.session() as session:
        return [dict(r) for r in session.run(query, params)]

@st.cache_data(ttl=300)
def get_graph_stats():
    stats = {}
    for r in run_cypher("MATCH (n) RETURN labels(n)[0] AS l, count(n) AS c"):
        if r["l"]: stats[r["l"]] = r["c"]
    return stats

ENTITY_PROMPT = """You are a legal query parser for Indian law.
Extract entities from the query. Return ONLY raw JSON — no markdown, no backticks.
Start with { and end with }
{"judges":[],"acts":[],"sections":[],"concepts":[],"year_from":null,"year_to":null}
Concepts: 1-3 words lowercase. Empty list if not found."""

def extract_query_entities(query):
    empty = {"judges":[],"acts":[],"sections":[],"concepts":[],"year_from":None,"year_to":None}
    try:
        resp = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role":"system","content":ENTITY_PROMPT},
                      {"role":"user","content":f"Query: {query}"}],
            max_tokens=200, temperature=0.0)
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*','',raw)
        raw = re.sub(r'```$','',raw).strip()
        m = re.search(r'\{[\s\S]*\}', raw)
        return json.loads(m.group() if m else raw)
    except: return empty

def graph_retrieval(entities):
    results = []
    for sec in entities.get("sections",[]):
        results.extend(run_cypher("""
            MATCH (c:Case)-[:REFERENCES_SECTION]->(s:Section)
            WHERE s.number CONTAINS $sec
            RETURN c.id AS case_id, c.title AS title, c.citation AS citation,
                   c.year AS year, c.summary AS summary, 'Section: '+s.number AS graph_path
            ORDER BY c.year DESC LIMIT 5""", {"sec":sec}))
    for concept in entities.get("concepts",[]):
        results.extend(run_cypher("""
            MATCH (c:Case)-[:INVOLVES_CONCEPT]->(lc:LegalConcept)
            WHERE lc.name CONTAINS $concept
            RETURN c.id AS case_id, c.title AS title, c.citation AS citation,
                   c.year AS year, c.summary AS summary, 'Concept: '+lc.name AS graph_path
            ORDER BY c.year DESC LIMIT 5""", {"concept":concept.lower()}))
    for judge in entities.get("judges",[]):
        results.extend(run_cypher("""
            MATCH (c:Case)-[:DECIDED_BY]->(j:Judge)
            WHERE j.name CONTAINS $judge
            RETURN c.id AS case_id, c.title AS title, c.citation AS citation,
                   c.year AS year, c.summary AS summary, 'Judge: '+j.name AS graph_path
            ORDER BY c.year DESC LIMIT 5""", {"judge":judge}))
    for act in entities.get("acts",[]):
        results.extend(run_cypher("""
            MATCH (c:Case)-[:CITES_ACT]->(a:Act)
            WHERE a.name CONTAINS $act
            RETURN c.id AS case_id, c.title AS title, c.citation AS citation,
                   c.year AS year, c.summary AS summary, 'Act: '+a.name AS graph_path
            ORDER BY c.year DESC LIMIT 5""", {"act":act}))
    yr_from = entities.get("year_from")
    yr_to   = entities.get("year_to")
    if yr_from: results = [r for r in results if r.get("year") and r["year"] >= yr_from]
    if yr_to:   results = [r for r in results if r.get("year") and r["year"] <= yr_to]
    return results

def vector_retrieval(query, top_k=5):
    emb = embedder.encode(query).tolist()
    return run_cypher("""
        CALL db.index.vector.queryNodes('case_embeddings', $top_k, $embedding)
        YIELD node, score
        RETURN node.id AS case_id, node.title AS title, node.citation AS citation,
               node.year AS year, node.summary AS summary,
               'Similarity: '+toString(round(score*100)/100.0) AS graph_path
    """, {"top_k":top_k,"embedding":emb})

def merge_results(graph_results, vector_results):
    seen, merged = set(), []
    for r in graph_results:
        if r["case_id"] not in seen:
            seen.add(r["case_id"]); r["retrieval_type"]="Graph"; merged.append(r)
    for r in vector_results:
        if r["case_id"] not in seen:
            seen.add(r["case_id"]); r["retrieval_type"]="Vector"; merged.append(r)
    return merged[:8]

SYNTHESIS_PROMPT = """You are an expert legal analyst specializing in Indian law.
Answer the legal question using ONLY the provided case summaries.
Cite cases as [Case N]. Structure: direct answer → key cases → legal principle.
Be concise and factual. Under 250 words. If cases are insufficient, say so clearly."""

def synthesize_answer(query, cases):
    context = "\n\n".join(
        f"[Case {i+1}] {c.get('title','?')} | {c.get('citation','N/A')} | Year: {c.get('year','?')}\n"
        f"Summary: {c.get('summary','No summary.')}"
        for i,c in enumerate(cases))
    try:
        resp = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role":"system","content":SYNTHESIS_PROMPT},
                      {"role":"user","content":f"Question: {query}\n\nCASES:\n{context}"}],
            max_tokens=1000, temperature=0.2)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate answer: {e}"

def graphrag_answer(query):
    entities       = extract_query_entities(query)
    graph_results  = graph_retrieval(entities)
    vector_results = vector_retrieval(query)
    merged         = merge_results(graph_results, vector_results)
    if not merged:
        return {"answer":"No relevant cases found.","sources":[],
                "entities":entities,"graph_hits":0,"vector_hits":0}
    answer = synthesize_answer(query, merged)
    return {"answer":answer,"sources":merged,"entities":entities,
            "graph_hits":len(graph_results),"vector_hits":len(vector_results)}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ LegalGraph AI")
    st.caption("Indian Supreme Court · Knowledge Graph RAG")
    st.divider()

    st.subheader("📊 Graph Stats")
    try:
        stats = get_graph_stats()
        c1, c2 = st.columns(2)
        c1.metric("Cases",    stats.get("Case",0))
        c1.metric("Judges",   stats.get("Judge",0))
        c2.metric("Acts",     stats.get("Act",0))
        c2.metric("Concepts", stats.get("LegalConcept",0))
    except:
        st.warning("Could not load stats")

    st.divider()
    st.subheader("💡 Try These Queries")
    examples = [
        "Cases decided by B.R. Gavai on Section 302 murder where conviction was set aside",
        "Cases on adverse possession and land ownership dispute",
        "Cases involving natural justice principles",
        "Land acquisition compensation disputes 2023",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["query_input"] = ex
            st.rerun()

    st.divider()
    st.subheader("🏗️ Graph Schema")
    st.code("""(Case)-[:DECIDED_BY]->(Judge)
(Case)-[:CITES_ACT]->(Act)
(Case)-[:REFERENCES_SECTION]->(Section)
(Case)-[:INVOLVES_CONCEPT]->(LegalConcept)""", language="text")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("⚖️ LegalGraph AI")
st.caption("Indian Supreme Court Judgments · Hybrid GraphRAG · Neo4j + Semantic Search")
st.divider()

query = st.text_area(
    "**Ask a legal question:**",
    value=st.session_state.get("query_input",""),
    height=100,
    placeholder="e.g. Cases on land acquisition compensation decided by M. R. Shah"
)

search_clicked = st.button("🔍 Search Knowledge Graph", type="primary")


if search_clicked and query.strip():
    with st.spinner("Searching graph and generating answer..."):
        start  = time.time()
        result = graphrag_answer(query)
        elapsed = time.time() - start

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱ Time",        f"{elapsed:.1f}s")
    m2.metric("🔗 Graph Hits", result["graph_hits"])
    m3.metric("🔍 Vector Hits",result["vector_hits"])
    m4.metric("📄 Sources",    len(result["sources"]))
    st.divider()

    col_ans, col_src = st.columns([3,2], gap="large")


    with col_ans:
        st.subheader("📋 Answer")
        if result["sources"]:
            st.success(result["answer"])
        else:
            st.warning(result["answer"])

        entities = result["entities"]
        tags = []
        for key, label in [("sections","Section"),("concepts","Concept"),
                            ("judges","Judge"),("acts","Act")]:
            for val in (entities.get(key) or []):
                tags.append(f"`{label}: {val}`")
        if tags:
            st.caption("Entities detected: " + "  |  ".join(tags))

    with col_src:
        st.subheader("📚 Source Cases")
        for i, case in enumerate(result["sources"], 1):
            icon  = "🔗" if case.get("retrieval_type") == "Graph" else "🔍"
            label = f"{icon} [{i}] {str(case.get('title','?'))[:45]}"
            with st.expander(label):
                st.markdown(f"**Citation:** `{case.get('citation','N/A')}`")
                st.markdown(f"**Year:** {case.get('year','N/A')}")
                st.markdown(f"**Via:** {case.get('retrieval_type','')} — {case.get('graph_path','')}")
                if case.get("summary"):
                    st.markdown(f"**Summary:** {case['summary']}")

elif search_clicked:
    st.warning("Please enter a legal question.")

else:
    st.info("""
    **How this works:**
    1. Your question is parsed to extract legal entities (judges, acts, sections, concepts)
    2. **Graph traversal** finds cases linked to those entities in Neo4j
    3. **Vector search** finds semantically similar cases using embeddings
    4. Results are merged, and an LLM synthesises a cited answer

    👈 Use the sidebar examples to get started, or type your own question above.
    """)

st.subheader("Architecture")
st.code("""
User Query
    │
    ├──► Entity Extraction (Openai/gpt-oss-20b)
    │         judges / acts / sections / concepts
    │
    ├──► Graph Traversal (Neo4j Cypher)
    │         (Case)-[:DECIDED_BY]->(Judge)
    │         (Case)-[:REFERENCES_SECTION]->(Section)
    │         (Case)-[:INVOLVES_CONCEPT]->(LegalConcept)
    │
    ├──► Vector Search (Neo4j vector index)
    │         all-MiniLM-L6-v2 embeddings, cosine similarity
    │
    └──► Merge + LLM Synthesis (Openai/gpt-oss-20b)
              Cited answer with source cases
""", language="text")
