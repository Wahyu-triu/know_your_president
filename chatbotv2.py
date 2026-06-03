"""
RAG Chatbot - Pidato Presiden RI
Streamlit app that uses FAISS vector store as knowledge base

Requirements:
    pip install streamlit langchain langchain-google-genai langchain-community
                faiss-cpu pandas dateparser uuid

Run:
    streamlit run chatbot_rag.py
"""

import time
import faiss
import os
import pandas as pd
import dateparser
import streamlit as st

from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Know-Your-President Chatbot",
    page_icon="🇮🇩",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #c0392b 0%, #8B0000 100%);
        color: white;
    }
    [data-testid="stSidebar"] *:not(input):not(textarea):not(select):not(option) {
        color: white !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select {
        color: black !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #FFD700 !important; }

    /* Chat bubbles */
    .user-bubble {
        background: #c0392b;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 20%;
        text-align: right;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .assistant-bubble {
        background: #f5f0eb;
        color: #1a1a1a;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 20% 8px 0;
        font-size: 0.95rem;
        line-height: 1.6;
        border-left: 4px solid #c0392b;
    }
    .source-tag {
        display: inline-block;
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
        color: #7a5800;
    }
    .chat-container { max-height: 60vh; overflow-y: auto; padding: 8px 0; }
    .stButton > button {
        background: #c0392b;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #8B0000; }
    div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "gemini-embedding-2-preview"
CHAT_MODEL      = "gemini-2.5-flash"
OUTPUT_DIM      = 768
CHUNK_SIZE      = 1500
CHUNK_OVERLAP   = 200
TOP_K           = 4          # how many chunks to retrieve per query
BATCH_SIZE      = 10

SYSTEM_PROMPT = """Kamu adalah asisten yang membantu menjawab pertanyaan tentang pidato-pidato Presiden Republik Indonesia.
Jawab berdasarkan konteks yang diberikan. Jika informasi tidak ada dalam konteks, sampaikan dengan jujur.
Gunakan bahasa yang sopan dan informatif. Sertakan tanggal atau judul pidato jika relevan."""

# ── Helper: build vector store ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_vector_store(csv_path: str, api_key: str):
    """Load CSV, chunk content, embed, and return FAISS vector store."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    df = pd.read_csv(csv_path)

    # Parse dates where possible
    def safe_parse(d):
        try:
            return dateparser.parse(str(d)).strftime("%Y-%m-%d")
        except Exception:
            return str(d)

    df["date"] = df["date"].apply(safe_parse)

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = []
    for _, row in df.iterrows():
        chunks = splitter.split_text(str(row.get("content", "")))
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "title": row.get("title", ""),
                    "date":  row.get("date", ""),
                    "link":  row.get("link", ""),
                },
            ))

    # Build FAISS index
    index = faiss.IndexFlatL2(OUTPUT_DIM)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in documents]
    progress = st.progress(0, text="Menyiapkan knowledge base …")

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs   = documents[i:i + BATCH_SIZE]
        batch_uuids  = uuids[i:i + BATCH_SIZE]
        batch_texts  = [d.page_content for d in batch_docs]
        batch_metas  = [d.metadata for d in batch_docs]

        batch_embeddings = embeddings.embed_documents(
            batch_texts,
            output_dimensionality=OUTPUT_DIM,
        )

        vector_store.add_embeddings(
            text_embeddings=list(zip(batch_texts, batch_embeddings)),
            metadatas=batch_metas,
            ids=batch_uuids,
        )

        progress.progress(
            min((i + BATCH_SIZE) / len(documents), 1.0),
            text=f"Memproses {min(i + BATCH_SIZE, len(documents))}/{len(documents)} chunks …"
        )
        time.sleep(0.3)

    progress.empty()
    return vector_store, embeddings, len(documents)

# ── Load from local vectorstore ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_vector_store(path: str, api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # required for local FAISS loads
    )
    return vector_store, embeddings

# ── Helper: RAG query ─────────────────────────────────────────────────────────

def rag_query(question: str, vector_store, embeddings, chat_history: list, api_key: str):
    """Retrieve relevant chunks, then generate answer with Gemini."""

    # 1. Embed question and retrieve top-K chunks
    q_embedding = embeddings.embed_query(question, output_dimensionality=OUTPUT_DIM)
    results     = vector_store.similarity_search_by_vector(q_embedding, k=TOP_K)

    # 2. Build context string
    context_parts = []
    sources       = []
    for doc in results:
        meta = doc.metadata
        context_parts.append(
            f"[{meta.get('date','')} | {meta.get('title','')}]\n{doc.page_content}"
        )
        sources.append(meta)

    context = "\n\n---\n\n".join(context_parts)

    # 3. Build messages for Gemini
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.3,
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Include recent chat history (last 6 turns)
    for turn in chat_history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    # Final user message with context injected
    messages.append(HumanMessage(content=f"""Konteks dari pidato presiden:
{context}

Pertanyaan: {question}"""))

    response = llm.invoke(messages)
    return response.content, sources


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# Know Your President")
    st.markdown("### RAG Chatbot")
    st.markdown("---")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        # placeholder="AIza...",
        help="Masukkan Google Gemini API key kamu",
    )

    csv_file = st.text_input(
        "Path file CSV",
        value="",
        help="Path ke file CSV hasil scraping",
    )

    load_btn = st.button("🔄 Klik Untuk Set-up", use_container_width=True)

    st.markdown("---")
    st.markdown("**Tentang:**")
    st.markdown("""
            Chatbot ini menggunakan RAG *(Retrieval-Augmented Generation)* untuk menjawab pertanyaan berdasarkan kumpulan pidato Presiden RI dari setneg.go.id.

            **Cara kerja:**
            1. Pidato di-*chunk* dan di-*embed*
            2. Pertanyaanmu di-*embed* lalu dicocokkan
            3. Chunk relevan dikirim ke Gemini sebagai konteks
            4. Gemini menjawab berdasarkan konteks
        """)

    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vector_store"  not in st.session_state: st.session_state.vector_store  = None
if "embeddings"    not in st.session_state: st.session_state.embeddings    = None
if "kb_loaded"     not in st.session_state: st.session_state.kb_loaded     = False
if "chunk_count"   not in st.session_state: st.session_state.chunk_count   = 0
if "pending_query" not in st.session_state: st.session_state.pending_query = None

# ── Helper: send message callback ──────────────────────────────────────────────

def send_message():
    """Callback to handle message sending and clear input."""
    user_input = st.session_state.user_input
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.pending_query = user_input
        st.session_state.user_input = ""

# ── Load knowledge base ───────────────────────────────────────────────────────

if load_btn:
    if not api_key:
        st.error("⚠️ Masukkan Google API Key terlebih dahulu.")
    else:
        try:
            with st.spinner("Membangun knowledge base …"):
                if os.path.exists("faiss_index"):
                    vs, emb = load_vector_store("faiss_index", api_key)
                    st.session_state.vector_store = vs
                    st.session_state.embeddings   = emb
                    st.session_state.kb_loaded    = True
                    st.success(f"✅ Knowledge berhasil di load.")
                else:
                    vs, emb, n = build_vector_store(csv_file, api_key)
                    st.session_state.vector_store = vs
                    st.session_state.embeddings   = emb
                    st.session_state.kb_loaded    = True
                    st.session_state.chunk_count  = n
                    st.success(f"✅ Knowledge base siap! {n} chunks diindeks.")
        except FileNotFoundError:
            st.error(f"❌ File `{csv_file}` tidak ditemukan.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ── Main chat UI ──────────────────────────────────────────────────────────────

st.markdown("## 💬 Know Your President From His Speech")

if not st.session_state.kb_loaded:
    st.info("👈 Masukkan API key dan muat knowledge base terlebih dahulu dari sidebar.")
else:
    st.caption(f"Knowledge base aktif· Model: {CHAT_MODEL}")

    # Process pending query before rendering chat
    if st.session_state.pending_query:
        with st.spinner("Mencari & menghasilkan jawaban …"):
            try:
                answer, sources = rag_query(
                    question=st.session_state.pending_query,
                    vector_store=st.session_state.vector_store,
                    embeddings=st.session_state.embeddings,
                    chat_history=st.session_state.chat_history[:-1],
                    api_key=api_key,
                )
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"❌ Error saat query: {e}")
        st.session_state.pending_query = None

    # Render chat history
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {turn["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">🤖 {turn["content"]}</div>', unsafe_allow_html=True)
            if turn.get("sources"):
                cols = st.columns(len(turn["sources"]))
                for col, src in zip(cols, turn["sources"]):
                    with col:
                        label = f"📄 {src.get('date','')}"
                        title = src.get("title", "")
                        link  = src.get("link", "#")
                        st.markdown(
                            f'<a href="{link}" target="_blank" class="source-tag" title="{title}">{label}</a>',
                            unsafe_allow_html=True,
                        )

    # Input area
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input(
            "Pertanyaan",
            placeholder="Contoh: Apa yang disampaikan Presiden tentang ekonomi?",
            label_visibility="collapsed",
            key="user_input",
            on_change=send_message,
        )
    with col2:
        st.button("Kirim ➤", use_container_width=True, on_click=send_message)