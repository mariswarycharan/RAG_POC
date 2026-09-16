"""RAG Retrieval — a retrieval explorer for any chunked JSONL corpus.

A single-page Streamlit app over a Chroma index. There is no LLM in the loop:
you ask, it retrieves, reranks and shows you the evidence with the scores that
justify it.

Point it at any .jsonl in dataset/ - the schema is discovered per file.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import ui
from rag_core import DATASET_DIR, RagEngine, list_datasets

st.set_page_config(
    page_title="RAG Retrieval",
    page_icon=":material/search:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine(dataset_path: str) -> RagEngine:
    """One engine per dataset file, cached across reruns."""
    return RagEngine(dataset_path)


def init_state():
    st.session_state.setdefault("messages", [])   # chat transcript
    st.session_state.setdefault("results", [])    # RetrievalResult per answer
    st.session_state.setdefault("viewing", None)  # index into results
    st.session_state.setdefault("pending", None)  # query awaiting execution


init_state()

DATASETS = list_datasets()
if not DATASETS:
    st.error(
        f"No `.jsonl` corpus found in `{DATASET_DIR}`. "
        "Drop one in and reload — every line needs a text field "
        "(`chunk_text`, `text`, `content`, `body`, `passage` or `page_content`)."
    )
    st.stop()

st.session_state.setdefault("dataset", str(DATASETS[0]))
if st.session_state.dataset not in {str(p) for p in DATASETS}:
    st.session_state.dataset = str(DATASETS[0])

try:
    with st.spinner("Loading the embedding model and building the vector index…"):
        engine = get_engine(st.session_state.dataset)
except Exception as exc:  # a malformed corpus should explain itself, not 500
    st.error(f"Could not load **{st.session_state.dataset}**\n\n{exc}")
    st.stop()

stats = engine.stats()
SAMPLE_QUERIES = engine.suggested_queries()


# ---------------------------------------------------------------------------
# Sidebar — menu, controls, sample queries
# ---------------------------------------------------------------------------

def reset_conversation():
    st.session_state.messages = []
    st.session_state.results = []
    st.session_state.viewing = None


def init_controls():
    for key, value in {"top_k": 5, "min_relevance": 0.10}.items():
        st.session_state.setdefault(key, value)


init_controls()


def render_sidebar():
    st.markdown(
        '<div class="sb-brand"><div class="sb-mark">RAG</div>'
        '<div><div class="sb-title">RAG Retrieval</div>'
        '<div class="sb-sub">Hybrid search · no LLM</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-label">Menu</div>', unsafe_allow_html=True)
    if st.button("＋  New search", use_container_width=True):
        reset_conversation()
        st.rerun()

    if len(DATASETS) > 1:
        names = [p.name for p in DATASETS]
        current = names.index(Path(st.session_state.dataset).name)
        chosen = st.selectbox("Corpus", names, index=current,
                              label_visibility="collapsed")
        if chosen != names[current]:
            st.session_state.dataset = str(DATASETS[names.index(chosen)])
            reset_conversation()
            st.rerun()

    st.markdown('<div class="sb-label">Controls</div>', unsafe_allow_html=True)
    st.slider("Top K", 1, 10, key="top_k",
              help="How many document chunks end up in the report.")
    st.slider("Minimum relevance", 0.0, 1.0, step=0.05, key="min_relevance",
              help="Chunks scoring below this are dropped from the report.")

    st.markdown('<div class="sb-label">Try a query</div>', unsafe_allow_html=True)
    for i, q in enumerate(SAMPLE_QUERIES):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()


with st.sidebar:
    render_sidebar()


# ---------------------------------------------------------------------------
# Layout — retrieved chunks take the main column, the chat sits beside it
# ---------------------------------------------------------------------------

report_col, chat_col = st.columns([2.45, 1], gap="large")


# ---------------------------------------------------------------------------
# Main column — the retrieved chunks
# ---------------------------------------------------------------------------

def render_report():
    idx = st.session_state.viewing
    if idx is None or idx >= len(st.session_state.results):
        render_empty_report()
        return
    result = st.session_state.results[idx]

    st.markdown(
        f'<div class="panel-head"><div class="t">Retrieved chunks</div>'
        f'<span class="sub">{len(result.hits)} of '
        f'{result.metrics["candidates_grouped"]} considered</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="note">Query · “{ui.esc(result.query)}”</div>',
                unsafe_allow_html=True)
    st.markdown(ui.summary_html(result.metrics, result.timings), unsafe_allow_html=True)

    if len(st.session_state.results) > 1:
        labels = [f"{i + 1}. {r.query[:44]}" for i, r in enumerate(st.session_state.results)]
        picked = st.selectbox("Report", labels, index=idx, label_visibility="collapsed")
        new_idx = labels.index(picked)
        if new_idx != idx:
            st.session_state.viewing = new_idx
            st.rerun()

    if not result.hits:
        st.markdown(
            '<div class="empty"><span class="ic">◎</span><b>Nothing cleared the threshold</b>'
            "Lower the minimum relevance in the sidebar, or rephrase the query.</div>",
            unsafe_allow_html=True,
        )
        return

    for hit in result.hits:
        st.markdown(ui.chunk_card_html(hit), unsafe_allow_html=True)

    with st.expander("Pipeline timings"):
        for stage in result.trace:
            st.markdown(
                f'<div class="trace-row"><span class="d"></span>'
                f'<span class="l">{ui.esc(stage["label"])}</span>'
                f'<span class="x">{stage["detail"]}</span>'
                f'<span class="t">{stage["ms"]:.0f} ms</span></div>',
                unsafe_allow_html=True,
            )


def render_empty_report():
    st.markdown(
        '<div class="page-head"><div class="t">RAG Retrieval</div>'
        '<span class="tag">retrieval only</span></div>'
        '<p class="page-sub">Hybrid vector + keyword search with cross-encoder '
        "reranking. Answers are the source chunks themselves — nothing is "
        "generated.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="empty"><span class="ic">◆</span>'
        f'<b>Ask {ui.esc(stats["name"])} something</b>'
        f'{stats["sections"]} chunks are indexed as {stats["passages"]} passages. '
        f"Retrieved chunks appear here in full — text, tables, images, colours "
        f"and fonts — with their scores. Use the chat on the right.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Centre column — conversation
# ---------------------------------------------------------------------------

def answer_summary_html(result) -> str:
    if not result.hits:
        return (
            '<div class="bubble-bot"><div class="headline">No chunk passed the '
            "relevance threshold.</div>Try rephrasing, or lower the minimum "
            "relevance in the sidebar.</div>"
        )
    m = result.metrics
    items = "".join(
        f'<li><span class="res-n">{h.rank}</span>'
        f'<span class="res-t">{ui.esc(h.section.section_label)}</span>'
        + (f'<span class="res-p">{ui.esc(h.section.page_label)}</span>'
           if h.section.page_label else "")
        + f'<span class="res-s">{ui.pct(h.relevance)}</span></li>'
        for h in result.hits
    )
    return (
        f'<div class="bubble-bot">'
        f'<div class="headline">{len(result.hits)} chunks retrieved · '
        f'{m["confidence"].lower()} confidence</div>'
        f"Scored {m['scanned']} passages in {result.timings['total_ms']:.0f} ms. "
        f"Top match {ui.pct(m['top_relevance'])}, mean {ui.pct(m['mean_relevance'])}."
        f'<ul class="res-list">{items}</ul></div>'
    )


def replay_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="row-user"><div class="bubble-user">'
                            f'{ui.esc(msg["text"])}</div></div>',
                            unsafe_allow_html=True)
        else:
            result = st.session_state.results[msg["result"]]
            with st.chat_message("assistant"):
                with st.expander(f"Thought for {result.timings['total_ms']:.0f} ms"):
                    for stage in result.trace:
                        st.markdown(
                            f'<div class="trace-row"><span class="d"></span>'
                            f'<span class="l">{ui.esc(stage["label"])}</span>'
                            f'<span class="x">{stage["detail"]}</span></div>',
                            unsafe_allow_html=True,
                        )
                st.markdown(answer_summary_html(result), unsafe_allow_html=True)
                if msg["result"] != st.session_state.viewing:
                    if st.button("Show these chunks ◂", key=f"view_{msg['result']}"):
                        st.session_state.viewing = msg["result"]
                        st.rerun()


def run_query(query: str):
    """Execute one retrieval, streaming the pipeline steps as they happen."""
    with st.chat_message("user"):
        st.markdown(f'<div class="row-user"><div class="bubble-user">'
                    f'{ui.esc(query)}</div></div>',
                    unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=True) as status:
            def on_step(label, detail):
                st.markdown(
                    f'<div class="trace-row"><span class="d"></span>'
                    f'<span class="l">{ui.esc(label)}</span>'
                    f'<span class="x">{ui.esc(detail)}</span></div>',
                    unsafe_allow_html=True,
                )

            # Candidate pool, hybrid retrieval and reranking are always on;
            # only Top K and the relevance floor are exposed in the UI.
            result = engine.search(
                query,
                top_k=st.session_state.top_k,
                min_relevance=st.session_state.min_relevance,
                on_step=on_step,
            )
            status.update(
                label=f"Thought for {result.timings['total_ms']:.0f} ms",
                state="complete",
                expanded=False,
            )
        st.markdown(answer_summary_html(result), unsafe_allow_html=True)

    st.session_state.results.append(result)
    idx = len(st.session_state.results) - 1
    st.session_state.messages.append({"role": "user", "text": query})
    st.session_state.messages.append({"role": "assistant", "result": idx})
    st.session_state.viewing = idx
    trim_history()


# A session keeps every past result so you can flip between reports. Cap it so
# a long session cannot grow without bound.
MAX_KEPT_RESULTS = 15


def trim_history():
    """Drop the oldest searches once the session exceeds the cap."""
    drop = len(st.session_state.results) - MAX_KEPT_RESULTS
    if drop <= 0:
        return
    st.session_state.results = st.session_state.results[drop:]
    kept = []
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            if msg["result"] < drop:
                if kept and kept[-1]["role"] == "user":
                    kept.pop()          # drop its question too
                continue
            msg = {"role": "assistant", "result": msg["result"] - drop}
        kept.append(msg)
    st.session_state.messages = kept
    st.session_state.viewing = len(st.session_state.results) - 1


# The report renders first so it owns the top of the page, but the chat has to
# run before it - that is where a pending query is executed and appended.
with chat_col:
    st.markdown(
        '<div class="panel-head"><div class="t">Chat</div>'
        f'<span class="sub">{len(st.session_state.results)} search(es)</span></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages and not st.session_state.pending:
        st.markdown(
            '<div class="empty"><span class="ic">◆</span><b>Ask a question</b>'
            "Pick a suggestion in the sidebar, or type below. Retrieved chunks "
            "appear on the left.</div>",
            unsafe_allow_html=True,
        )

    replay_history()

    pending = st.session_state.pending
    if pending:
        st.session_state.pending = None
        run_query(pending)
        st.rerun()

with report_col:
    render_report()

typed = st.chat_input("Ask your documents…")
if typed:
    st.session_state.pending = typed
    st.rerun()
