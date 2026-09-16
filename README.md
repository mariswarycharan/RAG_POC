# RAG Retrieval

A single-page Streamlit app that searches a chunked JSONL corpus and shows you
the evidence it found, with the scores that justify each result.

There is **no LLM in the loop.** Nothing is generated or summarised — the
"answer" is the retrieved source material itself, rendered cleanly.

```
query ─► bge-small embedding ─┬─► Chroma (cosine, HNSW) ─┐
                              │                          ├─► RRF fusion ─► cross-encoder rerank ─► section rollup ─► report
                              └─► BM25 (lexical) ────────┘
```

## What it does

- **Hybrid retrieval** — dense vector search and BM25 keyword search run in
  parallel and are merged with reciprocal-rank fusion.
- **Cross-encoder reranking** — every surviving candidate is re-scored against
  the query by a cross-encoder. This is the single biggest accuracy win.
- **Parent-section rollup** — long sections are indexed as overlapping
  passages so nothing overflows the embedder's context, then rolled back up so
  you read whole sections, not fragments.
- **A readable report** — the retrieved chunks are the main column. Each one
  shows its full `chunk_text`, then its markdown tables as real tables, its
  figures inline, its colour swatches and the fonts the page used, then its
  retrieval metrics, then a **Show details** drawer with the chunk's
  provenance (pages covered, matched fields and how each was scored).

## Any dataset

Drop any `.jsonl` into `dataset/`. Every file there appears in the sidebar's
corpus picker, each gets its own Chroma collection, and the corpus title,
statistics and suggested queries are all derived from whatever loaded — there
is nothing corpus-specific in the code.

**The only required field is the text.** It is read from the first of these
keys that is present:

| Field | Accepted keys | Missing? |
|---|---|---|
| text | `chunk_text`, `text`, `content`, `body`, `passage`, `page_content` | row is skipped |
| title | `section_label`, `title`, `heading`, `section`, `name`, `header` | `Section <n>` |
| id | `chunk_id`, `id`, `_id`, `uuid`, `chunkId` | generated from file + line |
| document | `document_id`, `doc_id`, `source`, `document`, `file_name`, `filename` | the file stem |
| pages | `page_start`, `page_end`, `source_pages` | page labels are hidden |
| extras | `tables`, `images`, `color_swatches`, `matched_fields`, `fonts_present` | simply not rendered |

Blank lines, malformed JSON and rows with no text are skipped rather than
aborting the load, and duplicate ids are suffixed so they cannot overwrite
each other in the vector store.

To check this on your own data:

```bash
python verify_datasets.py
```

It builds several deliberately mismatched corpora in a temp directory — a
LangChain-style `page_content`/`title` export, a bare `text`-only file, and one
with duplicate ids plus a corrupt line — runs each through the real engine, and
then does the same for everything in `dataset/`. Non-zero exit on any failure.

## Scoring

The headline **relevance** is a composite, not a single model output:

```
signal    = 0.60 × rerank + 0.25 × vector + 0.15 × term-coverage
relevance = signal × substance
```

| Component | What it is |
|---|---|
| `rerank` | cross-encoder logit, divided by a temperature of 3 and squashed to 0–1 |
| `vector` | Chroma cosine similarity, min-maxed across the candidate pool |
| `term coverage` | fraction of the query's content words present in the section |
| `substance` | 0.4–1.0, scaling down sections that are near-empty divider pages |

**Why a blend and not just the reranker.** On the bundled corpus the reranker
occasionally buries a section that both the vector search and keyword overlap
agree on, and a plain sigmoid over its logits pins almost everything to 0.0 or
1.0 — useless to read off a dashboard. The blend keeps the reranker dominant
while staying legible. Every component is displayed separately on each card, so
the composite is never a black box.

Query-level metrics — top relevance, mean, score margin, retriever agreement,
term coverage, latency, passages scored — sit at the top of the report.

> These are **retrieval confidence signals, not measured accuracy.** There
> are no ground-truth relevance labels for these corpora, so nothing here is a
> precision or recall figure. Treat them as a way to tell a confident hit from
> a guess.

## Stack

| Piece | Choice | Why |
|---|---|---|
| UI | Streamlit | single file, deploys free |
| Vector store | Chroma (persistent, cosine) | local, no service to run |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | strong for its size |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` | ~90 MB, fits the free tier |
| Lexical | `rank-bm25` | pure Python, no index server |
| Runtime | `fastembed` → onnxruntime | **no torch**, which is what keeps the app inside the 1 GB free-tier budget |

Measured footprint: **~410 MB after 10 queries** in one session. Two settings
do most of that work — images served from `static/` rather than base64, and
`RERANK_BATCH_SIZE = 1`. Both are explained in [SETUP.md](SETUP.md#5-tuning);
raising either will put the app back over the free tier's limit.

## Layout

| File | Role |
|---|---|
| `app.py` | Streamlit page: sidebar, chat column, right-hand panel |
| `rag_core.py` | loading, passage windowing, Chroma index, retrieval, scoring |
| `ui.py` | theme CSS and the HTML for a rendered chunk |
| `dataset/*.jsonl` | your corpora - every file here shows up in the app |
| `static/` | figures, served by URL at `app/static/...` |
| `verify_datasets.py` | proves the pipeline works on unfamiliar schemas |
| `SETUP.md` | local setup and Streamlit Community Cloud deployment |

## Run it

```bash
pip install -r requirements.txt && streamlit run app.py
```

First launch downloads the two ONNX models (~220 MB) and builds the index.
Later launches reuse the cache in `.chroma/`, which is rebuilt automatically
whenever the dataset or the embedding settings change.

Full instructions, including free-tier deployment, are in [SETUP.md](SETUP.md).

## Figures

Images live under `static/`, which Streamlit serves at `app/static/...`, and
are referenced by URL. A `local_path` of `extracted_images/x.png` in the JSONL
resolves to `static/extracted_images/x.png`. The bundled corpus ships all 72
of its figures, each labelled with its page, classification and source path,
and each opens full screen on click.

Anything not under `static/` degrades to a labelled placeholder. That is
deliberate: base64-inlining images instead produced 22 MB of HTML across five
queries and was enough on its own to exhaust the free tier. See SETUP.md.
