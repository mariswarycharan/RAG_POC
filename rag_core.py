"""Retrieval core for the RAG Retrieval explorer.

Pipeline: sections -> passages -> (dense via Chroma + BM25) -> RRF fusion
-> cross-encoder rerank -> parent-section rollup -> metrics.

There is no generative model anywhere in here: the tool returns the retrieved
evidence and the scores that justify it.
"""

from __future__ import annotations

# --- Streamlit Community Cloud sqlite3 shim (must run before chromadb) ------
import sys

try:  # pragma: no cover - environment dependent
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ---------------------------------------------------------------------------

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PERSIST_DIR = PROJECT_ROOT / ".chroma"


def list_datasets(directory: Path = DATASET_DIR) -> list:
    """Every .jsonl corpus sitting in the dataset folder, newest name first."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.jsonl"), key=lambda p: p.name.lower())


def default_dataset(directory: Path = DATASET_DIR):
    found = list_datasets(directory)
    return found[0] if found else None


def collection_name(path: Path) -> str:
    """A Chroma-safe collection name derived from the file name.

    Each dataset gets its own collection so switching between them in the UI
    never mixes vectors or forces a rebuild.
    """
    slug = re.sub(r"[^a-z0-9_-]+", "_", path.stem.lower()).strip("_-")
    slug = (slug or "corpus")[:48]
    if len(slug) < 3:
        slug = f"ds_{slug}"
    if not slug[0].isalnum():
        slug = f"d{slug}"
    return slug


DATASET_PATH = default_dataset()

# --- Models -----------------------------------------------------------------
# Both run on onnxruntime through fastembed, so torch is never installed. That
# is what keeps the app inside the Streamlit Community Cloud memory budget.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # 384-dim, ~130 MB
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # cross-encoder, ~90 MB
# Swap RERANK_MODEL to "BAAI/bge-reranker-base" for a stronger reranker. It
# needs ~1.1 GB of RAM, so it will OOM the free Community Cloud tier - use it
# only when self-hosting. See SETUP.md.

# How many query-passage pairs the cross-encoder scores per forward pass.
# fastembed defaults to 64, which sent one session to 1.9 GB and got the app
# killed on Community Cloud after a handful of queries: onnxruntime sizes its
# workspace for the largest batch shape it has seen and never gives it back.
#
# Passages here vary from a few words to 180, and a batch pads every pair to
# the longest one in it, so large batches also waste compute. Measured over
# eight queries on this corpus, smaller is both lighter and faster:
#
#     batch  peak RSS  rerank cost  ms/query
#         1    387 MB        65 MB       926
#         4    562 MB       241 MB      1240
#        16   1262 MB       941 MB      1542
#        32   1937 MB      1617 MB      1625
#
# There is no trade-off to balance, so this is 1. Raise it only if you switch
# to a corpus with uniform passage lengths and can re-measure both columns.
RERANK_BATCH_SIZE = 1

# --- Passage windowing ------------------------------------------------------
PASSAGE_WORDS = 180
PASSAGE_OVERLAP = 45

# --- Relevance calibration --------------------------------------------------
# The cross-encoder emits logits in roughly [-11, +9]. A plain sigmoid pins
# almost everything to 0.0 or 1.0, which is useless to read off a UI, so the
# logit is divided by this temperature before squashing.
RERANK_TEMPERATURE = 3.0

# The headline relevance score is a weighted blend rather than the reranker
# alone: on this corpus the reranker occasionally buries a section that dense
# retrieval and keyword overlap both agree on. Weights must sum to 1.0.
W_RERANK, W_DENSE, W_COVERAGE = 0.60, 0.25, 0.15

# Several "sections" are just divider pages carrying two or three words. They
# embed suspiciously well against short queries, so scale their score down by
# how much body text they actually have.
SUBSTANCE_FLOOR = 0.40
SUBSTANCE_FULL_AT = 40  # words needed to earn the full score

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "of", "on", "or", "should",
    "that", "the", "their", "them", "there", "these", "this", "to", "use",
    "used", "using", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\+/#\.]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


# Control characters, soft hyphens and zero-width spaces the PDF extraction
# left behind. Tab, newline and carriage return are deliberately kept.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f­​]")


def clean_text(text: str) -> str:
    """Normalise raw extracted text for display and indexing."""
    return _CONTROL_RE.sub("", text).replace("\x0c", "\n")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """One chunk from the JSONL, i.e. one document section."""

    chunk_id: str
    document_id: str
    section_label: str
    page_start: int
    page_end: int
    source_pages: list
    matched_fields: list
    field_match_details: list
    text: str
    tables: list
    images: list
    color_swatches: list
    fonts_present: list
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def has_pages(self) -> bool:
        """Plenty of corpora carry no page numbers at all."""
        return bool(self.source_pages) or self.page_start > 0

    @property
    def page_label(self) -> str:
        if not self.has_pages:
            return ""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Passage:
    """A retrieval unit: one window of a section."""

    passage_id: str
    parent_id: str
    index: int
    total: int
    text: str


@dataclass
class Hit:
    """A section returned to the UI, with every score that produced it."""

    section: Section
    rank: int
    best_passage: Passage
    dense: float           # raw cosine similarity, 0-1
    dense_norm: float      # cosine min-maxed across the candidate pool
    bm25: float            # BM25 min-maxed across the candidate pool
    fusion: float          # normalised reciprocal-rank fusion score
    rerank: float          # cross-encoder logit squashed to 0-1
    rerank_logit: float    # the raw cross-encoder output
    keyword_coverage: float
    substance: float       # penalises near-empty divider sections
    relevance: float       # the composite headline score
    dense_rank: Any
    bm25_rank: Any
    fusion_rank: int
    matched_terms: list
    passages_hit: int


@dataclass
class RetrievalResult:
    query: str
    hits: list
    metrics: dict
    timings: dict
    trace: list
    params: dict


# ---------------------------------------------------------------------------
# Loading + windowing
# ---------------------------------------------------------------------------

# Field aliases, so a JSONL written by a different pipeline still loads. Only
# the chunk text is mandatory; everything else degrades to a sensible default.
_TEXT_KEYS = ("chunk_text", "text", "content", "body", "passage", "page_content")
_LABEL_KEYS = ("section_label", "title", "heading", "section", "name", "header")
_DOC_KEYS = ("document_id", "doc_id", "source", "document", "file_name", "filename")
_ID_KEYS = ("chunk_id", "id", "_id", "uuid", "chunkId")

# Labels this module invented because the record had no title of its own.
_PLACEHOLDER_LABEL = re.compile(r"^Section \d+$")


def _first(record: dict, keys, default=""):
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, "", [], {}) and not isinstance(value, str):
            return value
    return default


def _as_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def load_sections(path: Path = DATASET_PATH) -> list:
    """Read a JSONL corpus into Section objects.

    Lines that are unparseable or carry no text are skipped rather than
    aborting the load, so one bad row cannot take the whole app down.
    """
    if path is None:
        return []

    sections = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict):
                continue

            text = clean_text(str(_first(r, _TEXT_KEYS, "")))
            label = str(_first(r, _LABEL_KEYS, "")) or f"Section {lineno}"
            if not text.strip() and not label.strip():
                continue

            pages = [p for p in _as_list(r.get("source_pages")) if isinstance(p, int)]
            page_start = _as_int(r.get("page_start"), pages[0] if pages else 0)
            page_end = _as_int(r.get("page_end"), pages[-1] if pages else page_start)

            sections.append(
                Section(
                    chunk_id=str(_first(r, _ID_KEYS, f"{path.stem}_{lineno}")),
                    document_id=str(_first(r, _DOC_KEYS, path.stem)),
                    section_label=label,
                    page_start=page_start,
                    page_end=page_end,
                    source_pages=pages,
                    matched_fields=[
                        str(f) for f in _as_list(r.get("matched_fields")) if f
                    ],
                    field_match_details=_as_list(r.get("field_match_details")),
                    text=text,
                    tables=[t for t in _as_list(r.get("tables")) if isinstance(t, dict)],
                    images=[i for i in _as_list(r.get("images")) if isinstance(i, dict)],
                    color_swatches=_as_list(r.get("color_swatches")),
                    fonts_present=[str(f) for f in _as_list(r.get("fonts_present")) if f],
                    raw=r,
                )
            )

    # Duplicate ids would silently overwrite each other in Chroma.
    seen = {}
    for sec in sections:
        seen[sec.chunk_id] = seen.get(sec.chunk_id, 0) + 1
        if seen[sec.chunk_id] > 1:
            sec.chunk_id = f"{sec.chunk_id}__{seen[sec.chunk_id]}"
    return sections


def build_passages(sections: Iterable) -> list:
    """Sliding-window each section so nothing overflows the embedder's context.

    Most sections are short and stay whole; only the long ones split.
    """
    passages = []
    for sec in sections:
        words = sec.text.split()
        if not words:
            words = [sec.section_label]
        windows = []
        if len(words) <= PASSAGE_WORDS:
            windows.append(" ".join(words))
        else:
            step = PASSAGE_WORDS - PASSAGE_OVERLAP
            for start in range(0, len(words), step):
                window = words[start:start + PASSAGE_WORDS]
                if len(window) < 30 and windows:
                    break
                windows.append(" ".join(window))
                if start + PASSAGE_WORDS >= len(words):
                    break
        # Tables carry meaning the prose does not; index them as their own units.
        for tbl in sec.tables:
            md = (tbl.get("markdown") or "").strip()
            if md:
                windows.append("Table on page {}\n{}".format(tbl.get("page_number", "?"), md))
        total = len(windows)
        for i, w in enumerate(windows):
            passages.append(
                Passage(
                    passage_id=f"{sec.chunk_id}::{i}",
                    parent_id=sec.chunk_id,
                    index=i,
                    total=total,
                    # Prefix the heading so short windows keep their context.
                    text=f"{sec.section_label}\n{w}",
                )
            )
    return passages


def dataset_fingerprint(path: Path = DATASET_PATH) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(f"{EMBED_MODEL}|{PASSAGE_WORDS}|{PASSAGE_OVERLAP}|v3".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _minmax(values: dict) -> dict:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: (1.0 if hi > 0 else 0.0) for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _sigmoid(x: float, temperature: float = 1.0) -> float:
    z = max(-30.0, min(30.0, x / temperature))
    return 1.0 / (1.0 + math.exp(-z))


def _substance(word_count: int) -> float:
    """1.0 for a real section, down to SUBSTANCE_FLOOR for a divider page."""
    filled = min(1.0, word_count / SUBSTANCE_FULL_AT)
    return SUBSTANCE_FLOOR + (1.0 - SUBSTANCE_FLOOR) * filled


def _rrf(rank, k: int = 60) -> float:
    return 0.0 if rank is None else 1.0 / (k + rank)


def _collection_names(client) -> set:
    """Chroma returns names in 0.6+, collection objects before that."""
    out = set()
    for c in client.list_collections():
        out.add(c if isinstance(c, str) else getattr(c, "name", str(c)))
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RagEngine:
    """Owns the models, the Chroma collection and the BM25 index."""

    def __init__(self, dataset_path: Path = DATASET_PATH, persist_dir: Path = PERSIST_DIR):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from rank_bm25 import BM25Okapi
        import chromadb
        from chromadb.config import Settings

        if dataset_path is None:
            raise FileNotFoundError(
                f"No .jsonl corpus found in {DATASET_DIR}. Drop one in and reload."
            )

        self.dataset_path = Path(dataset_path)
        self.collection_name = collection_name(self.dataset_path)
        self.sections = load_sections(self.dataset_path)
        if not self.sections:
            raise ValueError(
                f"{self.dataset_path.name} has no usable records. Each line needs "
                f"text under one of: {', '.join(_TEXT_KEYS)}."
            )
        self.by_id = {s.chunk_id: s for s in self.sections}
        self.passages = build_passages(self.sections)
        self.passage_by_id = {p.passage_id: p for p in self.passages}
        self.built_fresh = False

        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.reranker = TextCrossEncoder(model_name=RERANK_MODEL)

        self.bm25 = BM25Okapi([content_terms(p.text) for p in self.passages])
        self._bm25_ids = [p.passage_id for p in self.passages]

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self._ensure_collection(persist_dir)

    # -- index -------------------------------------------------------------
    def _ensure_collection(self, persist_dir: Path):
        name = self.collection_name
        fingerprint = dataset_fingerprint(self.dataset_path)
        stamp_file = persist_dir / "build.json"
        stamp = {}
        if stamp_file.exists():
            try:
                loaded = json.loads(stamp_file.read_text(encoding="utf-8"))
                stamp = loaded if isinstance(loaded, dict) else {}
            except Exception:
                stamp = {}

        existing = _collection_names(self.client)
        recorded = stamp.get(name, {}) if isinstance(stamp.get(name), dict) else {}

        if recorded.get("fingerprint") == fingerprint and name in existing:
            col = self.client.get_collection(name)
            if col.count() == len(self.passages):
                return col
            self.client.delete_collection(name)
        elif name in existing:
            self.client.delete_collection(name)

        col = self.client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        vectors = list(self.embedder.embed([p.text for p in self.passages]))
        col.add(
            ids=[p.passage_id for p in self.passages],
            embeddings=[v.tolist() for v in vectors],
            documents=[p.text for p in self.passages],
            metadatas=[
                {
                    "parent_id": p.parent_id,
                    "index": p.index,
                    "total": p.total,
                    "section": self.by_id[p.parent_id].section_label,
                    "page_start": self.by_id[p.parent_id].page_start,
                    "page_end": self.by_id[p.parent_id].page_end,
                }
                for p in self.passages
            ],
        )
        stamp[name] = {"fingerprint": fingerprint, "passages": len(self.passages)}
        stamp_file.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
        self.built_fresh = True
        return col

    # -- stats -------------------------------------------------------------
    def corpus_name(self) -> str:
        """A readable title for the corpus, whatever the dataset looks like."""
        docs = {s.document_id for s in self.sections if s.document_id}
        if len(docs) == 1:
            raw = docs.pop()
        else:
            raw = self.dataset_path.stem
        pretty = re.sub(r"[_\-]+", " ", str(raw)).strip()
        pretty = re.sub(r"\s+", " ", pretty)
        return pretty.title() if pretty.islower() or pretty.isupper() else pretty

    def suggested_queries(self, limit: int = 6) -> list:
        """Seed the UI with real headings from whichever corpus loaded.

        Corpora with no titles get a placeholder label ("Section 4"), which
        makes a useless suggestion, so those fall back to an opening phrase
        from the chunk itself.
        """
        seen, out = set(), []
        for sec in sorted(self.sections, key=lambda s: -s.word_count):
            label = sec.section_label.strip()
            if _PLACEHOLDER_LABEL.match(label):
                words = sec.text.split()[:7]
                label = " ".join(words).rstrip(".,;:")
            if not (6 <= len(label) <= 52):
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(label)
            if len(out) >= limit:
                break
        return out

    def stats(self) -> dict:
        pages = {p for s in self.sections for p in (s.source_pages or [])}
        return {
            "name": self.corpus_name(),
            "file": self.dataset_path.name,
            "documents": len({s.document_id for s in self.sections if s.document_id}),
            "sections": len(self.sections),
            "passages": len(self.passages),
            "tables": sum(len(s.tables) for s in self.sections),
            "images": sum(len(s.images) for s in self.sections),
            "words": sum(s.word_count for s in self.sections),
            "pages": len(pages),
            "embed_model": EMBED_MODEL,
            "rerank_model": RERANK_MODEL,
            "dim": 384,
        }

    # -- search ------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates: int = 25,
        use_bm25: bool = True,
        use_rerank: bool = True,
        min_relevance: float = 0.0,
        alpha: float = 0.6,
        on_step=None,
    ) -> RetrievalResult:
        """Run the full pipeline. `on_step(name, detail)` streams progress."""

        def step(name, detail):
            if on_step:
                on_step(name, detail)

        timings = {}
        trace = []
        q_terms = content_terms(query)

        # 1. embed query -----------------------------------------------------
        t0 = time.perf_counter()
        step("Embedding query", f"{EMBED_MODEL} · 384-dim")
        qvec = list(self.embedder.query_embed([query]))[0]
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000
        trace.append({
            "label": "Embed query",
            "detail": f"{EMBED_MODEL} -> 384-dim vector",
            "ms": timings["embed_ms"],
        })

        # 2. dense search ----------------------------------------------------
        t0 = time.perf_counter()
        step("Dense vector search", f"Chroma · top {candidates} passages")
        n = min(candidates, len(self.passages))
        res = self.collection.query(
            query_embeddings=[qvec.tolist()],
            n_results=n,
            include=["distances", "metadatas"],
        )
        dense_ids = res["ids"][0]
        dense_scores = {
            pid: max(0.0, 1.0 - float(d))
            for pid, d in zip(dense_ids, res["distances"][0])
        }
        dense_ranks = {pid: i + 1 for i, pid in enumerate(dense_ids)}
        dense_norm = _minmax(dense_scores)
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000
        trace.append({
            "label": "Dense search (Chroma)",
            "detail": f"cosine over {len(self.passages)} passages -> {len(dense_ids)} candidates",
            "ms": timings["dense_ms"],
        })

        # 3. lexical search --------------------------------------------------
        bm25_scores = {}
        bm25_ranks = {}
        if use_bm25 and q_terms:
            t0 = time.perf_counter()
            step("Lexical BM25 search", f"{len(q_terms)} content terms")
            raw = self.bm25.get_scores(q_terms)
            order = np.argsort(raw)[::-1][:n]
            top = [(self._bm25_ids[i], float(raw[i])) for i in order if raw[i] > 0]
            bm25_scores = _minmax(dict(top))
            bm25_ranks = {pid: i + 1 for i, (pid, _) in enumerate(top)}
            timings["bm25_ms"] = (time.perf_counter() - t0) * 1000
            trace.append({
                "label": "Lexical search (BM25)",
                "detail": "terms: {} -> {} candidates".format(
                    ", ".join(q_terms[:8]) or "none", len(top)
                ),
                "ms": timings["bm25_ms"],
            })

        # 4. fusion ----------------------------------------------------------
        t0 = time.perf_counter()
        step("Reciprocal-rank fusion", "merging dense + lexical rankings")
        pool = set(dense_scores) | set(bm25_scores)
        fused_raw = {
            pid: alpha * _rrf(dense_ranks.get(pid)) + (1 - alpha) * _rrf(bm25_ranks.get(pid))
            for pid in pool
        }
        fused = _minmax(fused_raw)
        fusion_order = sorted(pool, key=lambda p: fused_raw[p], reverse=True)
        fusion_ranks = {pid: i + 1 for i, pid in enumerate(fusion_order)}
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000
        trace.append({
            "label": "Reciprocal-rank fusion",
            "detail": f"{len(dense_scores)} dense + {len(bm25_scores)} lexical -> {len(pool)} unique passages",
            "ms": timings["fusion_ms"],
        })

        # 5. rerank ----------------------------------------------------------
        shortlist = fusion_order[: min(len(fusion_order), max(candidates, top_k * 4))]
        if use_rerank and shortlist:
            t0 = time.perf_counter()
            step("Cross-encoder rerank", f"scoring {len(shortlist)} query-passage pairs")
            docs = [self.passage_by_id[pid].text for pid in shortlist]
            logits = {
                pid: float(s)
                for pid, s in zip(
                    shortlist,
                    self.reranker.rerank(query, docs, batch_size=RERANK_BATCH_SIZE),
                )
            }
            rerank_scores = {
                pid: _sigmoid(v, RERANK_TEMPERATURE) for pid, v in logits.items()
            }
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            trace.append({
                "label": "Cross-encoder rerank",
                "detail": f"{RERANK_MODEL} scored {len(shortlist)} pairs",
                "ms": timings["rerank_ms"],
            })
        else:
            logits = {pid: 0.0 for pid in shortlist}
            rerank_scores = {pid: fused.get(pid, 0.0) for pid in shortlist}

        # 6. roll passages up to their parent section ------------------------
        step("Grouping into sections", "best passage wins per section")
        grouped = {}
        for pid in shortlist:
            grouped.setdefault(self.passage_by_id[pid].parent_id, []).append(pid)

        prelim = []
        for parent_id, pids in grouped.items():
            best = max(pids, key=lambda p: (rerank_scores.get(p, 0.0), fused.get(p, 0.0)))
            sec = self.by_id[parent_id]
            passage = self.passage_by_id[best]
            sec_terms = set(content_terms(sec.text + " " + sec.section_label))
            matched = [t for t in dict.fromkeys(q_terms) if t in sec_terms]
            coverage = len(matched) / len(set(q_terms)) if q_terms else 0.0
            rr = rerank_scores.get(best, 0.0)
            fu = fused.get(best, 0.0)
            dn = dense_norm.get(best, 0.0)
            substance = _substance(sec.word_count)
            if use_rerank:
                signal = W_RERANK * rr + W_DENSE * dn + W_COVERAGE * coverage
            else:
                # No reranker: lean on the fused ranking instead.
                signal = (W_RERANK + W_DENSE) * fu + W_COVERAGE * coverage
            prelim.append(
                Hit(
                    section=sec,
                    rank=0,
                    best_passage=passage,
                    dense=dense_scores.get(best, 0.0),
                    dense_norm=dn,
                    bm25=bm25_scores.get(best, 0.0),
                    fusion=fu,
                    rerank=rr,
                    rerank_logit=logits.get(best, 0.0),
                    keyword_coverage=coverage,
                    substance=substance,
                    relevance=signal * substance,
                    dense_rank=dense_ranks.get(best),
                    bm25_rank=bm25_ranks.get(best),
                    fusion_rank=fusion_ranks.get(best, 0),
                    matched_terms=matched,
                    passages_hit=len(pids),
                )
            )

        prelim.sort(key=lambda h: h.relevance, reverse=True)
        kept = [h for h in prelim if h.relevance >= min_relevance][:top_k]
        for i, h in enumerate(kept, start=1):
            h.rank = i

        timings["total_ms"] = sum(v for k, v in timings.items() if k.endswith("_ms"))
        trace.append({
            "label": "Section rollup",
            "detail": f"{len(grouped)} sections from {len(shortlist)} passages -> top {len(kept)} returned",
            "ms": 0.0,
        })

        return RetrievalResult(
            query=query,
            hits=kept,
            metrics=self._summarise(kept, prelim, len(shortlist)),
            timings=timings,
            trace=trace,
            params={
                "top_k": top_k,
                "candidates": candidates,
                "use_bm25": use_bm25,
                "use_rerank": use_rerank,
                "min_relevance": min_relevance,
                "alpha": alpha,
            },
        )

    # -- query-level metrics ------------------------------------------------
    @staticmethod
    def _summarise(kept: list, prelim: list, scanned: int) -> dict:
        if not kept:
            return {
                "top_relevance": 0.0, "mean_relevance": 0.0, "margin": 0.0,
                "coverage": 0.0, "agreement": 0.0, "rank_shift": 0,
                "scanned": scanned, "returned": 0, "candidates_grouped": len(prelim),
                "confidence": "No match",
            }
        rels = [h.relevance for h in kept]
        top = rels[0]
        margin = top - rels[1] if len(rels) > 1 else top
        coverage = float(np.mean([h.keyword_coverage for h in kept]))
        # How often the dense and lexical retrievers agreed on the same passage.
        both = sum(1 for h in kept if h.dense_rank and h.bm25_rank)
        agreement = both / len(kept)
        rank_shift = sum(abs(h.fusion_rank - h.rank) for h in kept)

        if top >= 0.75 and margin >= 0.10:
            conf = "High"
        elif top >= 0.45:
            conf = "Moderate"
        elif top >= 0.20:
            conf = "Low"
        else:
            conf = "Very low"

        return {
            "top_relevance": top,
            "mean_relevance": float(np.mean(rels)),
            "margin": margin,
            "coverage": coverage,
            "agreement": agreement,
            "rank_shift": rank_shift,
            "scanned": scanned,
            "returned": len(kept),
            "candidates_grouped": len(prelim),
            "confidence": conf,
        }
