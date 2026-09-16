# Setup & deployment

Two parts: running it locally, then hosting it free on Streamlit Community
Cloud.

---

## 1. Run locally

**Requires Python 3.9–3.12.** (Python 3.13 is fine locally but Streamlit Cloud
tops out at 3.12, so 3.12 is the safe choice.)

```bash
python -m venv .venv
```

Activate it — macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Then install and run:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>.

**The first launch is slow (1–2 minutes).** It downloads two ONNX models
(~220 MB total) and embeds the corpus. Every launch after that reads the cached
index from `.chroma/` and starts in a couple of seconds.

The index rebuilds itself automatically whenever `dataset/*.jsonl`, the
embedding model, or the passage-window settings change — a fingerprint of all
three is stored in `.chroma/build.json`. To force a rebuild, delete `.chroma/`.

---

## 2. Push to GitHub

The repo is already initialised. Streamlit Community Cloud deploys from
GitHub, so you need it there.

```bash
git add -A && git commit -m "RAG retrieval UI"
```

Create an **empty** repo on GitHub (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/<you>/<repo>.git
```

```bash
git branch -M main && git push -u origin main
```

The repo can be public or private — Community Cloud handles both on the free
plan.

### What must be committed

- `app.py`, `rag_core.py`, `ui.py`
- `requirements.txt`
- everything in `dataset/` you want available in the deployed app
- `static/`, if your corpus references figures
- `.streamlit/config.toml` (it turns on static file serving)

`.chroma/` is gitignored on purpose. The cloud container rebuilds it on first
boot; committing it would just ship a stale index.

---

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository** — `<you>/<repo>`
   - **Branch** — `main`
   - **Main file path** — `app.py`
4. Open **Advanced settings** and set **Python version** to **3.12**.
5. **Deploy.**

The first build takes roughly 5–10 minutes: it installs the dependencies, then
the container downloads the models and builds the index on first page load.
Watch the build log in the right-hand panel — the app is live once it says
`You can now view your Streamlit app`.

Every push to `main` redeploys automatically.

### Free-plan limits worth knowing

| Limit | Value | What it means here |
|---|---|---|
| Memory | ~1 GB | Measured ~410 MB after 10 queries in one session. Two things keep it there: ONNX models instead of torch, and `RERANK_BATCH_SIZE` (see Tuning). |
| Private apps | 1 | Public apps are unlimited. |
| Sleep | after ~7 days idle | Anyone can wake it; the next visitor waits for the rebuild. |
| Disk | ephemeral | `.chroma/` is rebuilt after every restart. Fine — it takes seconds. |

### Two things that are already handled

**sqlite3.** Community Cloud ships a system sqlite older than the 3.35 Chroma
requires. `requirements.txt` pulls `pysqlite3-binary` on Linux and the top of
`rag_core.py` swaps it in before Chroma is imported. Don't remove either half —
without them the deploy fails with
`unsupported version of sqlite3`.

**Torch.** Nothing in the dependency tree pulls it. If you add a library that
does, the install will likely blow the free tier's memory and disk.

**Reranker batch size.** `RERANK_BATCH_SIZE = 1` in `rag_core.py`. fastembed's
default of 64 made onnxruntime reserve a workspace sized for the largest batch
it had seen and never release it — a session climbed past 1.9 GB and got
killed after five or six queries. Don't raise it without re-measuring.

**Images.** They are served as URLs from `static/`, not base64-inlined. See
below; inlining them put 22 MB of HTML on the wire across five queries.

---

## 4. Figures

A corpus points at images by relative path, e.g.

```
extracted_images/POLIVY_GLOBAL_BRAND_BOOK_Q1_2026_1/p014/pymupdf_vector_render_ba447c625c.png
```

Images live under **`static/`** and are served by URL, because
`server.enableStaticServing` is on in `.streamlit/config.toml`. Streamlit
exposes that folder at `app/static/...`, so a `local_path` of
`extracted_images/x.png` must exist at `static/extracted_images/x.png`.

`static/extracted_images/` is committed (about 13 MB), so the bundled corpus
renders all 72 of its figures. For a new corpus, drop its image folder inside
`static/` and commit it.

**Only files under `static/` are rendered.** Anything else degrades to a
labelled placeholder — deliberately. Images used to be base64-inlined into the
page, which produced 22 MB of HTML across five queries, regenerated on every
rerun, and was the first thing to push the app over its memory limit.
`MAX_IMAGES_PER_CHUNK` (12) in `ui.py` still caps how many render per chunk.

**Before committing images, check the total repo size.** GitHub warns above
1 GB and Community Cloud clones the whole repo on every build.

---

## 5. Tuning

### Swap the reranker

In `rag_core.py`:

```python
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # ~90 MB  (default)
```

`BAAI/bge-reranker-base` is stronger, but it needs roughly **1.1 GB of RAM and
will OOM the free Community Cloud tier.** Use it only when self-hosting or on a
paid plan. Other fastembed cross-encoders that do fit:
`jinaai/jina-reranker-v1-tiny-en`, `Xenova/ms-marco-MiniLM-L-12-v2`.

Changing the reranker takes effect immediately. Changing `EMBED_MODEL` also
changes the index fingerprint, so the index rebuilds on next start.

### Other knobs

All in `rag_core.py`:

| Constant | Default | Effect |
|---|---|---|
| `PASSAGE_WORDS` / `PASSAGE_OVERLAP` | 180 / 45 | passage window; changing either rebuilds the index |
| `RERANK_TEMPERATURE` | 3.0 | lower = more extreme 0/1 scores |
| `W_RERANK` / `W_DENSE` / `W_COVERAGE` | .60 / .25 / .15 | relevance blend; must sum to 1.0 |
| `SUBSTANCE_FLOOR` / `SUBSTANCE_FULL_AT` | 0.40 / 40 | how hard near-empty sections are penalised |
| `RERANK_BATCH_SIZE` | 1 | query-passage pairs per forward pass. **The main memory control.** Larger is both heavier and slower here — the comment above it has the measurements |

Top-K and minimum relevance are live in the sidebar. The candidate pool,
hybrid retrieval and reranking are always on.

### Use a different dataset

Drop another `.jsonl` into `dataset/` — no code change needed. Every file
there is listed in the sidebar's corpus picker, and each keeps its own Chroma
collection, so switching between them does not force a rebuild.

Only the text is required. It is read from the first present of `chunk_text`,
`text`, `content`, `body`, `passage` or `page_content`. Titles, ids, document
names and page numbers each have their own list of accepted keys and fall back
to sensible defaults; see the field table in [README.md](README.md#any-dataset).
Blank lines, malformed JSON and text-less rows are skipped rather than fatal.

Confirm it before deploying:

```bash
python verify_datasets.py
```

---

## Troubleshooting

**`RuntimeError: Your system has an unsupported version of sqlite3`** — the
`pysqlite3-binary` line is missing from `requirements.txt`, or the shim at the
top of `rag_core.py` was moved below the `chromadb` import. It must run first.

**"This app has gone over its resource limits"** — memory. In order of
likelihood: `RERANK_BATCH_SIZE` was raised, images were moved out of `static/`
back into base64, `RERANK_MODEL` was switched to `BAAI/bge-reranker-base`, or
something pulled torch in. To see where it goes, run the engine outside
Streamlit and print `psutil.Process().memory_info().rss` after each search —
the reranker is the only step that should move it.

**First load times out** — reload the page. The models are downloading; the
second attempt hits a warm cache.

**Stale or wrong results after editing a dataset** — delete `.chroma/` and
restart. Normally the per-collection fingerprint catches this automatically.

**A corpus won't load** — the app names the reason on screen. The usual cause
is that no line carries a recognised text key. Run `python verify_datasets.py`
to see it from the terminal.

**Icons render as words like `keyboard_arrow_right`** — a CSS `font-family`
rule is overriding Streamlit's Material Symbols font. Keep the font rule in
`ui.py` scoped to `html, body, .stApp`; never widen it to `[class*="st-"]`.
