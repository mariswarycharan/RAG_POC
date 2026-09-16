"""Presentation layer: theme CSS plus the HTML that renders a retrieved chunk.

Each chunk is rendered as a single self-contained HTML card so the prose,
tables, images and metrics stay visually welded together. Markdown tables are
converted to HTML here rather than handed to st.markdown, which lets the card
own its own scrolling and typography.
"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Streamlit serves ./static at the URL prefix below when
# server.enableStaticServing is on. Figures are referenced by URL from there
# rather than base64-inlined into the page: inlining 13 MB of PNGs produced
# multi-megabyte HTML on every rerun, which is what pushed the app over the
# Community Cloud memory limit after a handful of queries.
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_URL = "app/static"

# Image paths from the dataset are resolved against these, in order.
IMAGE_ROOTS = [STATIC_DIR, PROJECT_ROOT, PROJECT_ROOT / "dataset", PROJECT_ROOT / "assets"]

MAX_IMAGES_PER_CHUNK = 12


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

CSS = """
<style>
:root{
  --blue-800:#083E7D; --blue-700:#0B4F9E; --blue-600:#0A66C2; --blue-500:#2B86D9;
  --blue-100:#DCE9F8; --blue-50:#EFF5FD; --blue-25:#F7FAFE;
  --ink:#16202C; --ink-2:#33465C; --muted:#6A7C91; --faint:#94A5B8;
  --line:#E3EAF2; --line-2:#EEF2F7; --white:#fff;
  --ok:#1A7F5A; --warn:#B4761B; --bad:#B4453C;
  --radius:12px;
  --shadow:0 1px 2px rgba(16,40,70,.05), 0 6px 18px rgba(16,40,70,.05);
}

/* ---------- shell ---------- */
.stApp{ background:var(--white); }
#MainMenu, footer, header [data-testid="stStatusWidget"]{ visibility:hidden; }
/* Streamlit's header is absolutely positioned and 60px tall, and the main
   area scrolls underneath it. Without this clearance the page title and the
   top of the report sit behind it when scrolled all the way up. */
/* No max-width: the two columns should take whatever room there is, so
   collapsing the sidebar actually widens the report instead of just
   re-centring a capped block. */
[data-testid="stMainBlockContainer"]{
  padding-top:4.5rem; padding-bottom:7rem; max-width:none;
}
/* Note: do NOT widen this to [class*="st-"] - that selector also hits
   Streamlit's icon spans and replaces the Material Symbols font, which makes
   every icon render as its literal ligature name. */
html, body, .stApp{
  font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif;
  color:var(--ink);
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:var(--blue-25); border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1.1rem; }
.sb-brand{ display:flex; gap:.7rem; align-items:center; padding:.1rem 0 .9rem; }
.sb-mark{
  width:36px; height:36px; border-radius:9px; flex:none;
  background:linear-gradient(140deg,var(--blue-600),var(--blue-800));
  color:#fff; font-weight:700; font-size:.95rem;
  display:flex; align-items:center; justify-content:center; letter-spacing:.02em;
}
.sb-title{ font-weight:650; font-size:.95rem; line-height:1.15; }
.sb-sub{ font-size:.72rem; color:var(--muted); margin-top:.12rem; }
.sb-label{
  font-size:.68rem; font-weight:650; letter-spacing:.09em; text-transform:uppercase;
  color:var(--faint); margin:1.1rem 0 .45rem;
}
.db-card{
  background:#fff; border:1px solid var(--line); border-radius:var(--radius);
  padding:.7rem .8rem; box-shadow:var(--shadow);
}
.db-name{ font-weight:600; font-size:.8rem; margin-bottom:.15rem; word-break:break-word; }
.db-meta{ font-size:.7rem; color:var(--muted); margin-bottom:.55rem; }
.db-grid{ display:grid; grid-template-columns:1fr 1fr; gap:.4rem; }
.db-cell{ background:var(--blue-25); border-radius:8px; padding:.35rem .45rem; }
.db-cell b{ display:block; font-size:.95rem; color:var(--blue-700); line-height:1.1; }
.db-cell span{ font-size:.64rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.db-model{
  display:flex; justify-content:space-between; gap:.5rem;
  font-size:.68rem; padding:.3rem 0; border-top:1px dashed var(--line-2);
}
.db-model span:first-child{ color:var(--muted); }
.db-model span:last-child{ font-family:ui-monospace,"Cascadia Mono",Menlo,monospace; color:var(--ink-2); }

/* ---------- page header ---------- */
.page-head{ display:flex; align-items:baseline; gap:.6rem; margin:0 0 .1rem; }
.page-head .t{ font-size:1.22rem; font-weight:650; margin:0; letter-spacing:-.01em; }
.page-head .tag{
  font-size:.64rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  color:var(--blue-700); background:var(--blue-50); border:1px solid var(--blue-100);
  padding:.15rem .45rem; border-radius:999px;
}
.page-sub{ font-size:.8rem; color:var(--muted); margin:0 0 1rem; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{
  background:transparent; padding:.25rem 0; border:none; gap:0;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"]{ display:none; }
.row-user{ display:flex; justify-content:flex-end; }
.bubble-user{
  background:var(--blue-600); color:#fff; padding:.6rem .85rem; border-radius:12px 12px 3px 12px;
  max-width:88%; font-size:.88rem; line-height:1.45;
}
.bubble-bot{
  background:var(--blue-25); border:1px solid var(--line); border-radius:12px 12px 12px 3px;
  padding:.7rem .9rem; font-size:.87rem; line-height:1.5; color:var(--ink-2);
}
.bubble-bot .headline{ color:var(--ink); font-weight:600; font-size:.9rem; margin-bottom:.4rem; }
.res-list{ margin:.5rem 0 0; padding:0; list-style:none; }
.res-list li{
  display:flex; gap:.55rem; align-items:center; padding:.3rem 0;
  border-top:1px solid var(--line-2); font-size:.82rem;
}
.res-list li:first-child{ border-top:none; }
.res-n{
  flex:none; width:19px; height:19px; border-radius:5px; background:var(--blue-600); color:#fff;
  font-size:.66rem; font-weight:700; display:flex; align-items:center; justify-content:center;
}
.res-t{ flex:1; color:var(--ink); }
.res-p{ color:var(--faint); font-size:.72rem; }
.res-s{ font-variant-numeric:tabular-nums; font-weight:650; color:var(--blue-700); font-size:.78rem; }

/* ---------- right panel ---------- */
.panel-head{
  display:flex; align-items:center; justify-content:space-between;
  padding-bottom:.5rem; margin-bottom:.2rem; border-bottom:2px solid var(--blue-600);
}
.panel-head .t{ font-size:.94rem; font-weight:650; margin:0; }
.panel-head .sub{ font-size:.72rem; color:var(--muted); }

/* summary metric strip */
.mstrip{ display:grid; grid-template-columns:repeat(4,1fr); gap:.4rem; margin:.7rem 0 .2rem; }
.mbox{
  background:#fff; border:1px solid var(--line); border-radius:10px;
  padding:.45rem .5rem; text-align:center;
}
.mbox b{ display:block; font-size:1.02rem; color:var(--blue-700); font-variant-numeric:tabular-nums; line-height:1.2; }
.mbox span{ font-size:.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.mbox.conf-High b{ color:var(--ok); } .mbox.conf-Moderate b{ color:var(--blue-700); }
.mbox.conf-Low b{ color:var(--warn); } .mbox.conf-Verylow b, .mbox.conf-Nomatch b{ color:var(--bad); }

/* ---------- chunk card ---------- */
.chunk{
  background:#fff; border:1px solid var(--line); border-radius:var(--radius);
  box-shadow:var(--shadow); margin:.85rem 0; overflow:hidden;
}
.chunk-top{
  display:flex; gap:.6rem; align-items:flex-start;
  padding:.7rem .85rem .6rem; border-bottom:1px solid var(--line-2);
  background:linear-gradient(180deg,var(--blue-25),#fff);
}
.chunk-rank{
  flex:none; width:24px; height:24px; border-radius:7px; background:var(--blue-600); color:#fff;
  font-size:.74rem; font-weight:700; display:flex; align-items:center; justify-content:center;
}
.chunk-id{ flex:1; min-width:0; }
.chunk-id .t{ font-size:.9rem; font-weight:650; margin:0; line-height:1.25; color:var(--ink); }
.chunk-id .path{
  font-size:.64rem; color:var(--faint); margin-top:.18rem;
  font-family:ui-monospace,"Cascadia Mono",Menlo,monospace;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.chunk-score{ flex:none; text-align:right; }
.chunk-score b{ font-size:1.02rem; color:var(--blue-700); font-variant-numeric:tabular-nums; }
.chunk-score span{ display:block; font-size:.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }

.chip-row{ display:flex; flex-wrap:wrap; gap:.3rem; padding:.5rem .85rem 0; }
.chip{
  font-size:.66rem; padding:.13rem .42rem; border-radius:999px;
  background:var(--blue-50); color:var(--blue-700); border:1px solid var(--blue-100);
}
.chip.n{ background:#F4F6F9; color:var(--ink-2); border-color:var(--line); }

.chunk-body{ padding:.6rem .85rem .2rem; font-size:.83rem; line-height:1.6; color:var(--ink-2); }
.chunk-body .sub-h{
  font-size:.8rem; font-weight:650; color:var(--blue-800);
  margin:.9rem 0 .3rem; padding-left:.5rem; border-left:3px solid var(--blue-500);
}
.chunk-body .sub-h:first-child{ margin-top:0; }
.chunk-body p{ margin:.35rem 0; }
.chunk-body ul{ margin:.35rem 0; padding-left:1.05rem; }
.chunk-body li{ margin:.16rem 0; }
.chunk-body mark{
  background:var(--blue-100); color:var(--blue-800); font-weight:550;
  padding:0 .12em; border-radius:2px;
}
/* block headings that separate text / tables / images / colours / fonts */
.blk{ margin:1.1rem 0 0; }
.blk-h{
  display:flex; align-items:baseline; gap:.45rem;
  font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--blue-700); padding-bottom:.28rem; margin-bottom:.5rem;
  border-bottom:1px solid var(--blue-100);
}
.blk-h .n{
  font-weight:600; letter-spacing:0; text-transform:none;
  font-size:.68rem; color:var(--muted);
}

/* tables */
.tbl-wrap{ margin:.7rem 0; }
.tbl-cap{
  font-size:.7rem; font-weight:650; color:var(--ink-2); margin-bottom:.3rem;
}
.tbl-cap span{ font-weight:400; color:var(--muted); }
.tbl-scroll{ overflow-x:auto; border:1px solid var(--line); border-radius:9px; }
.tbl-scroll table{ border-collapse:collapse; width:100%; font-size:.76rem; }
.tbl-scroll th{
  background:var(--blue-600); color:#fff; font-weight:600; text-align:left;
  padding:.4rem .55rem; white-space:nowrap;
}
.tbl-scroll td{ padding:.35rem .55rem; border-top:1px solid var(--line-2); color:var(--ink-2); }
.tbl-scroll tbody tr:nth-child(even){ background:var(--blue-25); }

/* images */
.img-grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:.6rem; margin:.6rem 0; }
.img-frame{ border:1px solid var(--line); border-radius:9px; overflow:hidden; background:#fff; }
.img-frame .lab{
  font-size:.68rem; font-weight:650; color:var(--ink-2);
  padding:.32rem .5rem; background:var(--blue-25); border-bottom:1px solid var(--line-2);
}
.img-frame .lab span{ font-weight:400; color:var(--muted); }
.img-frame .shot{
  display:flex;
  background:repeating-conic-gradient(#F4F7FB 0 25%, #fff 0 50%) 50%/14px 14px;
  display:flex; align-items:center; justify-content:center; min-height:110px; padding:.4rem;
}
.img-frame img{ max-width:100%; max-height:230px; display:block; }
.img-frame .cap{
  font-size:.58rem; color:var(--faint); padding:.28rem .5rem;
  border-top:1px solid var(--line-2); word-break:break-all; line-height:1.3;
  font-family:ui-monospace,"Cascadia Mono",Menlo,monospace;
}

/* Click-to-enlarge. A hidden checkbox drives it, so the same <img> element is
   promoted to a full-screen overlay - no second copy of the data URI, and no
   JavaScript, which Streamlit strips from markdown anyway. */
.lb-cb{ position:absolute; width:0; height:0; opacity:0; pointer-events:none; }
.lb-open{ display:block; cursor:zoom-in; margin:0; }
/* A sticky column creates its own stacking context, which would otherwise
   trap the fixed overlay underneath the sidebar (z 999991) and the chat
   column. Lift the column only while one of its lightboxes is open. */
div[data-testid="stColumn"]:has(.lb-cb:checked){ z-index:1000000; }
.lb-cb:checked + .lb-open{
  position:fixed; inset:0; z-index:2147483000; cursor:zoom-out;
  background:rgba(9,24,44,.90); padding:4vh 4vw;
  display:flex; align-items:center; justify-content:center;
}
.lb-cb:checked + .lb-open .shot{
  background:none; min-height:0; padding:0; max-width:100%; max-height:100%;
}
.lb-cb:checked + .lb-open img{
  max-width:92vw; max-height:88vh; border-radius:6px;
  box-shadow:0 18px 60px rgba(0,0,0,.5); background:#fff;
}
.lb-cb:checked + .lb-open::after{
  content:"\\2715  Close"; position:fixed; top:2.2vh; right:3vw;
  font-size:.82rem; font-weight:600; color:#fff;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35);
  border-radius:8px; padding:.35rem .7rem;
}
.lb-cb:checked + .lb-open::before{
  content:attr(data-cap); position:fixed; left:0; right:0; bottom:2.2vh;
  text-align:center; color:#DCE9F8; font-size:.78rem; padding:0 4vw;
}
.img-miss{
  min-height:88px; display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:.2rem; border:1px dashed var(--line); border-radius:9px; padding:.5rem; text-align:center;
  background:var(--blue-25);
}
.img-miss .ic{ font-size:1rem; opacity:.45; }
.img-miss .t{ font-size:.62rem; color:var(--muted); }
.img-miss .f{
  font-size:.55rem; color:var(--faint); word-break:break-all; line-height:1.25;
  font-family:ui-monospace,"Cascadia Mono",Menlo,monospace;
}

/* colour swatches */
.sw-row{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.1rem 0 .2rem; }
.sw{
  display:flex; align-items:center; gap:.35rem; border:1px solid var(--line);
  border-radius:8px; padding:.2rem .5rem .2rem .25rem; font-size:.68rem; color:var(--ink-2);
  font-family:ui-monospace,"Cascadia Mono",Menlo,monospace;
}
.sw i{ width:16px; height:16px; border-radius:4px; border:1px solid rgba(0,0,0,.12); display:block; }
.sw em{ font-style:normal; color:var(--faint); font-size:.62rem; }

/* fonts */
.font-row{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.1rem 0 .2rem; }
.font-pill{
  border:1px solid var(--line); border-radius:8px; padding:.22rem .55rem;
  font-size:.72rem; color:var(--ink-2); background:#fff;
}

/* the show-details drawer at the foot of a card */
.card-det{ border-top:1px solid var(--line-2); }
.card-det summary{
  cursor:pointer; user-select:none; list-style:none;
  font-size:.72rem; font-weight:650; color:var(--blue-700);
  padding:.5rem .85rem; background:#fff;
}
.card-det summary:hover{ background:var(--blue-25); }
.card-det summary::-webkit-details-marker{ display:none; }
.card-det summary::before{ content:"▸  "; }
.card-det[open] summary::before{ content:"▾  "; }
.card-det .det-body{
  padding:.2rem .85rem .8rem; font-size:.76rem; line-height:1.65; color:var(--ink-2);
  background:#fff;
}
.card-det .det-body p{ margin:.3rem 0; }
.card-det .det-body ol{ margin:.35rem 0; padding-left:1.4rem; }
.card-det .det-body li{ margin:.3rem 0; }
.card-det .det-body b{ color:var(--ink); font-weight:600; }

/* per-chunk metrics */
.mets{ border-top:1px solid var(--line-2); background:var(--blue-25); padding:.55rem .85rem .6rem; }
.mets-lab{ font-size:.6rem; text-transform:uppercase; letter-spacing:.07em; color:var(--faint); margin-bottom:.35rem; }
.mets-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(84px,1fr)); gap:.4rem .5rem; }
.met .k{ font-size:.6rem; color:var(--muted); display:flex; justify-content:space-between; gap:.25rem; }
.met .k b{ color:var(--ink); font-variant-numeric:tabular-nums; font-weight:650; }
.met .bar{ height:3px; border-radius:2px; background:var(--blue-100); margin-top:.2rem; overflow:hidden; }
.met .bar i{ display:block; height:100%; background:var(--blue-600); border-radius:2px; }
.met.alt .bar i{ background:var(--blue-500); }

/* empty / info states */
.empty{
  border:1px dashed var(--line); border-radius:var(--radius); padding:1.6rem 1rem;
  text-align:center; color:var(--muted); font-size:.82rem; background:var(--blue-25);
}
.empty .ic{ font-size:1.4rem; display:block; margin-bottom:.4rem; opacity:.5; }
.empty b{ color:var(--ink-2); display:block; margin-bottom:.25rem; font-size:.86rem; }

.note{
  font-size:.72rem; color:var(--muted); background:var(--blue-25);
  border-left:3px solid var(--blue-500); border-radius:0 8px 8px 0; padding:.45rem .6rem; margin:.5rem 0;
}

/* thinking trace */
.trace-row{ display:flex; gap:.5rem; align-items:baseline; font-size:.78rem; padding:.13rem 0; }
.trace-row .d{ flex:none; width:6px; height:6px; border-radius:50%; background:var(--blue-500); }
.trace-row .l{ font-weight:600; color:var(--ink-2); }
.trace-row .x{ color:var(--muted); flex:1; }
.trace-row .t{ color:var(--faint); font-variant-numeric:tabular-nums; font-size:.7rem; }

/* ---------- streamlit widget polish ---------- */
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapsedControl"] button{
  border-radius:8px;
}
.stButton>button{
  border-radius:9px; border:1px solid var(--line); background:#fff; color:var(--ink-2);
  font-size:.78rem; font-weight:550; padding:.3rem .6rem;
}
.stButton>button:hover{ border-color:var(--blue-500); color:var(--blue-700); background:var(--blue-25); }
[data-testid="stSidebar"] .stButton>button{ width:100%; text-align:left; }
div[data-testid="stExpander"] details{
  border:1px solid var(--line); border-radius:10px; background:#fff;
}
div[data-testid="stExpander"] summary{ font-size:.78rem; font-weight:550; }
[data-testid="stMetricValue"]{ font-size:1.05rem; color:var(--blue-700); }

/* the chat bar sits under the conversation column on the right */
[data-testid="stBottomBlockContainer"]{
  background:linear-gradient(180deg,rgba(255,255,255,0),#fff 22%);
  padding-bottom:1rem;
}
[data-testid="stBottomBlockContainer"] .stChatInput{ max-width:none; }
[data-testid="stChatInput"]{ border-radius:11px; border:1px solid var(--line); box-shadow:var(--shadow); }
[data-testid="stChatInput"]:focus-within{ border-color:var(--blue-500); }

@media (min-width:1200px){
  /* The report column is 2.45 of 3.45, so the chat starts at ~71%. */
  [data-testid="stBottomBlockContainer"] > div{ padding-left:72.5%; }

  /* Both columns scroll on their own, so the page itself is never taller than
     the viewport. That matters because Streamlit auto-scrolls the app to the
     bottom whenever a chat message is added - with a page-height report that
     would land the reader on the last chunk instead of the best one. */
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
    align-self:flex-start; position:sticky; top:4.5rem;
    max-height:calc(100vh - 11rem); overflow-y:auto; overflow-x:hidden;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
    padding-right:.9rem;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]::-webkit-scrollbar{
    width:8px;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]::-webkit-scrollbar-thumb{
    background:var(--line); border-radius:4px;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:hover::-webkit-scrollbar-thumb{
    background:#CBD6E4;
  }
  /* Guard: a column nested inside one of those must not scroll on its own. */
  div[data-testid="stColumn"] div[data-testid="stColumn"]{
    position:static; max-height:none; overflow:visible;
  }
  /* The columns reserve their own room for the chat bar, so the page below
     them needs almost none - this is what stops the app scrolling at all. */
  [data-testid="stMainBlockContainer"]{ padding-bottom:1.5rem; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(text) -> str:
    return html.escape(str(text), quote=True)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """Escape, then re-enable the tiny bit of markdown the corpus uses."""
    out = esc(text)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _CODE.sub(r"<code>\1</code>", out)
    return out


def highlight(text: str, terms) -> str:
    """Mark query terms inside already-escaped text."""
    out = _inline(text)
    for term in sorted({t for t in terms if len(t) > 3}, key=len, reverse=True):
        out = re.sub(
            r"(?<![\w>])(" + re.escape(esc(term)) + r")(?![\w<])",
            r"<mark>\1</mark>",
            out,
            flags=re.IGNORECASE,
        )
    return out


# ---------------------------------------------------------------------------
# Markdown tables -> HTML
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|") and line.count("|") >= 2


def md_table_to_html(markdown: str, caption: str = "", escape_caption: bool = True) -> str:
    """Convert one GitHub-flavoured markdown table into a scrollable HTML table."""
    rows = [l for l in markdown.strip().split("\n") if l.strip()]
    if not rows:
        return ""
    header, body = None, []
    for i, line in enumerate(rows):
        if _SEP_RE.match(line) and i > 0:
            header = _cells(rows[i - 1])
            body = [_cells(r) for r in rows[i + 1:] if is_table_line(r)]
            break
    if header is None:
        header, body = _cells(rows[0]), [_cells(r) for r in rows[1:]]

    width = max([len(header)] + [len(r) for r in body] or [0])
    header += [""] * (width - len(header))

    thead = "".join(f"<th>{_inline(c)}</th>" for c in header)
    trs = []
    for row in body:
        row = row + [""] * (width - len(row))
        trs.append("".join(f"<td>{_inline(c)}</td>" for c in row))
    tbody = "".join(f"<tr>{r}</tr>" for r in trs)

    text = esc(caption) if escape_caption else caption
    cap = f'<div class="tbl-cap">{text}</div>' if caption else ""
    return (
        f'<div class="tbl-wrap">{cap}<div class="tbl-scroll"><table>'
        f"<thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody>"
        f"</table></div></div>"
    )


# ---------------------------------------------------------------------------
# Body prose -> HTML
# ---------------------------------------------------------------------------

_BULLETS = ("•", "▪", "‣", "·", "- ", "– ", "— ")


def _is_bullet(line: str) -> bool:
    s = line.strip()
    return bool(s) and (s.startswith(_BULLETS))


def _strip_bullet(line: str) -> str:
    s = line.strip()
    for b in _BULLETS:
        if s.startswith(b):
            return s[len(b):].strip()
    return s


def _is_heading(line: str, nxt: str) -> bool:
    """Decide whether a line is a section heading rather than wrapped prose.

    The source is hard-wrapped, so fragments like "POLIVY-R-C" or "HP" land on
    their own line and used to be mistaken for headings. Requiring several
    words, some lower case and a following line filters those out.
    """
    s = line.strip()
    if not s or not (8 <= len(s) <= 72) or _is_bullet(s):
        return False
    if s.endswith((".", ",", ";", ":", "?", "-", "/")):
        return False
    words = s.split()
    if not (2 <= len(words) <= 9):
        return False
    if not any(c.islower() for c in s):   # an all-caps fragment, not a heading
        return False
    if sum(c.isdigit() for c in s) > len(s) / 4:  # a run of table numbers
        return False
    titled = sum(1 for w in words if w[:1].isupper())
    return titled >= max(1, len(words) - 2) and bool(nxt.strip())


def _absorb_wrap(lines, i: int):
    """Rejoin a heading that the PDF wrapped onto a second line.

    Returns the heading text and the index of the line after it.
    """
    head = lines[i].strip()
    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
    if (
        nxt
        and len(nxt.split()) <= 2
        and not nxt.endswith((".", ",", ";", ":", "?", "*"))
        and not _is_bullet(nxt)
        and len(head) + len(nxt) <= 90
        and nxt[:1].isupper()
    ):
        return f"{head} {nxt}", i + 2
    return head, i + 1


def _implicit_list(lines):
    """Recover a bullet list that the PDF exported as a caption grid.

    Figure captions come out as runs of very short lines where one label
    repeats ("Do NOT", "Grade", ...). Splitting on that repeated label turns an
    unreadable run-on paragraph back into the list it was drawn as. Returns
    None when the run is ordinary wrapped prose.
    """
    if len(lines) < 4:
        return None
    if sum(len(l) for l in lines) / len(lines) > 32:
        return None

    counts = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    marker, hits = max(counts.items(), key=lambda kv: kv[1])
    # A real repeated label is a word or two ("Do NOT", "Grade"). Placeholder
    # cell values from a flattened table ("XX", "-") are not, and turning
    # those into a bullet each makes the chunk far harder to read.
    if hits < 2 or not marker[:1].isupper() or not (3 <= len(marker) <= 24):
        return None
    if len(marker.split()) > 3:
        return None

    items, current = [], []
    for line in lines:
        if line == marker:
            if current:
                items.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        items.append(" ".join(current))

    if len(items) < 2:
        return None
    # Items that are barely longer than the marker itself are table cells.
    if sum(len(i) for i in items) / len(items) < len(marker) + 9:
        return None
    return items


def body_to_html(text: str, terms=()) -> str:
    """Turn PDF-extracted prose into readable HTML.

    The source is hard-wrapped mid-sentence, so lines inside a paragraph are
    rejoined; blank lines, bullets and heading-shaped lines break the flow.
    """
    out = []
    para, bullets = [], []

    def flush_para():
        if not para:
            return
        items = _implicit_list(para)
        if items:
            body = "".join(f"<li>{highlight(x, terms)}</li>" for x in items)
            out.append(f"<ul>{body}</ul>")
        else:
            out.append(f"<p>{highlight(' '.join(para), terms)}</p>")
        para.clear()

    def flush_bullets():
        if bullets:
            items = "".join(f"<li>{highlight(b, terms)}</li>" for b in bullets)
            out.append(f"<ul>{items}</ul>")
            bullets.clear()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_para()
            flush_bullets()
            i += 1
            continue

        # Lone glyphs (dingbats the extractor could not map) are noise.
        if len(stripped) == 1 and not stripped.isalnum():
            i += 1
            continue

        if is_table_line(line):
            block = []
            while i < len(lines) and is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            flush_para()
            flush_bullets()
            out.append(md_table_to_html("\n".join(block)))
            continue

        if _is_bullet(line):
            flush_para()
            bullets.append(_strip_bullet(line))
            i += 1
            continue

        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        starts_block = i == 0 or not lines[i - 1].strip()
        if starts_block and not para and _is_heading(line, nxt):
            flush_bullets()
            heading, i = _absorb_wrap(lines, i)
            out.append(f'<div class="sub-h">{highlight(heading, terms)}</div>')
            continue

        flush_bullets()
        para.append(stripped)
        i += 1

    flush_para()
    flush_bullets()
    # The whole chunk is always rendered - no truncation, no "show more".
    return "".join(out) or "<p>No body text in this section.</p>"


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def resolve_image(local_path: str):
    if not local_path:
        return None
    rel = Path(str(local_path).replace("\\", "/"))
    for root in IMAGE_ROOTS:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def image_url(path: Path):
    """URL for a resolved image, or None if it is not under static/.

    Only files Streamlit can serve get a URL. Everything else degrades to a
    placeholder - deliberately, because the alternative (base64-inlining it)
    is what blew the memory budget.
    """
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(STATIC_DIR.resolve())
    except (ValueError, OSError):
        return None
    return f"{STATIC_URL}/{rel.as_posix()}"


def images_to_html(images, scope: str = "") -> str:
    """Render the section's figures, each labelled with its own metadata.

    A resolved image is wrapped in a checkbox-driven lightbox so clicking it
    opens it full screen. Files that were never shipped alongside the JSONL
    fall back to a placeholder naming the page, classification and path.
    """
    if not images:
        return ""
    shown = images[:MAX_IMAGES_PER_CHUNK]
    cards = []
    for n, img in enumerate(shown, start=1):
        page = img.get("page_number", "?")
        kind = img.get("classification", "figure")
        local = str(img.get("local_path", "") or "unknown")
        label = (
            f'<div class="lab">Image {n} '
            f"<span>&middot; page {esc(page)} &middot; {esc(kind)}</span></div>"
        )
        url = image_url(resolve_image(local))
        if url:
            uid = "lb" + hashlib.md5(f"{scope}|{local}|{n}".encode()).hexdigest()[:10]
            caption = f"Image {n} - page {page} - {kind} - {Path(local).name}"
            body = (
                f'<input class="lb-cb" type="checkbox" id="{uid}">'
                f'<label class="lb-open" for="{uid}" data-cap="{esc(caption)}">'
                f'<span class="shot">'
                # No loading="lazy": before it loads the img is a 0x0 box, so the
                # intersection check never fires and it stays blank.
                f'<img src="{esc(url)}" alt="Image {n} from page {esc(page)}">'
                f"</span></label>"
            )
        else:
            body = (
                f'<div class="img-miss"><span class="ic">&#128443;</span>'
                f'<span class="t">Not served from static/</span>'
                f'<span class="f">{esc(Path(local).name)}</span></div>'
            )
        cards.append(
            f'<figure class="img-frame">{label}{body}'
            f'<figcaption class="cap">{esc(local)}</figcaption></figure>'
        )

    extra = len(images) - len(shown)
    more = (
        f'<div class="note">{extra} further image(s) in this section not shown.</div>'
        if extra > 0 else ""
    )
    return (
        f'<div class="blk"><div class="blk-h">Images'
        f'<span class="n">{len(images)} in this chunk</span></div>'
        f'<div class="img-grid">{"".join(cards)}</div>{more}</div>'
    )


def tables_to_html(tables) -> str:
    """Each table gets its own heading and page reference."""
    usable = [t for t in tables if (t.get("markdown") or "").strip()]
    if not usable:
        return ""
    blocks = []
    for n, tbl in enumerate(usable, start=1):
        caption = f"Table {n} <span>&middot; page {esc(tbl.get('page_number', '?'))}</span>"
        blocks.append(md_table_to_html(tbl["markdown"], caption, escape_caption=False))
    return (
        f'<div class="blk"><div class="blk-h">Tables'
        f'<span class="n">{len(usable)} in this chunk</span></div>'
        f'{"".join(blocks)}</div>'
    )


def swatches_to_html(swatches) -> str:
    """Colour swatches, under an explicit label."""
    if not swatches:
        return ""
    chips = []
    for item in swatches[:24]:
        if isinstance(item, (list, tuple)) and item:
            hex_code, count = str(item[0]), (item[1] if len(item) > 1 else "")
        else:
            hex_code, count = str(item), ""
        safe = hex_code if re.fullmatch(r"#[0-9A-Fa-f]{3,8}", hex_code) else "#CCCCCC"
        tail = f"<em>&times;{esc(count)}</em>" if count != "" else ""
        chips.append(
            f'<span class="sw"><i style="background:{safe}"></i>{esc(hex_code)}{tail}</span>'
        )
    return (
        f'<div class="blk"><div class="blk-h">Colours'
        f'<span class="n">{len(swatches)} swatch(es), with pixel counts</span></div>'
        f'<div class="sw-row">{"".join(chips)}</div></div>'
    )


def fonts_to_html(fonts) -> str:
    """The `fonts_present` list from the JSONL."""
    if not fonts:
        return ""
    pills = "".join(f'<span class="font-pill">{esc(f)}</span>' for f in fonts)
    return (
        f'<div class="blk"><div class="blk-h">Fonts'
        f'<span class="n">{len(fonts)} from <code>fonts_present</code></span></div>'
        f'<div class="font-row">{pills}</div></div>'
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _meter(label: str, value: float, display=None, alt=False) -> str:
    width = max(0.0, min(1.0, float(value))) * 100
    shown = display if display is not None else f"{value:.3f}"
    cls = "met alt" if alt else "met"
    return (
        f'<div class="{cls}"><div class="k"><span>{esc(label)}</span><b>{esc(shown)}</b></div>'
        f'<div class="bar"><i style="width:{width:.1f}%"></i></div></div>'
    )


def hit_metrics_html(hit) -> str:
    moved = hit.fusion_rank - hit.rank
    arrow = "&#8593;" if moved > 0 else ("&#8595;" if moved < 0 else "&#8722;")
    meters = "".join([
        _meter("Relevance", hit.relevance, pct(hit.relevance)),
        _meter("Rerank", hit.rerank, f"{hit.rerank:.3f}", alt=True),
        _meter("Vector cos", hit.dense, f"{hit.dense:.3f}", alt=True),
        _meter("BM25", hit.bm25, f"{hit.bm25:.3f}", alt=True),
        _meter("Fusion", hit.fusion, f"{hit.fusion:.3f}", alt=True),
        _meter("Term cover", hit.keyword_coverage, pct(hit.keyword_coverage), alt=True),
    ])
    detail = (
        f"logit {hit.rerank_logit:+.2f} &middot; fused rank {hit.fusion_rank} "
        f"&rarr; {hit.rank} {arrow} &middot; {hit.passages_hit} passage(s) matched"
    )
    return (
        f'<div class="mets"><div class="mets-lab">Retrieval metrics &middot; {detail}</div>'
        f'<div class="mets-grid">{meters}</div></div>'
    )


def summary_html(metrics: dict, timings: dict) -> str:
    conf = metrics.get("confidence", "-")
    cls = "conf-" + conf.replace(" ", "")
    boxes = [
        (pct(metrics["top_relevance"]), "Top relevance", ""),
        (pct(metrics["mean_relevance"]), f"Mean of {metrics['returned']}", ""),
        (conf, "Confidence", cls),
        (f"{timings.get('total_ms', 0):.0f} ms", "Latency", ""),
        (pct(metrics["margin"]), "Score margin", ""),
        (pct(metrics["coverage"]), "Term coverage", ""),
        (pct(metrics["agreement"]), "Retriever agree", ""),
        (str(metrics["scanned"]), "Passages scored", ""),
    ]
    cells = "".join(
        f'<div class="mbox {c}"><b>{esc(v)}</b><span>{esc(l)}</span></div>'
        for v, l, c in boxes
    )
    return f'<div class="mstrip">{cells}</div>'


# ---------------------------------------------------------------------------
# The chunk card
# ---------------------------------------------------------------------------

def _pages_sentence(sec) -> str:
    if not sec.has_pages:
        return ""
    if sec.page_start == sec.page_end:
        located = f"This content is located on page {sec.page_start}."
    else:
        located = (
            f"This content is located on pages {sec.page_start} to {sec.page_end}."
        )
    if sec.source_pages:
        covered = ", ".join(str(p) for p in sec.source_pages)
        located += (
            f" The specific source page numbers covered are: {esc(covered)}."
        )
    return f"<p>{located}</p>"


def details_prose_html(sec) -> str:
    """The provenance drawer at the foot of a card, written out in prose."""
    parts = [f'<p>It belongs to the section titled "<b>{esc(sec.section_label)}</b>".</p>']

    pages = _pages_sentence(sec)
    if pages:
        parts.append(pages)

    if sec.document_id:
        parts.append(f"<p>It comes from the document <b>{esc(sec.document_id)}</b>.</p>")

    if sec.matched_fields:
        fields = ", ".join(esc(f) for f in sec.matched_fields)
        parts.append(
            f"<p>The fields that were matched for this chunk are: {fields}.</p>"
        )

    rows = [d for d in sec.field_match_details if isinstance(d, dict)]
    if rows:
        items = []
        for d in rows:
            name = esc(d.get("field", "unnamed"))
            group = esc(d.get("attribute_group", "an unnamed group"))
            score = d.get("score")
            reason = esc(d.get("selection_reason", "unspecified"))
            score_txt = (
                f"{float(score):.4f}" if isinstance(score, (int, float)) else esc(score)
            )
            items.append(
                f'<li>The field "<b>{name}</b>" belongs to the attribute group '
                f'"{group}". It was matched with a relevance score of {score_txt}, '
                f'and the reason for selecting it was "{reason}".</li>'
            )
        parts.append("<p>Details about how each field was matched:</p>")
        parts.append(f'<ol>{"".join(items)}</ol>')

    parts.append(
        f"<p>Chunk id: <b>{esc(sec.chunk_id)}</b> &middot; {sec.word_count} words"
        f" &middot; {len(sec.tables)} table(s) &middot; {len(sec.images)} image(s).</p>"
    )

    return (
        f'<details class="card-det"><summary>Show details</summary>'
        f'<div class="det-body">{"".join(parts)}</div></details>'
    )


def chunk_card_html(hit) -> str:
    """One retrieved chunk: its text in full, then its media, then provenance."""
    sec = hit.section
    terms = hit.matched_terms

    return (
        f'<div class="chunk">'
        f'<div class="chunk-top">'
        f'<div class="chunk-rank">{hit.rank}</div>'
        f'<div class="chunk-id"><div class="t">{esc(sec.section_label)}</div>'
        f'<div class="path">{esc(sec.chunk_id)}</div></div>'
        f'<div class="chunk-score"><b>{pct(hit.relevance)}</b><span>relevance</span></div>'
        f"</div>"
        f'<div class="chunk-body">'
        f"{body_to_html(sec.text, terms)}"
        f"{tables_to_html(sec.tables)}"
        f"{images_to_html(sec.images, sec.chunk_id)}"
        f"{swatches_to_html(sec.color_swatches)}"
        f"{fonts_to_html(sec.fonts_present)}"
        f"</div>"
        f"{details_prose_html(sec)}"
        f"{hit_metrics_html(hit)}"
        f"</div>"
    )
