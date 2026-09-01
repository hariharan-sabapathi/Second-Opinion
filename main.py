"""Second Opinion — Gradio front end for the self-reflective retrieval loop."""

import html
import os
import re
import shutil
import time

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_graph import CHROMA_DB_PATH, MAX_ITERATIONS, app, embeddings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ── Small HTML helpers ─────────────────────────────────────────────────────────

def note(text, tone="quiet"):
    """A single-line status note with a coloured marker."""
    return f'<p class="note note--{tone}"><span class="marker"></span>{html.escape(text)}</p>'


ANSWER_EMPTY = "*The answer will appear here.*"


def empty(text):
    return f'<p class="empty">{html.escape(text)}</p>'


VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(.*)$", re.I)
REASON_RE = re.compile(r"^\s*REASON\s*:\s*(.*)$", re.I)
QUERY_RE = re.compile(r"^\s*REFINED_QUERY\s*:\s*(.*)$", re.I)
ROUND_RE = re.compile(r"^\s*Iteration\s+(\d+)", re.I)


def parse_entry(entry, fallback_round):
    """Pull the round number, verdict, reason and rewritten query out of one grading log entry."""
    parsed = {"round": fallback_round, "verdict": "", "reason": "", "query": ""}
    for line in entry.splitlines():
        if (m := ROUND_RE.match(line)):
            parsed["round"] = int(m.group(1))
        elif (m := VERDICT_RE.match(line)):
            parsed["verdict"] = m.group(1).strip()
        elif (m := REASON_RE.match(line)):
            parsed["reason"] = m.group(1).strip()
        elif (m := QUERY_RE.match(line)):
            parsed["query"] = m.group(1).strip()
    return parsed


def slip(entry, fallback_round, forced=False):
    """Render one grading round as a review slip."""
    p = parse_entry(entry, fallback_round)
    approved = "YES" in p["verdict"].upper()
    stamp = "Approved" if approved else "Sent back"
    tone = "pass" if approved else "back"
    reason = p["reason"] or "The grader returned no reason for this round."
    query = p["query"]

    rewrite = ""
    if not approved and query and query.upper() != "NONE":
        rewrite = (
            '<p class="rewrite"><span>Searched again for</span>'
            f'<code>{html.escape(query)}</code></p>'
        )

    override = (
        '<p class="override">Round limit reached, so the answer was written '
        'from the best context found so far.</p>'
        if forced else ""
    )

    return f"""
    <article class="slip slip--{tone}">
      <div class="slip-head">
        <span class="round">Round {p['round']}</span>
        <span class="stamp stamp--{tone}">{stamp}</span>
      </div>
      <p class="reason">{html.escape(reason)}</p>
      {rewrite}
      {override}
    </article>
    """


def render_review(log, rounds, seconds):
    if not log:
        return empty("No review notes yet.")

    forced_index = len(log) - 1 if rounds >= MAX_ITERATIONS else -1
    slips = "".join(
        slip(entry, i + 1, forced=(i == forced_index and "YES" not in entry.upper()))
        for i, entry in enumerate(log)
    )
    word = "round" if rounds == 1 else "rounds"
    summary = f"{rounds} {word} of retrieval in {seconds:.1f}s"
    return f'<div class="review"><p class="review-meta">{summary}</p>{slips}</div>'


# ── Document handling ──────────────────────────────────────────────────────────

def db_ready():
    return os.path.isdir(CHROMA_DB_PATH) and any(os.scandir(CHROMA_DB_PATH))


def process_pdf(file):
    """Load, chunk and index a PDF into the Chroma store, replacing whatever was there."""
    if file is None:
        yield note("Choose a PDF first.", "warn"), gr.update(interactive=False)
        return

    yield note("Reading and indexing the file…", "busy"), gr.update(interactive=False)

    try:
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        # Gradio returns a path string; older versions hand back a file-like object.
        file_path = file if isinstance(file, str) else file.name
        pages = PyPDFLoader(file_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(pages)
        Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)

        yield (
            note(f"{len(chunks)} passages indexed from {len(pages)} pages. Ask away.", "pass"),
            gr.update(interactive=True),
        )
    except Exception as e:
        yield (
            note(f"Couldn't index that file: {e}", "back"),
            gr.update(interactive=False),
        )


def clear_pdf():
    """Drop the vector store and put the page back to its starting state."""
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    return (
        None,
        note("Document removed. Upload another to keep going."),
        gr.update(interactive=False),
        empty("No review notes yet."),
        ANSWER_EMPTY,
    )


# ── Asking ─────────────────────────────────────────────────────────────────────

def ask_question(question):
    """Run the graph and stream the review notes, then the answer."""
    question = (question or "").strip()

    if not question:
        yield empty("Type a question first."), ANSWER_EMPTY
        return

    if not db_ready():
        yield note("No document indexed yet — upload a PDF and process it.", "warn"), \
              ANSWER_EMPTY
        return

    yield (
        f'<div class="review"><article class="slip slip--busy">'
        f'<div class="slip-head"><span class="round">Round 1</span>'
        f'<span class="stamp stamp--busy">Reviewing</span></div>'
        f'<p class="reason">Retrieving passages for '
        f'<em>{html.escape(question)}</em> and grading them.</p></article></div>',
        "*Writing the answer once the context passes review…*",
    )

    started = time.perf_counter()
    try:
        result = app.invoke({
            "question": question,
            "refined_query": "",
            "context": "",
            "reflection": "",
            "answer": "",
            "iterations": 0,
            "reflection_log": [],
        })
    except Exception as e:
        yield note(f"The run stopped: {e}", "back"), "*No answer this time.*"
        return

    elapsed = time.perf_counter() - started
    yield (
        render_review(result.get("reflection_log", []), result.get("iterations", 0), elapsed),
        result.get("answer") or "*The model returned nothing.*",
    )


def clear_question():
    return "", empty("No review notes yet."), ANSWER_EMPTY


# ── Look and feel ──────────────────────────────────────────────────────────────

PAPER = "#DEE2DA"     # the desk
SHEET = "#F5F6F2"     # the page
INK = "#232C27"
INK_SOFT = "#5C665D"
RULE = "#C3C9BD"
PENCIL = "#A93B2B"    # correction red
STAMP = "#3D6B4F"     # approval green

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Newsreader"), "Georgia", "serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=PAPER, body_background_fill_dark=PAPER,
    body_text_color=INK, body_text_color_dark=INK,
    body_text_color_subdued=INK_SOFT, body_text_color_subdued_dark=INK_SOFT,
    background_fill_primary=PAPER, background_fill_primary_dark=PAPER,
    background_fill_secondary=SHEET, background_fill_secondary_dark=SHEET,
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_text_color=INK_SOFT, block_label_text_color_dark=INK_SOFT,
    block_title_text_color=INK_SOFT, block_title_text_color_dark=INK_SOFT,
    block_info_text_color=INK_SOFT, block_info_text_color_dark=INK_SOFT,
    block_shadow="none",
    block_radius="2px",
    border_color_primary=RULE, border_color_primary_dark=RULE,
    panel_background_fill=SHEET, panel_background_fill_dark=SHEET,
    panel_border_color=RULE, panel_border_color_dark=RULE,
    input_background_fill=SHEET, input_background_fill_dark=SHEET,
    input_border_color=RULE, input_border_color_dark=RULE,
    input_placeholder_color="#9AA396", input_placeholder_color_dark="#9AA396",
    input_radius="2px",
    button_large_radius="2px", button_small_radius="2px",
    button_primary_background_fill=INK, button_primary_background_fill_dark=INK,
    button_primary_background_fill_hover="#38453B",
    button_primary_background_fill_hover_dark="#38453B",
    button_primary_text_color=SHEET, button_primary_text_color_dark=SHEET,
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_background_fill_hover="#D3D8CD",
    button_secondary_background_fill_hover_dark="#D3D8CD",
    button_secondary_text_color=INK, button_secondary_text_color_dark=INK,
    button_secondary_border_color=RULE, button_secondary_border_color_dark=RULE,
)

CSS = """
:root, .dark {
  --paper:#DEE2DA; --sheet:#F5F6F2; --ink:#232C27; --ink-soft:#5C665D;
  --rule:#C3C9BD; --pencil:#A93B2B; --stamp:#3D6B4F;
}
gradio-app, .gradio-container { background: var(--paper) !important; }
.gradio-container { max-width: 1060px !important; padding: 0 28px 72px !important; }
footer { display: none !important; }
* { -webkit-font-smoothing: antialiased; }

/* Masthead */
.masthead { padding: 56px 0 20px; border-bottom: 1px solid var(--rule); }
.masthead h1 {
  font-size: clamp(2.6rem, 6vw, 4rem); font-weight: 500; line-height: 1;
  letter-spacing: -0.03em; margin: 0 0 10px; color: var(--ink);
}
.masthead .deck {
  font-size: clamp(1.15rem, 2.2vw, 1.45rem); font-style: italic; font-weight: 400;
  line-height: 1.25; margin: 0 0 16px !important; color: var(--ink); max-width: 30ch;
}
.masthead p { margin: 0; max-width: 54ch; font-size: 1.12rem; line-height: 1.5; color: var(--ink-soft); }
.colophon {
  margin-top: 22px !important; font-family: var(--font-mono); font-size: .74rem;
  letter-spacing: .01em; color: var(--ink-soft);
}

/* Column headings */
.desk { padding-top: 28px; gap: 40px !important; }
.desk h2 {
  font-size: 1.12rem; font-weight: 600; margin: 0 0 2px; color: var(--ink);
}
.desk .hint { margin: 0 0 14px; font-size: .92rem; color: var(--ink-soft); }
.right-col { border-left: 1px solid var(--rule); padding-left: 40px !important; }

/* Notes and empty states */
.note { display:flex; gap:9px; align-items:baseline; margin:4px 0 0; font-size:.95rem; color: var(--ink); }
.note .marker { width:7px; height:7px; flex:none; background: var(--ink-soft); transform: translateY(-1px); }
.note--pass .marker { background: var(--stamp); }
.note--back .marker, .note--warn .marker { background: var(--pencil); }
.note--busy .marker { background: var(--ink-soft); animation: pulse 1.1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .25; } }
.empty { margin: 4px 0 0; font-style: italic; color: var(--ink-soft); font-size: .98rem; }

/* Review slips */
.review-meta {
  font-family: var(--font-mono); font-size: .74rem; color: var(--ink-soft);
  margin: 0 0 16px; padding-bottom: 10px; border-bottom: 1px solid var(--rule);
}
.slip {
  background: var(--sheet); border: 1px solid var(--rule); border-left: 3px solid var(--stamp);
  padding: 16px 20px 18px; margin-bottom: 12px; max-width: 74ch;
  animation: settle .28s ease-out;
}
.slip--back { border-left-color: var(--pencil); }
.slip--busy { border-left-color: var(--ink-soft); }
@keyframes settle { from { opacity: 0; transform: translateY(4px); } }
@media (prefers-reduced-motion: reduce) { .slip { animation: none; } .note--busy .marker { animation: none; } }
.slip-head { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom: 8px; }
.round { font-family: var(--font-mono); font-size: .72rem; color: var(--ink-soft); }
.stamp {
  font-family: var(--font-mono); font-size: .68rem; text-transform: uppercase;
  letter-spacing: .12em; padding: 3px 9px; border: 1.5px solid currentColor;
  transform: rotate(-2.2deg); color: var(--stamp);
}
.stamp--back { color: var(--pencil); }
.stamp--busy { color: var(--ink-soft); border-style: dashed; transform: none; }
.reason { margin: 0; font-size: 1.02rem; line-height: 1.5; color: var(--ink); }
.rewrite { margin: 12px 0 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }
.rewrite span { font-size: .88rem; color: var(--ink-soft); font-style: italic; }
.rewrite code {
  font-family: var(--font-mono); font-size: .82rem; color: var(--ink);
  border-bottom: 1px dashed var(--pencil); padding-bottom: 1px; background: none;
}
.override { margin: 10px 0 0; font-size: .88rem; color: var(--pencil); }

/* Answer sheet */
.answer-sheet { background: var(--sheet); border: 1px solid var(--rule); padding: 30px 34px 34px; }
.answer-sheet p, .answer-sheet li { font-size: 1.06rem; line-height: 1.62; max-width: 68ch; color: var(--ink); }
.answer-sheet h1, .answer-sheet h2, .answer-sheet h3 {
  font-weight: 600; font-size: 1.1rem; margin: 22px 0 6px; color: var(--ink);
}
.answer-sheet strong { font-weight: 600; }
.answer-sheet code { font-family: var(--font-mono); font-size: .88em; }
.answer-sheet > *:first-child { margin-top: 0; }

/* Gradio chrome */
.gradio-container label span { font-size: .84rem !important; color: var(--ink-soft) !important; }
.gradio-container textarea, .gradio-container input[type=text] {
  font-family: var(--font-mono) !important; font-size: .92rem !important; line-height: 1.55 !important;
}
button.lg, button.sm { font-family: var(--font-mono) !important; font-size: .8rem !important;
  letter-spacing: .02em !important; font-weight: 400 !important; }
button:focus-visible, textarea:focus-visible, input:focus-visible {
  outline: 2px solid var(--pencil) !important; outline-offset: 2px !important;
}
@media (max-width: 780px) {
  .right-col { border-left: none; padding-left: 0 !important; border-top: 1px solid var(--rule); padding-top: 24px !important; }
  .gradio-container { padding: 0 18px 48px !important; }
}
"""

# ── Page ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Second Opinion") as demo:

    gr.HTML(
        """
        <header class="masthead">
          <h1>Second Opinion</h1>
          <p class="deck">It reads the sources twice before it answers.</p>
          <p>Upload a PDF and ask a question. Instead of answering straight from the first
          passages it finds, the system grades them, rewrites its own search when they fall
          short, and only then writes an answer. Every round of that review is shown below.</p>
          <p class="colophon">gemma-4-26b-a4b-it &nbsp;·&nbsp; gemini-embedding-001
          &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; Chroma</p>
        </header>
        """
    )

    with gr.Row(equal_height=False, elem_classes="desk"):

        with gr.Column(scale=2, min_width=260):
            gr.HTML('<h2>Document</h2><p class="hint">One PDF at a time. Processing replaces the last one.</p>')
            file_input = gr.File(label="PDF", file_types=[".pdf"], file_count="single")
            with gr.Row():
                proc_btn = gr.Button("Process document", variant="primary", scale=3)
                clear_pdf_btn = gr.Button("Remove", variant="secondary", scale=1)
            status = gr.HTML(note("Nothing loaded yet."))

        with gr.Column(scale=3, elem_classes="right-col"):
            gr.HTML('<h2>Question</h2><p class="hint">Specific questions survive the review better than broad ones.</p>')
            q_input = gr.Textbox(
                label="Ask about the document",
                placeholder="What does the report conclude about…",
                lines=4,
                show_label=False,
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary", interactive=False, scale=3)
                clear_q_btn = gr.Button("Reset", variant="secondary", scale=1)

    gr.HTML('<h2 style="margin:44px 0 2px">Review notes</h2>'
            f'<p class="hint">The grader\'s verdict on each retrieval, up to {MAX_ITERATIONS} rounds.</p>')
    review_box = gr.HTML(empty("No review notes yet."))

    gr.HTML('<h2 style="margin:40px 0 12px">Answer</h2>')
    answer_box = gr.Markdown(ANSWER_EMPTY, elem_classes="answer-sheet", container=False)

    # ── Wiring ──
    proc_btn.click(process_pdf, inputs=file_input, outputs=[status, ask_btn])
    clear_pdf_btn.click(
        clear_pdf,
        outputs=[file_input, status, ask_btn, review_box, answer_box],
    )
    ask_btn.click(ask_question, inputs=q_input, outputs=[review_box, answer_box])
    q_input.submit(ask_question, inputs=q_input, outputs=[review_box, answer_box])
    clear_q_btn.click(clear_question, outputs=[q_input, review_box, answer_box])


if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
