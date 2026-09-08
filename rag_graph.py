import os
import time
from dotenv import load_dotenv
from typing import Optional, TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file!")

# Gemma 4 via Gemini API — temperature=0 for deterministic grading
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=api_key,
    temperature=0,
)

# gemini-embedding-001 with retrieval_document task type
# langchain-google-genai automatically switches to retrieval_query for embed_query calls
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    task_type="retrieval_document",
)

MAX_ITERATIONS = 3
CHROMA_DB_PATH = "./chroma_db"


# ── Cost tracking ────────────────────────────────────────────────────────────
# Pricing changes and isn't the same thing as "code we can verify," so it is not
# hardcoded here — that would silently mislabel every cost figure the moment the
# rate sheet moves. Set these two env vars (see .env.example) from
# https://ai.google.dev/gemini-api/docs/pricing to turn on $ cost tracking. Left
# unset, every cost_usd field below stays None and only token counts are reported.
def _float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value else None


GRADER_INPUT_PRICE_PER_1M = _float_env("GEMMA_INPUT_PRICE_PER_1M")
GRADER_OUTPUT_PRICE_PER_1M = _float_env("GEMMA_OUTPUT_PRICE_PER_1M")


def _estimate_cost_usd(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    if input_tokens is None or output_tokens is None:
        return None
    if GRADER_INPUT_PRICE_PER_1M is None or GRADER_OUTPUT_PRICE_PER_1M is None:
        return None
    return round(
        (input_tokens / 1_000_000) * GRADER_INPUT_PRICE_PER_1M
        + (output_tokens / 1_000_000) * GRADER_OUTPUT_PRICE_PER_1M,
        6,
    )


def _timed_llm_call(prompt: str, node: str, round_num: int) -> tuple[str, dict]:
    """Invoke the LLM once, recording wall-clock latency, token usage (from the
    response's standard usage_metadata) and estimated cost for that single call."""
    started = time.perf_counter()
    response = llm.invoke(prompt)
    latency_s = time.perf_counter() - started

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    metric = {
        "round": round_num,
        "node": node,
        "latency_s": round(latency_s, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": _estimate_cost_usd(input_tokens, output_tokens),
    }
    return extract_text(response), metric


def summarize_by_round(round_metrics: List[dict]) -> List[dict]:
    """Roll the flat per-node metric log up into one row per round (retrieve +
    grade [+ generate] combined), with running cumulative latency/tokens/cost.
    This is what both the UI and scripts/benchmark.py render as a table."""
    # "known" starts False and flips True the moment any node in the round
    # reports a real number — retrieve() always contributes latency but never
    # tokens/cost, and it must not blank out what grade_retrieval/generate did
    # report simply by being the node in the round that knows nothing about it.
    by_round: dict = {}
    for m in round_metrics:
        r = by_round.setdefault(m["round"], {
            "round": m["round"], "latency_s": 0.0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cost_usd": 0.0, "cost_known": False, "tokens_known": False,
        })
        r["latency_s"] += m["latency_s"]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if m[key] is not None:
                r[key] += m[key]
                r["tokens_known"] = True
        if m["cost_usd"] is not None:
            r["cost_usd"] += m["cost_usd"]
            r["cost_known"] = True

    rows = []
    cum_latency, cum_cost, cum_tokens = 0.0, 0.0, 0
    # Cumulative totals need every round so far to have reported real data —
    # unlike the per-round flags above, one round with no data here really
    # does make the running total incomplete, so this stays AND, not OR.
    cum_cost_known, cum_tokens_known = True, True
    for round_num in sorted(by_round):
        r = by_round[round_num]
        cum_latency += r["latency_s"]
        cum_cost_known = cum_cost_known and r["cost_known"]
        cum_tokens_known = cum_tokens_known and r["tokens_known"]
        if r["cost_known"]:
            cum_cost += r["cost_usd"]
        if r["tokens_known"]:
            cum_tokens += r["total_tokens"]
        rows.append({
            "round": round_num,
            "latency_s": round(r["latency_s"], 3),
            "cumulative_latency_s": round(cum_latency, 3),
            "input_tokens": r["input_tokens"] if r["tokens_known"] else None,
            "output_tokens": r["output_tokens"] if r["tokens_known"] else None,
            "total_tokens": r["total_tokens"] if r["tokens_known"] else None,
            "cumulative_total_tokens": cum_tokens if cum_tokens_known else None,
            "cost_usd": round(r["cost_usd"], 6) if r["cost_known"] else None,
            "cumulative_cost_usd": round(cum_cost, 6) if cum_cost_known else None,
        })
    return rows


def extract_text(response) -> str:
    """
    Safely extract a plain string from an LLM response.
    langchain-google-genai can return response.content as either a str
    or a list of content blocks (e.g. [{"type": "text", "text": "..."}]).
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()


class GraphState(TypedDict):
    """Shared state passed between every LangGraph node throughout the retrieval-validation loop."""

    question: str          # Original user question — never mutated
    refined_query: str     # Current search query (rewritten if needed)
    context: str           # Retrieved chunks joined as string
    reflection: str        # Latest grading output from the LLM
    answer: str            # Final generated answer
    iterations: int        # How many retrieval loops have run
    reflection_log: List[str]  # Full history of every grading step
    context_log: List[str]     # Retrieved context for every round, in order (for failure analysis)
    round_metrics: List[dict]  # Per-node latency/tokens/cost, one entry per LLM/retrieval call


def get_db() -> Chroma:
    """Open and return the persistent ChromaDB vector store."""
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve(state: GraphState) -> dict:
    """Retrieve top-k chunks using the current query (original or rewritten)."""
    query = state.get("refined_query") or state["question"]
    round_num = state.get("iterations", 0) + 1

    started = time.perf_counter()
    db = get_db()
    docs = db.similarity_search(query, k=4)
    latency_s = time.perf_counter() - started

    context = "\n\n".join(
        [f"[Chunk {i + 1}]:\n{d.page_content}" for i, d in enumerate(docs)]
    )

    # Embedding token usage isn't exposed by the LangChain wrapper's
    # similarity_search path, so only latency is tracked for this node.
    metric = {
        "round": round_num,
        "node": "retrieve",
        "latency_s": round(latency_s, 3),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
    }

    return {
        "context": context,
        "refined_query": query,
        "iterations": round_num,
        "context_log": list(state.get("context_log", [])) + [context],
        "round_metrics": list(state.get("round_metrics", [])) + [metric],
    }


# ── Node 2: Grade Retrieval ───────────────────────────────────────────────────
def grade_retrieval(state: GraphState) -> dict:
    """
    LLM judges whether the retrieved context is relevant and sufficient.
    Outputs a structured verdict and, if needed, a refined search query.
    """
    prompt = f"""You are a strict retrieval quality judge for a RAG system.

Your job: decide if the retrieved context is good enough to answer the question accurately.

Question: {state['question']}

Retrieved Context:
{state['context']}

Evaluate the context on three criteria:
1. RELEVANCE — Does it directly address the question?
2. SUFFICIENCY — Does it contain enough detail for a complete answer?
3. CONSISTENCY — Are there contradictions between chunks?

Reply in this EXACT format (no extra lines):
VERDICT: YES
REASON: <one sentence>
REFINED_QUERY: NONE

OR if context is not good enough:
VERDICT: NO
REASON: <one sentence explaining what is missing or wrong>
REFINED_QUERY: <a better, more specific search query to find the missing information>"""

    round_num = state.get("iterations", 1)
    content, metric = _timed_llm_call(prompt, "grade_retrieval", round_num)

    log_entry = f"Iteration {round_num}\n{content}"
    reflection_log = list(state.get("reflection_log", [])) + [log_entry]

    return {
        "reflection": content,
        "reflection_log": reflection_log,
        "round_metrics": list(state.get("round_metrics", [])) + [metric],
    }


# ── Node 3: Rewrite Query ─────────────────────────────────────────────────────
def rewrite_query(state: GraphState) -> dict:
    """
    Extract the REFINED_QUERY from the grader's output.
    Falls back to the original question if parsing fails.
    """
    reflection = state.get("reflection", "")
    refined = state["question"]  # safe fallback

    for line in reflection.splitlines():
        line = line.strip()
        if line.upper().startswith("REFINED_QUERY:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate and candidate.upper() != "NONE":
                refined = candidate
                break

    return {"refined_query": refined}


# ── Node 4: Generate Answer ───────────────────────────────────────────────────
def generate(state: GraphState) -> dict:
    """Generate the final answer grounded strictly in the validated context."""
    prompt = f"""You are a precise, helpful assistant. Answer the question using ONLY the provided context.
If the context is insufficient for a complete answer, clearly state what is missing — do not hallucinate.

Question: {state['question']}

Validated Context:
{state['context']}

Write a clear, structured answer grounded in the context above."""

    round_num = state.get("iterations", 1)
    content, metric = _timed_llm_call(prompt, "generate", round_num)
    return {
        "answer": content,
        "round_metrics": list(state.get("round_metrics", [])) + [metric],
    }


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_app(max_iterations: int = MAX_ITERATIONS):
    """Compile the graph with a given round cap baked into its router.

    Kept as a factory (rather than one module-level `should_continue`) so
    scripts/benchmark.py can compile cap=1, cap=2 and cap=3 versions of the
    exact same graph side by side, to measure what each cap actually buys.
    """

    def should_continue(state: GraphState) -> str:
        """
        Route to 'generate' if context passed grading or max iterations reached.
        Route to 'rewrite' otherwise to refine the query and re-retrieve.
        """
        iterations = state.get("iterations", 0)
        if iterations >= max_iterations:
            return "generate"  # forced exit — generate with best context so far

        reflection = state.get("reflection", "")
        for line in reflection.splitlines():
            if line.strip().upper().startswith("VERDICT:"):
                if "YES" in line.upper():
                    return "generate"
                break

        return "rewrite"

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_retrieval", grade_retrieval)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_retrieval")
    workflow.add_conditional_edges(
        "grade_retrieval",
        should_continue,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


app = build_app()
