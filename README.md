# Personalized Study Planner with RAG

An agentic AI project that **ingests course materials, builds a vector store, and uses an agent to generate
weekly study plans and solved practice questions**. It enforces **safety rules against hallucination** and
evaluates plan quality with **human-in-the-loop ratings stored in LangSmith**.

Runs with **no LLM key** (local TF-IDF embeddings + an extractive generator). A free LangSmith key adds
tracing and online storage of the human ratings.

## Quick start (Windows, VS Code terminal)

```
cd study-planner                      # the folder that contains requirements.txt
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt       # one command, no compiler needed
python -m study_planner ingest        # build the vector store from data/course_materials
python -m study_planner check         # tells you exactly what is missing, if anything
python -m study_planner plan --weeks 4 --hours 6
python -m pytest                      # 51 tests, no network needed
```

Or just double-click `setup_and_run.bat`.

**LangSmith key (free):** open the `.env` file, replace `PASTE_YOUR_KEY_HERE` with your key from
smith.langchain.com -> Settings -> API Keys, and save. Run `python -m study_planner check` to confirm.
The planner works without a key too: ratings are then stored locally and `sync` uploads them later.

Put your own `.md`, `.txt` or `.pdf` files in `data/course_materials/` and re-run `ingest`.

## Architecture

```
course files ─► ingest.py ─► sanitize (R1) ─► topic-aware chunks ─► embeddings.py ─► vectorstore.py (persisted)
                                                                                           │
user request ─► agent.py:  discover_topics ─► resolve_scope (R2 refuse unknown topics) ─► build_schedule
                           └► per topic: retrieve_context ─► draft_tasks ─► SafetyPolicy ─► draft_questions
                                         ─► SafetyPolicy ─► revise once with feedback ─► extractive fallback
                           └► evaluate_plan (auto scores) ─► save JSON + Markdown
                                                                   │
human rates plan (CLI) ─► feedback.py ─► outputs/feedback/ratings.jsonl ─► LangSmith feedback + golden dataset
```

| Module | Role |
|---|---|
| `ingest.py` | Loads md/txt/pdf, strips injection lines, splits by headings into chunks |
| `embeddings.py` | Hashed TF-IDF (offline) or OpenAI embeddings |
| `vectorstore.py` | Persistent numpy store with cosine search and topic filters |
| `agent.py` | The planning agent; every step is a traced LangSmith run |
| `generators.py` | `LLMGenerator` (JSON, context-only prompts) and `ExtractiveGenerator` (cannot hallucinate) |
| `safety.py` | All guardrails (below) |
| `schedule.py` | Deterministic week/time allocation, revision weeks, weak-topic weighting |
| `evaluators.py` | Automatic scores: groundedness, citations, coverage, balance, question quality |
| `feedback.py` | Human ratings, flagged-question review, LangSmith push, offline queue + `sync` |

## Safety rules (anti-hallucination)

| Rule | Enforcement |
|---|---|
| R1 | Prompt-injection lines in course files are removed at ingestion |
| R2 | Topics not covered by the material are **refused**, never invented |
| R3 | Every task/question must cite retrieved chunks; unknown citations are discarded |
| R4 | Questions and solutions must be lexically grounded in the cited context |
| R5 | URLs not present in the material are stripped |
| R6 | Arithmetic like `40 / (40 + 10) = 0.8` is re-computed; wrong equations are dropped |
| R7 | Numbers in a solution that cannot be traced are flagged `needs review` for a human |
| R8 | Malformed or too-short generator output is dropped |

Rejected LLM output triggers one revision with the rejection reasons, then a back-fill from the extractive
generator. Each plan includes a **Safety report** with counts of everything checked, dropped and flagged.

## Human-in-the-loop evaluation with LangSmith

1. Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) in `.env`. Each plan becomes one trace tree.
2. `plan` in an interactive terminal first asks you to approve or reject flagged questions, then asks for
   ratings 1-5 for overall, accuracy, workload and clarity, plus a comment.
3. Ratings are attached to the plan's root run as feedback keys `human_overall`, `human_accuracy`,
   `human_workload`, `human_clarity` (score normalised to 0-1, raw 1-5 in `value`), and `human_flagged_question_approved`.
   Automatic scores are attached as `auto_*` keys, so you can compare machine and human judgement in LangSmith.
4. Plans rated 4 or higher are added to the dataset `study-planner-golden-plans` for regression testing.
5. Non-interactive: `python -m study_planner rate <plan_id> --overall 4 --accuracy 5 --comment "..."`.
6. Ratings are always saved locally first. If LangSmith is unreachable, run `python -m study_planner sync` later.
7. `python -m study_planner report` shows mean ratings and low-rated plans.

## Commands

| Command | Purpose |
|---|---|
| `ingest` | Build the vector store |
| `topics` | List topics found |
| `plan` | `--weeks --hours --weak a,b --topics a,b --questions N --start-date YYYY-MM-DD --print --no-review --reviewer NAME --llm offline|anthropic|openai` |
| `practice TOPIC -n N` | Solved practice questions for one topic |
| `rate PLAN_ID --overall 1-5 ...` | Record human feedback |
| `check` | Diagnose setup problems (packages, .env, LangSmith key) |
| `sync`, `report`, `plans` | Push pending ratings, summarise ratings, list saved plans |

## Configuration

Only `.env` is needed, and only for LangSmith (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). No LLM key is
required: the planner uses a built-in extractive generator that builds tasks and questions directly from your
notes. (Advanced: the code can also use an LLM if `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set; every item it
writes goes through the same safety rules, and any failure falls back to the extractive generator.)

## Notes and limits

- Offline mode produces plans from definitions and worked examples already in your notes, so its questions are
  restatements rather than novel problems. An LLM key gives more varied questions, still checked by the same rules.
- Grounding is lexical (word overlap), a strong practical guard but not a proof of factual correctness; that is
  why flagged items go to a human.
- `outputs/` and `storage/` are generated; delete them freely.
