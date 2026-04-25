# 🧠 MeetingMind

**MeetingMind** is an AI-backed **meeting accountability** system. It ingests natural-language transcripts, classifies each turn (decision, action item, open question, deferral, or discussion), tracks **who promised what and by when**, and maintains **cross-meeting memory** so recurring issues surface before they quietly repeat.

## Problem

Enterprise collaboration research often cites that **poorly run meetings and unclear follow-up** cost the U.S. economy on the order of **tens of billions of dollars** per year in lost productivity. Individually, many knowledge workers report spending **on the order of 31 hours per month** in meetings, yet **decisions, owners, and deadlines** rarely live in one durable system tied back to the original conversation.

## Key features


| Area                         | What MeetingMind does                                                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Transcript understanding** | Parses numbered lines (`1. Name: text`) and merges continuations.                                                                |
| **LLM classification**       | Uses **OpenRouter** and **GPT-4o-mini** to label utterances and extract owner / verb / deadline when applicable.                 |
| **Accountability**           | Per-owner **assigned vs completed** counts, **follow-through ratio**, and stored item history.                                   |
| **Cross-meeting memory**     | **ChromaDB** (persistent embeddings) + **NetworkX** graph for issues across meetings, plus **recurring** detection.              |
| **Reporting**                | Structured JSON + **Markdown** report with *source line* under each item; flags **low confidence** and **high completion risk**. |
| **UI**                       | **Streamlit** app with category tabs, metrics, matplotlib bar chart, and graph visualization.                                    |


## System architecture (text)

```text
Transcript (txt)
   → TranscriptParser (turns + speakers + line refs)
   → DecisionExtractor (OpenRouter / GPT-4o-mini) → category, confidence, risk
        ├→ CrossMeetingMemory (Chroma + DiGraph) → similar issues, recurring
        ├→ AccountabilityTracker → JSON stats per owner
   → ReportGenerator → summary + markdown
   → Streamlit (app.py) – dashboards, charts, full report
```

## Tech stack


| Component      | Technology                                                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language       | **Python 3.11**                                                                                                                                       |
| UI             | **Streamlit**                                                                                                                                         |
| LLM API        | **openai** SDK, **OpenRouter** base URL, **gpt-4o-mini**                                                                                              |
| Vector store   | **ChromaDB** (persistent under `./data/chroma_db`)                                                                                                    |
| Graph          | **NetworkX** `DiGraph`, serialized to `./data/graph.json`                                                                                             |
| ML / metrics   | **scikit-learn** `classification_report`, **NumPy**                                                                                                   |
| Plots          | **Matplotlib**                                                                                                                                        |
| Tables         | **Pandas** (dependency / optional analysis)                                                                                                           |
| Optional stack | **PyTorch**, **transformers**, **sentence-transformers** (listed for environments that embed or train locally; core app path uses the OpenRouter API) |
| Tests          | **pytest**                                                                                                                                            |


## Setup (Python 3.11)

1. **Clone** this repository and open a terminal in the project root.
2. **Create a virtual environment** (recommended):
  ```text
   C:\Users\rahul\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
   .venv\Scripts\activate
  ```
3. **Install dependencies:**
  ```text
   C:\Users\rahul\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt
  ```
4. **Configuration:** API keys and model settings live in `config.py`. For production, prefer **environment variables** and never commit real secrets. Replace placeholder values in `config.py` with your own **OpenRouter** key and adjust `MODEL_NAME` as needed.

## How to run

```text
C:\Users\rahul\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

- Upload a `**.txt**` transcript (see `data/sample_transcript.txt` for a realistic example).  
- Set a **meeting id** and click **Process Meeting** to parse, classify, update memory, update accountability, and build the report.  
- Use **Add Prior Meeting to Memory** to **ingest** another transcript into Chroma and the graph without leaving the app.

## Example output (what to expect)

- **Tabs** for decisions, action items (with **owner / verb / deadline**), open questions, and deferrals.  
- **Warnings** for low-confidence extractions and high completion risk on action items missing owner or deadline.  
- **Dashboard** with counts, a **bar chart** of action items by owner, a **NetworkX** spring layout of stored issue nodes, and **expanders** for recurring issues.  
- A **Full Markdown Report** with *italicized* source lines under each bullet, plus a consolidated accountability section.

## Performance targets (design goals)

- **End-to-end processing** of a 30–50 turn transcript: interactive (seconds) on a typical dev machine, dominated by **LLM latency** and network round-trips.  
- **Chroma** queries: sub-second for `top-3` similarity on modest corpora.  
- **Streamlit** UI: stay responsive; long runs wrapped in `st.spinner`.  
- **Tests:** `pytest` should complete in under a minute for the unit tests (classification evaluation calls the live API and can be slower).

## Ethical considerations

- **Privacy:** transcripts can contain PII, trade secrets, and health or HR-sensitive material. **Limit retention**, **encrypt at rest** where required, and **restrict** who can open Chroma and JSON data files.  
- **Accuracy:** classifiers and summaries can **confuse** discussion with commitment. Treat outputs as **assistive**; require human review for compliance, legal, or financial commitments.  
- **Surveillance / culture:** per-owner metrics can be misused. Prefer **coaching and transparency** with teams, clear policies, and opt-in for accountability scoring.  
- **Bias and fairness:** models may reflect **linguistic bias**; validate on your domain before using metrics in performance reviews.  
- **API keys:** store secrets in **environment** or a vault, not in public repos.

## Team

- **Rahul Manohar Durshinapally**  
- **Hasith Reddy Rapolu**

Northeastern University

## License

Add your preferred license (e.g. MIT) when publishing the repository.