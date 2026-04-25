# 🧠 MeetingMind
**The AI That Remembers What Your Team Forgot It Promised**

MeetingMind is a multi-agent AI accountability system that transforms unstructured meeting transcripts into structured reports with persistent cross-session organizational memory. It implements four generative AI core components: Prompt Engineering, Fine-Tuning, Retrieval-Augmented Generation (RAG), and Synthetic Data Generation.

---

## Problem

U.S. businesses lose an estimated **$37 billion annually** to unproductive meetings (Atlassian). The average worker spends **31 hours per month** in meetings, yet decisions evaporate, action items go untracked, and the same issues get re-debated weeks later. No existing tool tracks whether what was decided actually happened — or flags when the same problem keeps surfacing meeting after meeting.

MeetingMind fixes this.

---

## Key Features

| Feature | Description |
|---|---|
| **Epistemic classification** | Distinguishes DECISION, ACTION_ITEM, OPEN_QUESTION, DEFERRAL, DISCUSSION — not just summarizes |
| **Cross-meeting memory** | ChromaDB + NetworkX graph tracks issues across sessions, flags recurring unresolved topics |
| **Decision decay scoring** | Flags action items missing owner or deadline as high completion risk |
| **Dual classification mode** | API Mode (GPT-4o-mini) for full extraction or Offline Mode (DistilBERT) for zero API cost |
| **Accountability tracking** | Per-owner follow-through ratio across meetings |
| **Source citations** | Every extracted item links to exact source transcript line — 0% hallucination rate |
| **Synthetic data generation** | 10-domain transcript generator with augmentation for testing |

---

## System Architecture

```
Transcript (.txt)
   |
   v
[Agent 1: Transcript Parser]
   Segments by speaker turn -> {turn_index, speaker, text, line_number}
   |
   v
[Agent 2: Decision Extractor]  <-- DUAL MODE
   API Mode  : GPT-4o-mini via OpenRouter (owner, verb, deadline extraction)
   Offline   : Fine-tuned DistilBERT (local, zero API cost)
   Output    : category, confidence, source_line, completion_risk
   |
   v
[Agent 3: Cross-Meeting Memory]  <-- RAG
   ChromaDB  : persistent vector storage (sentence-transformers embeddings)
   NetworkX  : knowledge graph (nodes=issues, edges=meeting sessions)
   Trigger   : node appears in 3+ meetings with no resolved status
   |
   v
[Agent 4: Accountability Tracker]
   Per-owner: assigned, completed, follow_through_ratio
   |
   v
[Agent 5: Report Generator]  <-- GPT-4o-mini
   Structured JSON + markdown with source citations
   |
   v
Streamlit Dashboard (5 tabs + knowledge graph + accountability chart)
```

---

## Generative AI Components Implemented

| Component | Implementation |
|---|---|
| **Prompt Engineering** | Systematic GPT-4o-mini prompts with category cues, JSON schema, confidence scoring |
| **Fine-Tuning** | DistilBERT fine-tuned on ICSI MRDA Corpus — 95.35% accuracy, all proposal targets exceeded |
| **RAG** | ChromaDB vector store + NetworkX graph, cross-meeting recall 0.90 |
| **Synthetic Data Generation** | 10-domain transcript generator, 20 files, 213 utterances, augmented variants |

---

## Performance Metrics

| Metric | Result | Target | Status |
|---|---|---|---|
| DECISION precision | 1.0000 | ≥ 0.80 | ✅ PASS |
| DEFERRAL precision | 1.0000 | ≥ 0.85 | ✅ PASS |
| Macro recall | 0.9385 | ≥ 0.75 | ✅ PASS |
| Overall accuracy | 95.35% | — | ✅ Strong |
| Hallucination rate | 0.00% | ≤ 5.00% | ✅ PASS |
| Cross-meeting recall | 0.9000 | ≥ 0.70 | ✅ PASS |
| pytest tests | 5/5 passing | All pass | ✅ PASS |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Web Interface | Streamlit |
| API Classifier | GPT-4o-mini via OpenRouter |
| Local Classifier | DistilBERT (HuggingFace Transformers) |
| Vector Database | ChromaDB (local persistent) |
| Knowledge Graph | NetworkX + matplotlib |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Fine-Tuning Data | ICSI MRDA Corpus (Shriberg et al. 2004) |
| Testing | pytest |
| Hosting | GitHub Pages |

---

## Setup

**Requirements:** Python 3.11, OpenRouter API key

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/rahulmanohar14/MeetingMind.git
cd MeetingMind
```

**Step 2 — Install dependencies:**
```bash
# Windows
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt

# Mac/Linux
pip3.11 install -r requirements.txt
```

**Step 3 — Set your API key:**

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

Or edit `config.py` directly and replace `OPENROUTER_API_KEY` with your key.

**Step 4 — Run the app:**
```bash
# Windows
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run app.py

# Mac/Linux
python3.11 -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## How to Use

1. **Upload a transcript** (.txt file) using the sidebar uploader
2. **Set a Meeting ID** (e.g. `meeting_001`)
3. **Select Classification Mode:**
   - 🌐 API Mode (GPT-4o-mini) — full extraction with owner/verb/deadline
   - ⚡ Offline Mode (DistilBERT) — local model, no API cost
4. **Click Process Meeting** — results appear in 5 tabs
5. **Add Prior Meetings to Memory** — upload previous meeting transcripts to build cross-meeting memory
6. **Check Dashboard** — recurring issues appear in red on the knowledge graph

Sample transcripts available in `data/`:
- `data/sample_transcript.txt` — product launch planning meeting
- `data/meeting_week1.txt`, `meeting_week2.txt`, `meeting_week3.txt` — 3-week standup sequence for cross-meeting memory demo

---

## Running Tests

```bash
# Run all unit tests
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe -m pytest tests/test_agents.py -v

# Run hallucination evaluation
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe tests/evaluate_hallucination.py

# Run cross-meeting recall evaluation
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe tests/evaluate_cross_meeting_recall.py

# Run fine-tuning classifier evaluation
C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe -c "from models.classifier import MeetingClassifier; MeetingClassifier().evaluate_sample()"
```

---

## Fine-Tuning

The fine-tuning notebook `MeetingMind_FineTuning.ipynb` reproduces the DistilBERT training:

1. Open Jupyter: `python3.11 -m jupyter notebook`
2. Open `MeetingMind_FineTuning.ipynb`
3. Set your OpenRouter API key in Cell 2
4. Run all cells (takes ~20 minutes on CPU)

Training data: ICSI MRDA Corpus (`MRDA-Corpus.zip`) + curated examples  
Output: `finetuning_outputs/` — model weights, confusion matrix, training curves, classification report

---

## Project Structure

```
MeetingMind/
├── agents/
│   ├── transcript_parser.py        # Speaker segmentation
│   ├── decision_extractor.py       # Dual-mode classification
│   ├── cross_meeting_memory.py     # RAG + knowledge graph
│   ├── accountability_tracker.py   # Follow-through tracking
│   └── report_generator.py        # JSON + markdown reports
├── models/
│   ├── classifier.py               # Batch classifier + evaluation
│   ├── local_classifier.py         # DistilBERT offline mode
│   └── meetingmind_model/          # Fine-tuned model weights
├── tests/
│   ├── test_agents.py              # pytest unit tests
│   ├── evaluate_hallucination.py   # Hallucination rate evaluation
│   └── evaluate_cross_meeting_recall.py  # Cross-meeting recall
├── data/
│   ├── sample_transcript.txt       # Example meeting transcript
│   ├── meeting_week1/2/3.txt       # 3-week standup demo sequence
│   ├── synthetic_transcripts/      # 20 generated transcripts
│   ├── example_outputs/            # Sample JSON/markdown outputs
│   └── training_dataset.csv        # Fine-tuning training data
├── docs/
│   ├── index.html                  # GitHub Pages web page
│   ├── confusion_matrix.png        # Model evaluation chart
│   ├── training_curves.png         # Training loss/accuracy
│   └── classification_report.txt   # Full sklearn report
├── app.py                          # Streamlit application
├── config.py                       # Configuration
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── MeetingMind_FineTuning.ipynb    # Fine-tuning notebook
```

---

## Ethical Considerations

- **Privacy:** All processing runs locally. No raw transcript text is persisted — only vector embeddings in ChromaDB.
- **Data control:** Users can delete meeting history at any time by clearing `data/chroma_db/` and `data/graph.json`.
- **Training data:** ICSI MRDA Corpus used under academic license. Curated examples are synthetic with no real personal data.
- **Known bias:** Trained on English-language professional meetings. Performance may degrade on non-English, informal, or highly technical meetings.
- **Accountability misuse:** Per-person metrics should inform team processes, not individual performance reviews.
- **API keys:** Store in `.env` file, never commit to version control.

---

## Team

**Rahul Manohar Durshinapally** — Core pipeline, Decision Extractor, Cross-Meeting Memory, Fine-tuning, Streamlit UI, Synthetic Data Generation

**Hasith Reddy Rapolu** — Transcript Parser, Accountability Tracker, Report Generator, Test Suite, GitHub Pages

Northeastern University — INFO 7375 Generative AI — April 2026

---

## License

MIT License