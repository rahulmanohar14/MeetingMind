import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from agents.transcript_parser import TranscriptParser
from agents.decision_extractor import DecisionExtractor
from agents.cross_meeting_memory import CrossMeetingMemory
from agents.accountability_tracker import AccountabilityTracker
from agents.report_generator import ReportGenerator
from config import DEFAULT_CLASSIFICATION_MODE

st.set_page_config(
    page_title="MeetingMind",
    page_icon="🧠",
    layout="wide",
)

DARK = "#0a1628"
ACCENT = "#14b8a6"


def _init_session():
    default_mode_label = (
        "⚡ Offline Mode (DistilBERT)"
        if (DEFAULT_CLASSIFICATION_MODE or "").strip().lower() == "local"
        else "🌐 API Mode (GPT-4o-mini)"
    )
    for k, v in {
        "extracted": None,
        "report": None,
        "md": None,
        "meeting_id": "meeting_001",
        "prior_meeting_id": "prior_001",
        "memory": None,
        "classification_mode_label": default_mode_label,
        "classification_mode_used": "api",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource
def _get_memory():
    return CrossMeetingMemory()


def _get_tracker():
    return AccountabilityTracker()


def _item_card(
    it: dict,
    *,
    show_action_metrics: bool = False,
    show_deferral_warning: bool = False,
):
    cat = (it.get("category") or "").upper()
    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.caption("**" + cat + "**" if cat else "ITEM")
        with c2:
            st.write(f"**{it.get('speaker', 'Unknown')}**")
        st.write(it.get("text", ""))
        st.caption(it.get("source_line", ""))
        if it.get("low_confidence"):
            st.warning("⚠️ Low confidence extraction")
        if cat == "ACTION_ITEM" and it.get("completion_risk") == "high":
            st.error("🔴 High completion risk — missing owner or deadline")
        if show_action_metrics and cat == "ACTION_ITEM":
            o, v, d = st.columns(3)
            with o:
                st.metric("Owner", it.get("owner") or "—")
            with v:
                st.metric("Verb", it.get("verb") or "—")
            with d:
                st.metric("Deadline", it.get("deadline") or "—")
        if show_deferral_warning and cat == "DEFERRAL":
            st.warning("⚠️ Unresolved — needs future owner assignment")


def main():
    _init_session()
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {DARK} 0%, #0f2d2a 100%); padding: 2rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
            <h1 style="color: #e2e8f0; margin:0;">🧠 MeetingMind</h1>
            <p style="color: {ACCENT}; font-size: 1.1rem; margin:0.5rem 0 0 0;">The AI That Remembers What Your Team Forgot It Promised</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        up = st.file_uploader("Upload transcript (.txt)", type=["txt"])
        mode_label = st.radio(
            "Classification Mode",
            options=["🌐 API Mode (GPT-4o-mini)", "⚡ Offline Mode (DistilBERT)"],
            index=(
                1
                if st.session_state.get("classification_mode_label")
                == "⚡ Offline Mode (DistilBERT)"
                else 0
            ),
        )
        st.session_state["classification_mode_label"] = mode_label
        if mode_label == "⚡ Offline Mode (DistilBERT)":
            st.info("Using local fine-tuned DistilBERT — faster, no API cost")
        else:
            st.info("Using GPT-4o-mini via OpenRouter — higher accuracy")
        meeting_id = st.text_input("Meeting ID", value=st.session_state.get("meeting_id", "meeting_001") or "meeting_001")
        st.session_state["meeting_id"] = meeting_id
        if st.button("🚀 Process Meeting"):
            if not up:
                st.error("Please upload a transcript file.")
            else:
                raw = up.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                st.session_state["transcript_text"] = raw
                st.session_state["do_process"] = True
        st.divider()
        prior = st.file_uploader("Add Prior Meeting to Memory", type=["txt"])
        prior_meeting_id = st.text_input(
            "Prior Meeting ID",
            value=st.session_state.get("prior_meeting_id", "prior_001") or "prior_001",
        )
        st.session_state["prior_meeting_id"] = prior_meeting_id
        st.caption("Change ID for each meeting you add (e.g. prior_001, prior_002...)")
        if st.button("📥 Ingest into memory (prior)"):
            if not prior:
                st.error("Upload a prior meeting transcript first.")
            else:
                pbytes = prior.read()
                ptext = (
                    pbytes.decode("utf-8", errors="replace")
                    if isinstance(pbytes, bytes)
                    else str(pbytes)
                )
                st.session_state["prior_text"] = ptext
                st.session_state["do_prior"] = True

    if st.session_state.get("do_prior"):
        st.session_state["do_prior"] = False
        with st.spinner("🧠 Ingesting prior meeting into memory…"):
            mem = _get_memory()
            seg = TranscriptParser().parse(st.session_state["prior_text"])
            ex = DecisionExtractor().extract(seg)
            mem.add_meeting(
                st.session_state.get("prior_meeting_id", "prior_001") or "prior_001",
                ex,
            )
            mem.save_graph()
        st.success("Prior meeting data added to cross-meeting memory.")

    if st.session_state.get("do_process"):
        st.session_state["do_process"] = False
        with st.spinner("🧠 Analyzing your meeting..."):
            text = st.session_state.get("transcript_text", "")
            segs = TranscriptParser().parse(text)
            selected_label = st.session_state.get(
                "classification_mode_label", "🌐 API Mode (GPT-4o-mini)"
            )
            selected_mode = "local" if selected_label == "⚡ Offline Mode (DistilBERT)" else "api"
            extracted = DecisionExtractor(mode=selected_mode).extract(segs)
            st.session_state["extracted"] = extracted
            st.session_state["classification_mode_used"] = selected_mode
            mem = _get_memory()
            mem.add_meeting(meeting_id, extracted)
            mem.save_graph()
            tracker = _get_tracker()
            tracker.track(extracted, meeting_id)
            tracker.save()
            acc = tracker.get_dashboard_data()
            rec = mem.check_recurring()
            rgen = ReportGenerator()
            report = rgen.generate(
                extracted,
                rec,
                acc,
                meeting_id,
            )
            st.session_state["report"] = report
            st.session_state["md"] = rgen.generate_markdown(report)
            st.session_state["recurring"] = rec
            st.session_state["acc"] = acc
            st.session_state["memory"] = mem

    extracted = st.session_state.get("extracted")
    if not extracted:
        st.info("Upload a `.txt` transcript in the sidebar and click **Process Meeting**.")
        return

    report = st.session_state.get("report") or {}
    acc = st.session_state.get("acc") or {}
    rec = st.session_state.get("recurring") or []
    mem = st.session_state.get("memory") or _get_memory()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Decisions",
            "✅ Action Items",
            "❓ Open Questions",
            "⏳ Deferred",
            "📊 Dashboard",
        ]
    )
    with tab1:
        for it in extracted:
            if (it.get("category") or "").upper() == "DECISION":
                _item_card(it)
    with tab2:
        for it in extracted:
            if (it.get("category") or "").upper() == "ACTION_ITEM":
                _item_card(it, show_action_metrics=True)
    with tab3:
        for it in extracted:
            if (it.get("category") or "").upper() == "OPEN_QUESTION":
                _item_card(it)
    with tab4:
        for it in extracted:
            if (it.get("category") or "").upper() == "DEFERRAL":
                _item_card(it, show_deferral_warning=True)
    with tab5:
        decisions_n = len(
            [x for x in extracted if (x.get("category") or "").upper() == "DECISION"]
        )
        act_n = len(
            [x for x in extracted if (x.get("category") or "").upper() == "ACTION_ITEM"]
        )
        oq_n = len(
            [
                x
                for x in extracted
                if (x.get("category") or "").upper() == "OPEN_QUESTION"
            ]
        )
        def_n = len(
            [x for x in extracted if (x.get("category") or "").upper() == "DEFERRAL"]
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decisions", decisions_n)
        c2.metric("Action items", act_n)
        c3.metric("Open questions", oq_n)
        c4.metric("Deferrals", def_n)
        st.subheader("Action items by owner")
        owners: dict = {}
        for it in extracted:
            if (it.get("category") or "").upper() == "ACTION_ITEM" and it.get("owner"):
                o = it["owner"]
                owners[o] = owners.get(o, 0) + 1
        if owners:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(list(owners.keys()), list(owners.values()), color=ACCENT)
            ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.caption("No named owners in action items yet.")
        st.subheader("Cross-meeting knowledge graph")
        G = mem.graph
        if G.number_of_nodes() == 0:
            st.caption("Graph is empty. Process meetings or add prior data.")
        else:
            rset = {r.get("node_id") for r in (rec or []) if r.get("node_id")}
            pos = nx.spring_layout(G, seed=42)
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            labels = {}
            for node in G.nodes:
                st_in = (G.nodes[node].get("resolution_status") or "") == "resolved"
                if node in rset:
                    color = "red"
                elif st_in:
                    color = "green"
                else:
                    color = "grey"
                node_text = str(G.nodes[node].get("text") or node)
                labels[node] = node_text[:20]
                x, y = pos[node]
                ax2.scatter([x], [y], c=color, s=800, zorder=2)
            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=7,
                font_color="black",
                ax=ax2,
            )
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", label="unknown", markerfacecolor="grey", markersize=8),
                Line2D([0], [0], marker="o", color="w", label="resolved", markerfacecolor="green", markersize=8),
                Line2D([0], [0], marker="o", color="w", label="recurring", markerfacecolor="red", markersize=8),
            ]
            ax2.legend(handles=legend_handles, loc="upper right", frameon=True)
            ax2.set_axis_off()
            st.pyplot(fig2)
        st.subheader("Recurring issues (3+ meetings, unresolved)")
        for r in rec or []:
            with st.expander(
                f"🔁 {r.get('text', '')[:80]}"
            ):
                st.write(
                    f"**Meetings:** {', '.join(r.get('meeting_ids', []) or [])}"
                )
                st.caption(
                    f"Status: {r.get('resolution_status', 'unresolved')}"
                )

    with st.expander("📄 Full Markdown Report"):
        mode_used = st.session_state.get("classification_mode_used", "api")
        mode_label = (
            "⚡ Offline Mode (DistilBERT)" if mode_used == "local" else "🌐 API Mode (GPT-4o-mini)"
        )
        st.caption(f"Classification mode used: {mode_label}")
        st.markdown(st.session_state.get("md") or "_(Run processing first)_")


main()
