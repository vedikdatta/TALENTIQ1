"""
TalentIQ — AI Resume Screener & Job Matcher
Hackathon build: HR-side, end-to-end resume screening with bulk upload,
semantic + skill matching, and full per-candidate explainability.
"""
import datetime as dt
import html
import io
import uuid
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import styling, bias_checker, interview_questions, email_automation, settings_store, auth, auth_ui, db
from src.matcher import (
    score_candidate,
    score_candidates_batch,
    preload_embedding_model,
    rank_candidates,
    skill_category_breakdown,
    embeddings_available,
    WEIGHTS,
)
from src.parser import extract_social_link_labels, extract_social_links, parse_document, parse_job_description
from src.sample_data import list_sample_resumes, load_sample_resume_bytes, list_sample_jds

st.set_page_config(
    page_title="TalentIQ | Enterprise Talent Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def load_ai_model():
    return preload_embedding_model()


# ============================================================
# TALENTIQ AI STARTUP
# ============================================================

if "_ai_startup_complete" not in st.session_state:
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown("## 🧭")
        st.title("TalentIQ")
        st.caption("Enterprise Talent Intelligence")

        with st.spinner("🧠 Initializing AI Engine..."):
            load_ai_model()

        st.success("AI Engine Ready ✓")

    st.session_state["_ai_startup_complete"] = True
    st.rerun()

styling.inject(st)

DEPARTMENTS = ["Engineering", "Data & AI", "Product", "Design", "Sales", "Finance", "Human Resources", "Operations"]
EDU_OPTIONS = ["Not detected", "High School", "Diploma", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD / Doctorate"]


# --------------------------------------------------------------------------
# Signed-in user
# --------------------------------------------------------------------------
def current_user():
    return st.session_state.get("user")


def current_user_id():
    user = current_user()
    return user["id"] if user else None


def sign_out():
    """Sign out of Supabase and clear the session state."""
    auth.sign_out()
    st.session_state.clear()
    st.rerun()


# --------------------------------------------------------------------------
# Session state & Supabase Data Sync
# --------------------------------------------------------------------------
def sync_from_database():
    """Attempt to hydrate in-memory state from Supabase if configured."""
    if not db.is_configured():
        return False
    try:
        reqs = db.fetch_requisitions()
        if reqs:
            st.session_state["requisitions"] = reqs
            if not st.session_state.get("active_req_id") or st.session_state["active_req_id"] not in reqs:
                st.session_state["active_req_id"] = list(reqs.keys())[0]
            num_ids = [
                int(k.split("_")[-1]) for k in reqs.keys()
                if k.startswith("req_") and k.split("_")[-1].isdigit()
            ]
            st.session_state["req_counter"] = max(num_ids) if num_ids else len(reqs)
            for rid in reqs:
                st.session_state["results"][rid] = db.fetch_results(rid)
            st.session_state["candidate_actions"] = db.fetch_all_candidate_actions()
            st.session_state["interview_guides"] = db.fetch_all_interview_guides()
            st.session_state["email_log"] = db.fetch_all_email_logs()
            return True
    except Exception as e:
        print(f"[Supabase sync error] {e}")
    return False


def init_state():
    ss = st.session_state
    ss.setdefault("requisitions", {})
    ss.setdefault("results", {})            # req_id -> list[MatchResult]
    ss.setdefault("candidate_actions", {})  # (req_id, candidate_id) -> "Shortlisted"/"Review"/"Rejected"
    ss.setdefault("active_req_id", None)
    ss.setdefault("selected_candidate", None)
    ss.setdefault("req_counter", 0)
    ss.setdefault("last_run_at", None)
    ss.setdefault("jd_draft_text", "")
    ss.setdefault("semantic_backend", "auto")
    ss.setdefault("default_weights", dict(WEIGHTS))  # org-wide default scoring weights (fractions summing to 1.0)
    ss.setdefault("custom_skills", [])      # extra taxonomy terms recruiters have added (e.g. LangGraph, vLLM)
    # Organization identity and app preferences are persisted in Supabase.
    # SMTP credentials remain in the existing private settings store and are
    # never written to the normal hr_settings database table.
    uid = current_user_id()
    if "cloud_hr_settings" not in ss:
        ss["cloud_hr_settings"] = db.ensure_hr_settings()

    cloud_settings = ss.get("cloud_hr_settings", {})

    if "org_settings" not in ss:
        ss["org_settings"] = {
            "company_name": cloud_settings.get("company_name", ""),
            "hr_sender_name": cloud_settings.get("hr_sender_name", ""),
            "hr_sender_title": cloud_settings.get("hr_sender_title", ""),
        }

    if "app_prefs" not in ss:
        ss["app_prefs"] = {
            "seed_demo_requisition": cloud_settings.get("seed_demo_requisition", True),
            "semantic_backend": cloud_settings.get("semantic_backend", "auto"),
            "default_weights": cloud_settings.get("default_weights", dict(WEIGHTS)) or dict(WEIGHTS),
            "custom_skills": cloud_settings.get("custom_skills", []) or [],
        }

    # Keep the legacy top-level state keys in sync with cloud settings.
    ss["semantic_backend"] = ss["app_prefs"].get("semantic_backend", "auto")
    ss["default_weights"] = ss["app_prefs"].get("default_weights", dict(WEIGHTS)) or dict(WEIGHTS)
    ss["custom_skills"] = list(ss["app_prefs"].get("custom_skills", []) or [])

    if "smtp_config" not in ss:
        ss["smtp_config"] = settings_store.load_smtp_config(uid)
    ss.setdefault("interview_guides", {})  # (req_id, candidate_id) -> list[question dicts]
    ss.setdefault("email_log", {})         # (req_id, candidate_id) -> list[str] status messages
    ss.setdefault("confirm_delete_req", None)

    # Sync data from Supabase once on session init
    if not ss.get("_synced_from_db"):
        ss["_synced_from_db"] = True
        sync_from_database()

    # Seed the demo requisition ONCE per session (and only if still empty and allowed)
    if not ss.get("_demo_seed_checked"):
        ss["_demo_seed_checked"] = True
        if not ss["requisitions"] and ss["app_prefs"].get("seed_demo_requisition", True):
            seed_default_requisition()


def seed_default_requisition():
    """Seed one ready-to-go requisition (Machine Learning Engineer) from the
    bundled sample JD so the app is immediately demoable."""
    jds = list_sample_jds()
    jd_text = jds.get("Machine Learning", "")
    parsed = parse_job_description(jd_text, extra_skills=st.session_state.get("custom_skills"))
    req_id = str(uuid.uuid4())
    req_data = {
        "id": req_id,
        "title": "Machine Learning Engineer",
        "department": "Data & AI",
        "jd_text": jd_text,
        "required_skills": parsed["required_skills"],
        "min_years": parsed["min_years"],
        "required_education": parsed["required_education"],
        "weights": dict(st.session_state.get("default_weights", WEIGHTS)),
        "created_at": dt.datetime.now(),
    }
    st.session_state["requisitions"][req_id] = req_data
    db.save_requisition(req_data)
    st.session_state["active_req_id"] = req_id
    st.session_state["req_counter"] = 1


def new_req_id():
    st.session_state["req_counter"] += 1
    return str(uuid.uuid4())


def get_active_req():
    rid = st.session_state["active_req_id"]
    return st.session_state["requisitions"].get(rid)


def delete_requisition(rid):
    """Remove a requisition and everything hanging off it (screened
    candidates, HR statuses, interview guides, email history). Safe to delete
    the last one — the workspace simply drops to zero active jobs."""
    ss = st.session_state
    ss["requisitions"].pop(rid, None)
    ss["results"].pop(rid, None)
    ss["candidate_actions"] = {k: v for k, v in ss["candidate_actions"].items() if k[0] != rid}
    ss["interview_guides"] = {k: v for k, v in ss["interview_guides"].items() if k[0] != rid}
    ss["email_log"] = {k: v for k, v in ss["email_log"].items() if k[0] != rid}
    remaining = list(ss["requisitions"].keys())
    ss["active_req_id"] = remaining[0] if remaining else None
    for key in ("upload_req_selector", "cand_req_selector", "dash_remove_req", "dash_remove_confirm"):
        ss.pop(key, None)
    ss["confirm_delete_req"] = None
    db.delete_requisition(rid)


def _local_now():
    """Current time in the viewer's own timezone.

    Uses the browser's timezone when Streamlit exposes it (st.context.timezone,
    newer Streamlit versions) so a cloud server running in UTC still greets
    users correctly. Falls back to the machine's local clock otherwise.
    """
    tz_name = None
    try:
        tz_name = st.context.timezone
    except Exception:
        pass
    if tz_name:
        try:
            return dt.datetime.now(ZoneInfo(tz_name))
        except Exception:
            pass
    return dt.datetime.now()


def time_of_day_greeting(now=None):
    """Good morning (05:00-11:59) / afternoon (12:00-16:59) / evening (17:00-04:59)."""
    hour = (now or _local_now()).hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    return "Good evening"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def sidebar_nav():
    if "navigate_to" in st.session_state:
        target = st.session_state.pop("navigate_to")
        if target in [
            "🏠 Dashboard",
            "📁 Resume Upload",
            "🧠 Candidate Intelligence",
            "➕ Create Job Requisition",
            "⚙️ Settings",
        ]:
            st.session_state["current_page"] = target

    st.session_state.setdefault("current_page", "🏠 Dashboard")

    
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0 18px 0;">
                <div style="background:#0d9488;width:34px;height:34px;border-radius:9px;
                            display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🧭</div>
                <div style="font-weight:800;font-size:1.15rem;color:white;">TalentIQ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user = current_user()
        if user:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.07);border-radius:10px;padding:9px 12px;margin-bottom:14px;">
                    <div style="font-weight:700;font-size:0.9rem;">{html.escape(user['name'])}</div>
                    <div style="font-size:0.75rem;opacity:0.75;word-break:break-all;">{html.escape(user['email'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        page = st.radio(
            "Navigate",
            [
                "🏠 Dashboard",
                "📁 Resume Upload",
                "🧠 Candidate Intelligence",
                "➕ Create Job Requisition",
                "⚙️ Settings",
            ],
            key="current_page",
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("AI PIPELINE")
        pipeline_steps = ["Resume Upload", "Document Extraction", "Candidate Parsing",
                          "Skill Intelligence", "Semantic Matching", "Evidence Scoring",
                          "Explainable Ranking", "HR Review"]
        st.markdown(
            "".join(f"<div style='font-size:0.82rem;opacity:0.85;padding:2px 0;'>› {s}</div>" for s in pipeline_steps),
            unsafe_allow_html=True,
        )
        st.markdown("---")
        n_results = sum(len(v) for v in st.session_state["results"].values())
        last_run = st.session_state["last_run_at"]
        last_run_str = last_run.strftime("%H:%M:%S") if last_run else "no runs yet"
        st.markdown(
            f"""
            <div style='font-size:0.78rem;line-height:1.9;opacity:0.9;'>
            🔒 Secure HR Workspace<br/>
            ⚙️ AI Engine Ready<br/>
            ✅ {n_results} candidates processed<br/>
            🕒 Last run: {last_run_str}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if st.button("Sign out", key="sidebar_sign_out"):
            sign_out()
    return page


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def page_dashboard():
    styling.topbar(st, "HR Dashboard")
    st.markdown(f"#### {time_of_day_greeting()}, HR Team")

    all_results = [r for lst in st.session_state["results"].values() for r in lst]
    n_jobs = len(st.session_state["requisitions"])
    n_resumes = len(all_results)
    all_skills = set()
    for r in all_results:
        all_skills.update(r.matched_skills)
        all_skills.update(r.extra_skills)
    n_skills = len(all_skills)
    n_shortlisted = sum(1 for v in st.session_state["candidate_actions"].values() if v == "Shortlisted")

    c1, c2, c3, c4 = st.columns(4)
    with c1: styling.kpi_card(st, "Active Jobs", n_jobs)
    with c2: styling.kpi_card(st, "Resumes Screened", n_resumes)
    with c3: styling.kpi_card(st, "Skills Identified", n_skills)
    with c4: styling.kpi_card(st, "Candidates Shortlisted", n_shortlisted)

    st.write("")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="tiq-card"><h4>Active Hiring Requisitions</h4>', unsafe_allow_html=True)
        rows = []
        for rid, req in st.session_state["requisitions"].items():
            n = len(st.session_state["results"].get(rid, []))
            avg = (
                round(sum(r.overall_score for r in st.session_state["results"][rid]) / n, 1)
                if n else "—"
            )
            rows.append({
                "Job Title": req["title"],
                "Department": req["department"],
                "Candidates": n,
                "Avg. Match": avg,
                "Status": "Screening Complete" if n else "Awaiting Resumes",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("No active job requisitions. Add one from **Create Job Requisition** in the sidebar "
                    "whenever you're ready to hire.")
        st.markdown("</div>", unsafe_allow_html=True)

        if rows:
            with st.expander("🗑 Remove a job requisition"):
                reqs_now = st.session_state["requisitions"]
                remove_rid = st.selectbox(
                    "Requisition to remove", options=list(reqs_now.keys()),
                    format_func=lambda x: f"{reqs_now[x]['title']} · {reqs_now[x]['department']}",
                    key="dash_remove_req",
                )
                st.caption("This also removes its screened candidates, HR statuses, and email/interview history. "
                           "You can remove every requisition to bring Active Jobs down to zero.")
                confirmed = st.checkbox("Yes, permanently remove this requisition", key="dash_remove_confirm")
                if st.button("Remove requisition", disabled=not confirmed, key="dash_remove_btn"):
                    delete_requisition(remove_rid)
                    st.rerun()

        if all_results:
            st.markdown('<div class="tiq-card"><h4>Top Skills Across Applicant Pool</h4>', unsafe_allow_html=True)
            freq = {}
            for r in all_results:
                for s in r.matched_skills + r.extra_skills:
                    freq[s] = freq.get(s, 0) + 1
            top = sorted(freq.items(), key=lambda x: -x[1])[:10]
            if top:
                fig = go.Figure(go.Bar(
                    x=[v for _, v in top][::-1], y=[k for k, _ in top][::-1],
                    orientation="h", marker_color="#0d9488",
                ))
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
                st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if all_results:
            st.markdown('<div class="tiq-card"><h4>Candidate Match Distribution</h4>', unsafe_allow_html=True)
            statuses = ["Strong Match", "Good Match", "Review", "Low Match"]
            counts = [sum(1 for r in all_results if r.status == s) for s in statuses]
            fig = go.Figure(go.Pie(
                labels=statuses, values=counts, hole=0.62,
                marker_colors=["#0d9488", "#2563eb", "#d97706", "#dc2626"],
            ))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="tiq-card"><h4>Experience Distribution</h4>', unsafe_allow_html=True)
            years = [r.years_experience for r in all_results]
            fig2 = go.Figure(go.Histogram(x=years, marker_color="#14b8a6", nbinsx=8))
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                                xaxis_title="Years", yaxis_title="Candidates")
            st.plotly_chart(fig2, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="tiq-card"><h4>No screenings yet</h4>'
                '<p style="color:#6b7290;">Head to <b>Resume Upload</b> to run your first AI screening — '
                'or load the bundled sample candidates for an instant demo.</p></div>',
                unsafe_allow_html=True,
            )

    if all_results:
        st.markdown('<div class="tiq-card"><h4>Skill Gap Analysis</h4>'
                     '<p style="color:#6b7290;margin-top:-8px;">Most frequently missing skills across the current applicant pool — useful for sourcing and internal upskilling decisions.</p>',
                     unsafe_allow_html=True)
        gap_freq = {}
        for r in all_results:
            for s in r.missing_skills:
                gap_freq[s] = gap_freq.get(s, 0) + 1
        top_gaps = sorted(gap_freq.items(), key=lambda x: -x[1])[:10]
        if top_gaps:
            fig3 = go.Figure(go.Bar(
                x=[k for k, _ in top_gaps], y=[v for _, v in top_gaps],
                marker_color="#dc2626",
            ))
            fig3.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("No missing-skill gaps detected across the current pool.")
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Resume Upload
# --------------------------------------------------------------------------
def _normalized_weights(sem, skl, exp, edu):
    """Turn four 0-100 slider values into fractions that sum to exactly 1.0."""
    total = sem + skl + exp + edu
    if total <= 0:
        return dict(WEIGHTS)
    return {
        "semantic_relevance": sem / total,
        "skill_alignment": skl / total,
        "experience_evidence": exp / total,
        "education_evidence": edu / total,
    }


def manage_requisition_panel(rid, req):
    """Requisition lifecycle management: edit core details, add/remove
    required skills manually, override scoring weights per-role, and
    delete/archive the requisition — closing the CRUD gap where only the
    raw JD text could be edited after creation."""
    with st.expander("⚙️ Manage this requisition (edit details, skills, weights, or delete)", expanded=False):
        with st.form(f"edit_req_{rid}"):
            c1, c2 = st.columns(2)
            new_title = c1.text_input("Job Title", value=req["title"])
            dept_idx = DEPARTMENTS.index(req["department"]) if req["department"] in DEPARTMENTS else 0
            new_department = c2.selectbox("Department", DEPARTMENTS, index=dept_idx)

            c3, c4 = st.columns(2)
            new_min_years = c3.number_input("Minimum years of experience", min_value=0.0, max_value=30.0,
                                             value=float(req["min_years"]), step=1.0)
            edu_idx = EDU_OPTIONS.index(req["required_education"]) if req["required_education"] in EDU_OPTIONS else 0
            new_education = c4.selectbox("Minimum education required", EDU_OPTIONS, index=edu_idx)

            st.markdown("**Required skills** (auto-extracted from the JD — add or remove as needed)")
            keep_skills = st.multiselect(
                "Keep these required skills", options=req["required_skills"],
                default=req["required_skills"], label_visibility="collapsed",
            )
            add_skill_text = st.text_input("Add a required skill (e.g. LangGraph, vLLM, SvelteKit)", value="")

            st.markdown("**Scoring weights for this requisition**")
            dw = req.get("weights", st.session_state["default_weights"])
            wc1, wc2, wc3, wc4 = st.columns(4)
            w_sem = wc1.slider("Semantic %", 0, 100, round(dw["semantic_relevance"] * 100))
            w_skl = wc2.slider("Skills %", 0, 100, round(dw["skill_alignment"] * 100))
            w_exp = wc3.slider("Experience %", 0, 100, round(dw["experience_evidence"] * 100))
            w_edu = wc4.slider("Education %", 0, 100, round(dw["education_evidence"] * 100))
            st.caption(f"Weights are normalized automatically (currently sum to {w_sem + w_skl + w_exp + w_edu}%).")

            save = st.form_submit_button("💾 Save changes", type="primary")

        if save:
            final_skills = list(dict.fromkeys(keep_skills + ([add_skill_text.strip()] if add_skill_text.strip() else [])))
            if add_skill_text.strip() and add_skill_text.strip() not in st.session_state["custom_skills"]:
                st.session_state["custom_skills"].append(add_skill_text.strip())
            req["title"] = new_title or req["title"]
            req["department"] = new_department
            req["min_years"] = float(new_min_years)
            req["required_education"] = new_education
            req["required_skills"] = final_skills
            req["weights"] = _normalized_weights(w_sem, w_skl, w_exp, w_edu)
            st.success("Requisition updated. Re-run screening for these changes to reflect in existing scores.")
            st.rerun()

        st.markdown("---")
        st.caption("Deleting a requisition also removes its screened candidates, HR statuses, and email/interview history.")
        if st.session_state["confirm_delete_req"] != rid:
            if st.button("🗑 Delete this requisition", key=f"del_req_{rid}"):
                st.session_state["confirm_delete_req"] = rid
                st.rerun()
        else:
            st.warning(f"Delete **{req['title']}** and all its screened candidates? This cannot be undone.")
            dc1, dc2 = st.columns(2)
            if dc1.button("Yes, delete permanently", key=f"del_req_confirm_{rid}", type="primary"):
                delete_requisition(rid)
                st.rerun()
            if dc2.button("Cancel", key=f"del_req_cancel_{rid}"):
                st.session_state["confirm_delete_req"] = None
                st.rerun()


def page_upload():
    styling.topbar(st, "Resume Screening")

    reqs = st.session_state["requisitions"]
    if not reqs:
        st.warning("Create a job requisition first (see **Create Job Requisition** in the sidebar).")
        return

    req_labels = {rid: f"{r['title']} · {r['department']}" for rid, r in reqs.items()}

    # --- FIX: stable key on the requisition selector so role switching sticks ---
    if "upload_req_selector" not in st.session_state or \
       st.session_state["upload_req_selector"] not in reqs:
        st.session_state["upload_req_selector"] = (
            st.session_state.get("active_req_id") or list(reqs.keys())[0]
        )

    rid = st.selectbox(
        "Screening for requisition",
        options=list(reqs.keys()),
        format_func=lambda x: req_labels[x],
        key="upload_req_selector",
    )
    st.session_state["active_req_id"] = rid
    req = reqs[rid]

    st.markdown(
        f'<div class="tiq-card"><h4>{req["title"]}</h4>'
        f'<p style="color:#6b7290;">{req["department"]} · Required skills: '
        f'{", ".join(req["required_skills"][:8]) if req["required_skills"] else "auto-detected from JD"}</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("View / edit job description text", expanded=False):
        edited = st.text_area("Job description", value=req["jd_text"], height=180, key=f"jd_edit_{rid}")
        if edited != req["jd_text"]:
            req["jd_text"] = edited
            parsed = parse_job_description(edited, extra_skills=st.session_state["custom_skills"])
            req["required_skills"] = parsed["required_skills"]
            req["min_years"] = parsed["min_years"]
            req["required_education"] = parsed["required_education"]

    manage_requisition_panel(rid, req)
    if rid not in st.session_state["requisitions"]:
        return  # requisition was just deleted

    #st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    st.markdown("##### 📤 Drop resumes here")
    st.caption("Upload PDF, DOCX, or TXT resumes in bulk — supports multiple files at once.")

    uploaded_files = st.file_uploader(
        "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    use_sample = st.checkbox("Also include the bundled sample candidates (8 demo resumes)", value=(len(uploaded_files or []) == 0))

    existing_results = st.session_state["results"].get(rid, [])
    upload_mode = "Append to existing pool"
    if existing_results:
        upload_mode = st.radio(
            f"This requisition already has {len(existing_results)} screened candidate"
            f"{'s' if len(existing_results) != 1 else ''}. What should this run do?",
            ["Append to existing pool", "Replace existing pool"],
            horizontal=True,
        )

    file_records = []
    if uploaded_files:
        for f in uploaded_files:
            file_records.append((f.name, f.getvalue(), f"{len(f.getvalue())/1024:.0f} KB"))
    if use_sample:
        for name in list_sample_resumes():
            data = load_sample_resume_bytes(name)
            file_records.append((name, data, f"{len(data)/1024:.0f} KB"))

    if file_records:
        st.write(f"**{len(file_records)} resumes selected**")
        preview_df = pd.DataFrame([{"Filename": n, "Size": s, "Status": "Ready"} for n, _, s in file_records])
        st.dataframe(preview_df, width='stretch', hide_index=True, height=min(38 * (len(preview_df) + 1), 300))

    run = st.button("⚡ Run AI Screening", type="primary", disabled=(len(file_records) == 0))
    #st.markdown("</div>", unsafe_allow_html=True)

    if run and file_records:
        jd_parsed = {
            "raw_text": req["jd_text"],
            "required_skills": req["required_skills"],
            "min_years": req["min_years"],
            "required_education": req["required_education"],
        }
        req_weights = req.get("weights", st.session_state["default_weights"])
        progress = st.progress(0.0, text="Parsing resumes…")
        # ---------------------------------------------------------
        # Phase 1: Parse all resumes
        # ---------------------------------------------------------
        parsed_resumes = []
        filenames = []

        for i, (fname, fbytes, _) in enumerate(file_records):

            doc = parse_document(
                fname,
                fbytes,
                extra_skills=st.session_state["custom_skills"],
            )

            parsed_resumes.append(doc)
            filenames.append(fname)
            db.upload_resume_file(fname, fbytes, req_id=rid)

            if (i + 1) % 5 == 0 or i + 1 == len(file_records):
                progress.progress(
                    (i + 1) / len(file_records),
                    text=f"Parsed {i + 1}/{len(file_records)} resumes…",
                )


        # ---------------------------------------------------------
        # Phase 2: Batch AI scoring
        # ---------------------------------------------------------
        progress.progress(
            0.0,
            text="AI matching resumes in batches…",
        )

        new_results = score_candidates_batch(
            parsed_resumes,
            jd_parsed,
            filenames=filenames,
            semantic_backend=st.session_state["semantic_backend"],
            weights=req_weights,
        )

        progress.progress(
            1.0,
            text=f"Scored {len(new_results)} resumes.",
        )
        if existing_results and upload_mode == "Append to existing pool":
            # Non-destructive merge: keyed by a stable candidate_id (filename +
            # name + email + phone) so re-uploading the same resume refreshes
            # its score in place instead of creating a duplicate row, while
            # genuinely new resumes are added alongside the previous pool.
            by_id = {r.candidate_id: r for r in existing_results}
            for nr in new_results:
                by_id[nr.candidate_id] = nr
            combined = list(by_id.values())
        else:
            combined = new_results

        combined = rank_candidates(combined, weights=req_weights)
        st.session_state["results"][rid] = combined
        st.session_state["last_run_at"] = dt.datetime.now()
        db.save_results(rid, combined, st.session_state.get("candidate_actions"))
        progress.empty()

        

        st.success(
            f"Screening complete — {len(combined)} candidates now in the pool for this requisition "
            f"({len(new_results)} just processed)."
        )

        st.balloons()
        import time
        time.sleep(0.8)
        
        # Ask the next rerun to open Candidate Intelligence.
        # We use a separate flag because current_page belongs to the
        # Streamlit radio widget and should not be changed after it is created.
        st.session_state["navigate_to"] = "🧠 Candidate Intelligence"

        st.rerun()


# --------------------------------------------------------------------------
# Candidate Intelligence
# --------------------------------------------------------------------------
def _initials_avatar(name, size=42):
    parts = name.split()
    initials = "".join(p[0] for p in parts[:2]).upper() if parts else "?"
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:#0d9488;color:white;'
        f'display:flex;align-items:center;justify-content:center;font-weight:700;font-size:{size*0.38}px;">'
        f"{initials}</div>"
    )


def _social_url(result, field_name):
    """Keep results created before social-link support displayable."""
    value = getattr(result, field_name, None)
    if value:
        return value
    return extract_social_links(getattr(result, "raw_text", "")).get(field_name, "Not detected")


@st.dialog("Candidate Profile", width="large")
def candidate_profile_dialog(rid, result):
    key_prefix = f"{rid}::{result.candidate_id}"
    c1, c2 = st.columns([1, 5])
    with c1:
        st.markdown(_initials_avatar(result.candidate_name, 56), unsafe_allow_html=True)
    with c2:
        st.markdown(f"### {result.candidate_name}")
        st.caption(f"{result.email}  ·  {result.phone}")
        social_fields = (
            ("LinkedIn", "linkedin_url"),
            ("GitHub", "github_url"),
            ("Portfolio", "portfolio_url"),
        )
        social_labels = extract_social_link_labels(getattr(result, "raw_text", ""))
        social_lines = [
            f"**{social_labels.get(field, label)}:** [{_social_url(result, field)}]({_social_url(result, field)})"
            for label, field in social_fields
            if _social_url(result, field) != "Not detected"
        ]
        if social_lines:
            st.markdown("  \n".join(social_lines))
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin:8px 0 16px 0;">'
        f'<div>{styling.status_badge_html(result.status)}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:#131b3d;">{result.overall_score}/100</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    current_action = st.session_state["candidate_actions"].get((rid, result.candidate_id), "—")
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("⭐ Shortlist", key=f"short_{key_prefix}", type="primary" if current_action != "Shortlisted" else "secondary"):
        st.session_state["candidate_actions"][(rid, result.candidate_id)] = "Shortlisted"
        db.update_candidate_action(rid, result.candidate_name, "Shortlisted")
        st.rerun()
    if a2.button("🔎 Move to Review", key=f"rev_{key_prefix}"):
        st.session_state["candidate_actions"][(rid, result.candidate_id)] = "Review"
        db.update_candidate_action(rid, result.candidate_name, "Review")
        st.rerun()
    if a3.button("✖ Reject", key=f"rej_{key_prefix}"):
        st.session_state["candidate_actions"][(rid, result.candidate_id)] = "Rejected"
        db.update_candidate_action(rid, result.candidate_name, "Rejected")
        st.rerun()
    profile_txt = build_profile_text(result)
    a4.download_button("⬇ Download", data=profile_txt, file_name=f"{result.candidate_name.replace(' ', '_')}_profile.txt",
                        key=f"dl_{key_prefix}")

    if current_action != "—":
        st.caption(f"HR status: **{current_action}**")

    st.markdown("---")
    st.markdown("##### AI Match Analysis")
    engine_display = {"tfidf": "TF-IDF", "embeddings": "sentence-transformer embeddings"}.get(
        result.semantic_engine, result.semantic_engine
    )
    st.caption(f"Semantic Relevance computed via {engine_display}.")
    factor_labels = {
        "semantic_relevance": "Semantic Relevance",
        "skill_alignment": "Skill Alignment",
        "experience_evidence": "Experience Evidence",
        "education_evidence": "Education Evidence",
    }
    for key, label in factor_labels.items():
        val = result.factors[key]
        st.write(f"{label} — **{val:.0f}%**")
        st.progress(min(max(val / 100, 0), 1.0))

    st.caption(f"Education: {result.education}  ·  Experience: {result.years_experience:.0f} years")

    st.markdown("##### Why This Match")
    st.write(result.rationale)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**✅ Matched Skills**")
        if result.matched_skills:
            st.markdown("".join(f'<span class="tiq-skill-chip chip-match">{s}</span>' for s in result.matched_skills), unsafe_allow_html=True)
        else:
            st.caption("No direct overlap with required skills detected.")
    with mc2:
        st.markdown("**⚠️ Missing / Not Detected**")
        if result.missing_skills:
            st.markdown("".join(f'<span class="tiq-skill-chip chip-missing">{s}</span>' for s in result.missing_skills), unsafe_allow_html=True)
        else:
            st.caption("No gaps against the role's detected required skills.")

    if result.extra_skills:
        st.markdown("**➕ Additional Skills on Resume**")
        st.markdown("".join(f'<span class="tiq-skill-chip chip-extra">{s}</span>' for s in result.extra_skills[:20]), unsafe_allow_html=True)

    with st.expander("📄 View raw resume text"):
        if result.raw_text and result.raw_text.strip():
            st.caption("Original extracted text — use this to verify where a skill or experience claim was found.")
            st.text_area(
                "Raw resume text", value=result.raw_text, height=280,
                key=f"rawtext_{key_prefix}", label_visibility="collapsed",
            )
        else:
            st.caption("No extractable text was found in this document.")

    if result.fairness_audit:
        fa = result.fairness_audit
        st.markdown("---")
        st.markdown("##### 🛡️ Fairness Audit")
        st.caption(
            f"Blind re-score (name/email/phone redacted): **{fa['blind_score']}** vs. original "
            f"**{fa['original_score']}** — Δ {fa['delta']:+.1f}. {fa['note']}"
        )

    req = st.session_state["requisitions"].get(rid, {})
    req_title = req.get("title", "")

    st.markdown("---")
    st.markdown("##### 🎯 Suggested Interview Questions")
    guide_key = (rid, result.candidate_id)
    if st.button("Generate Interview Questions", key=f"gen_q_{key_prefix}"):
        questions = interview_questions.generate_interview_questions(result, req_title)
        st.session_state["interview_guides"][guide_key] = questions
        db.save_interview_guide(rid, result.candidate_name, questions)
    guide = st.session_state["interview_guides"].get(guide_key)
    if guide:
        for i, q in enumerate(guide):
            st.markdown(f"**{q['category']}**")
            question = st.text_area(
                f"Question {i + 1}", value=q["question"], height=90,
                key=f"guide_q_{key_prefix}_{i}", label_visibility="collapsed",
            )
            q["question"] = question
            if st.button("🗑 Delete question", key=f"del_guide_q_{key_prefix}_{i}"):
                guide.pop(i)
                db.save_interview_guide(rid, result.candidate_name, guide)
                st.rerun()
        if st.button("➕ Add question", key=f"add_guide_q_{key_prefix}"):
            guide.append({"category": "Custom", "question": ""})
            db.save_interview_guide(rid, result.candidate_name, guide)
            st.rerun()
        guide_txt = interview_questions.format_questions_text(result.candidate_name, req_title, guide)
        st.download_button("⬇ Download interview guide", data=guide_txt,
                            file_name=f"{result.candidate_name.replace(' ', '_')}_interview_guide.txt",
                            key=f"dl_guide_{key_prefix}")
    else:
        st.caption("Generates 8 targeted questions from this candidate's matched skills, skill gaps, experience, and role fit.")

    st.markdown("---")
    st.markdown("##### 📧 Candidate Communication")
    org = st.session_state["org_settings"]
    default_template = email_automation.ACTION_TO_DEFAULT_TEMPLATE.get(current_action, "Selection / Offer")
    template_names = list(email_automation.TEMPLATES.keys())
    template_choice = st.selectbox(
        "Email type", template_names, index=template_names.index(default_template),
        key=f"tpl_{key_prefix}",
    )
    ctx = {
        "candidate_name": result.candidate_name,
        "job_title": req_title,
        "company_name": org["company_name"],
        "hr_sender_name": org["hr_sender_name"],
        "hr_sender_title": org["hr_sender_title"],
    }
    default_subject, default_body = email_automation.render_template(template_choice, ctx)
    subject = st.text_input("Subject", value=default_subject, key=f"subj_{key_prefix}_{template_choice}")
    body = st.text_area("Body", value=default_body, height=220, key=f"body_{key_prefix}_{template_choice}")

    to_email = result.email
    ec1, ec2 = st.columns(2)
    if ec1.button(f"✉️ Send to {to_email}", key=f"send_{key_prefix}", type="primary",
                  disabled=(to_email == "Not detected")):
        success, message = email_automation.send_email_smtp(st.session_state["smtp_config"], to_email, subject, body)
        st.session_state["email_log"].setdefault(guide_key, []).append(message)
        db.save_email_log(rid, result.candidate_name, message)
        if success:
            st.success(message)
        else:
            st.error(message)
    eml_bytes = email_automation.build_eml_bytes(
        st.session_state["smtp_config"].get("sender_email", ""), to_email, subject, body
    )
    ec2.download_button("⬇ Download as .eml", data=eml_bytes,
                         file_name=f"{result.candidate_name.replace(' ', '_')}_email.eml", mime="message/rfc822",
                         key=f"eml_{key_prefix}")
    if to_email == "Not detected":
        st.caption("No email address was detected on this resume — sending is disabled, but you can still edit and download the draft.")

    log = st.session_state["email_log"].get(guide_key, [])
    if log:
        st.caption(f"Last activity: {log[-1]}")


def build_profile_text(result):
    lines = [
        f"TalentIQ Candidate Profile",
        f"===========================",
        f"Name: {result.candidate_name}",
        f"Email: {result.email}",
        f"Phone: {result.phone}",
        f"LinkedIn: {_social_url(result, 'linkedin_url')}",
        f"GitHub: {_social_url(result, 'github_url')}",
        f"Portfolio: {_social_url(result, 'portfolio_url')}",
        f"Education: {result.education}",
        f"Experience: {result.years_experience:.0f} years",
        f"",
        f"Overall Match Score: {result.overall_score}/100 ({result.status})",
        f"",
        "Factor Breakdown:",
    ]
    for k, v in result.factors.items():
        lines.append(f"  - {k.replace('_', ' ').title()}: {v:.0f}%")
    lines += [
        "",
        f"Why This Match: {result.rationale}",
        "",
        "Matched Skills: " + (", ".join(result.matched_skills) or "None"),
        "Missing Skills: " + (", ".join(result.missing_skills) or "None"),
    ]
    return "\n".join(lines)


def page_candidates():
    styling.topbar(st, "Candidate Intelligence")

    reqs = st.session_state["requisitions"]
    if not reqs:
        st.warning("No job requisitions yet.")
        return

    req_labels = {rid: f"{r['title']} · {r['department']}" for rid, r in reqs.items()}

    # --- FIX: stable key on the requisition selector so role switching sticks ---
    if "cand_req_selector" not in st.session_state or \
       st.session_state["cand_req_selector"] not in reqs:
        st.session_state["cand_req_selector"] = (
            st.session_state.get("active_req_id") or list(reqs.keys())[0]
        )

    rid = st.selectbox(
        "Requisition",
        options=list(reqs.keys()),
        format_func=lambda x: req_labels[x],
        key="cand_req_selector",
    )
    st.session_state["active_req_id"] = rid
    results = st.session_state["results"].get(rid, [])

    if not results:
        st.info("No candidates screened yet for this requisition. Go to **Resume Upload** to run AI screening.")
        return

    st.markdown(f"**{len(results)} candidates analyzed** for *{reqs[rid]['title']}*")

    search = st.text_input("🔍 Search candidate name or skill", "")
    tabs = st.tabs(["All", "Strong Match", "Good Match", "Review", "Low Match", "Shortlisted"])
    tab_filters = ["All", "Strong Match", "Good Match", "Review", "Low Match", "Shortlisted"]

    last_filtered = {}  # label -> filtered list, so export-below-tabs knows what's showing
    for tab, label in zip(tabs, tab_filters):
        with tab:
            filtered = results
            if label == "Shortlisted":
                filtered = [r for r in results if st.session_state["candidate_actions"].get((rid, r.candidate_id)) == "Shortlisted"]
            elif label != "All":
                filtered = [r for r in results if r.status == label]
            if search:
                s = search.lower()
                filtered = [r for r in filtered if s in r.candidate_name.lower()
                            or any(s in sk.lower() for sk in r.matched_skills + r.extra_skills)]
            last_filtered[label] = filtered

            if not filtered:
                st.caption("No candidates in this view.")
                continue

            header_cols = st.columns([0.5, 2.2, 1.1, 1.6, 1.2, 1.2, 1.3])
            for c, h in zip(header_cols, ["Rank", "Candidate", "Score", "Status", "Experience", "HR Status", ""]):
                c.markdown(f"**{h}**")

            for i, r in enumerate(filtered, start=1):
                cols = st.columns([0.5, 2.2, 1.1, 1.6, 1.2, 1.2, 1.3])
                cols[0].write(f"{results.index(r) + 1:02d}")
                with cols[1]:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;">{_initials_avatar(r.candidate_name, 30)}'
                        f'<div><b>{r.candidate_name}</b><br/><span style="font-size:0.75rem;color:#6b7290;">{r.email}</span></div></div>',
                        unsafe_allow_html=True,
                    )
                cols[2].markdown(f"**{r.overall_score:.0f}**/100")
                cols[3].markdown(styling.status_badge_html(r.status), unsafe_allow_html=True)
                cols[4].write(f"{r.years_experience:.0f} yrs")
                action = st.session_state["candidate_actions"].get((rid, r.candidate_id), "—")
                cols[5].write(action)
                if cols[6].button("View Profile", key=f"view_{rid}_{r.candidate_id}_{label}"):
                    candidate_profile_dialog(rid, r)

    st.markdown("---")
    candidate_comparison_panel(rid, results)

    st.markdown("---")
    st.markdown("##### ⬇ Export")
    export_scope = st.radio(
        "Export which candidates?",
        ["All candidates", "Currently filtered / searched (All tab)", "Shortlisted only"],
        horizontal=True,
    )
    if export_scope == "Shortlisted only":
        export_rows = [r for r in results if st.session_state["candidate_actions"].get((rid, r.candidate_id)) == "Shortlisted"]
    elif export_scope == "Currently filtered / searched (All tab)":
        export_rows = last_filtered.get("All", results)
    else:
        export_rows = results

    export_df = pd.DataFrame([{
        "Rank": results.index(r) + 1,
        "Candidate": r.candidate_name,
        "Email": r.email,
        "Phone": r.phone,
        "LinkedIn": _social_url(r, "linkedin_url"),
        "GitHub": _social_url(r, "github_url"),
        "Portfolio": _social_url(r, "portfolio_url"),
        "Score": r.overall_score,
        "Status": r.status,
        "Semantic Relevance": r.factors["semantic_relevance"],
        "Skill Alignment": r.factors["skill_alignment"],
        "Experience Evidence": r.factors["experience_evidence"],
        "Education Evidence": r.factors["education_evidence"],
        "Experience (yrs)": r.years_experience,
        "Education": r.education,
        "Matched Skills": ", ".join(r.matched_skills),
        "Missing Skills": ", ".join(r.missing_skills),
        "HR Status": st.session_state["candidate_actions"].get((rid, r.candidate_id), ""),
    } for r in export_rows])
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇ Export {len(export_rows)} candidate{'s' if len(export_rows) != 1 else ''} (CSV)",
        data=csv, file_name=f"{reqs[rid]['title'].replace(' ', '_')}_shortlist.csv", mime="text/csv",
        disabled=(len(export_rows) == 0),
    )


def candidate_comparison_panel(rid, results):
    """Side-by-side comparison of 2-3 candidates: factor breakdown and a
    skills-overlap matrix against the role's required skills, so the
    deciding factor in shortlisting doesn't require flipping between
    separate profile dialogs."""
    st.markdown('<div class="tiq-card"><h4>🆚 Compare Candidates</h4>'
                '<p style="color:#6b7290;">Pick 2 or 3 candidates to compare head-to-head.</p>',
                unsafe_allow_html=True)

    by_id = {r.candidate_id: r for r in results}
    labels = {r.candidate_id: f"{r.candidate_name} ({r.overall_score:.0f}/100)" for r in results}
    picked = st.multiselect(
        "Candidates to compare", options=list(by_id.keys()),
        format_func=lambda cid: labels[cid], max_selections=3, key=f"compare_pick_{rid}",
    )

    if len(picked) < 2:
        st.caption("Select at least 2 candidates to see a comparison.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    chosen = [by_id[cid] for cid in picked]
    # Disambiguate identical display names (e.g. two "John Smith" resumes)
    # so their comparison columns don't collide and silently overwrite.
    name_counts = {}
    for c in chosen:
        name_counts[c.candidate_name] = name_counts.get(c.candidate_name, 0) + 1
    col_name = {
        c.candidate_id: (c.candidate_name if name_counts[c.candidate_name] == 1
                          else f"{c.candidate_name} ({c.candidate_id[:4]})")
        for c in chosen
    }

    rows = [
        {"Metric": "Overall Score", **{col_name[c.candidate_id]: f"{c.overall_score:.0f}/100" for c in chosen}},
        {"Metric": "Status", **{col_name[c.candidate_id]: c.status for c in chosen}},
        {"Metric": "Semantic Relevance", **{col_name[c.candidate_id]: f"{c.factors['semantic_relevance']:.0f}%" for c in chosen}},
        {"Metric": "Skill Alignment", **{col_name[c.candidate_id]: f"{c.factors['skill_alignment']:.0f}%" for c in chosen}},
        {"Metric": "Experience Evidence", **{col_name[c.candidate_id]: f"{c.factors['experience_evidence']:.0f}%" for c in chosen}},
        {"Metric": "Education Evidence", **{col_name[c.candidate_id]: f"{c.factors['education_evidence']:.0f}%" for c in chosen}},
        {"Metric": "Years of Experience", **{col_name[c.candidate_id]: f"{c.years_experience:.0f}" for c in chosen}},
        {"Metric": "Education", **{col_name[c.candidate_id]: c.education for c in chosen}},
        {"Metric": "Matched Skills (#)", **{col_name[c.candidate_id]: str(len(c.matched_skills)) for c in chosen}},
        {"Metric": "Missing Skills (#)", **{col_name[c.candidate_id]: str(len(c.missing_skills)) for c in chosen}},
    ]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("**Required skills — overlap matrix**")
    req_skills = sorted({s for c in chosen for s in (c.matched_skills + c.missing_skills)})
    if req_skills:
        matrix_rows = []
        for skill in req_skills:
            row = {"Skill": skill}
            for c in chosen:
                row[col_name[c.candidate_id]] = "✅" if skill in c.matched_skills else "—"
            matrix_rows.append(row)
        st.dataframe(pd.DataFrame(matrix_rows), width='stretch', hide_index=True)
    else:
        st.caption("No required skills detected for this role to compare against.")
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Create Job Requisition
# --------------------------------------------------------------------------
def page_requisition():
    styling.topbar(st, "Create Job Requisition")

    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    sample_jds = list_sample_jds()

    prefill_choice = st.selectbox(
        "Start from a sample job description (optional)",
        ["— Blank —"] + list(sample_jds.keys()),
        key="prefill_choice",
    )

    # --- FIX: whenever the sample dropdown changes, overwrite the draft and
    # bump a version counter so the textarea below is rebuilt with new text ---
    if st.session_state.get("last_prefill_applied") != prefill_choice:
        st.session_state["jd_draft_text"] = (
            "" if prefill_choice == "— Blank —" else sample_jds.get(prefill_choice, "")
        )
        st.session_state["last_prefill_applied"] = prefill_choice
        st.session_state["jd_widget_version"] = st.session_state.get("jd_widget_version", 0) + 1

    jd_version = st.session_state.get("jd_widget_version", 0)

    # JD text lives outside the form so the bias scanner below can react to
    # every edit, instead of only after Create Requisition is submitted.
    jd_text = st.text_area(
        "Job Description",
        value=st.session_state["jd_draft_text"],
        height=220,
        placeholder="Paste the full job description here…",
        key=f"jd_draft_widget_{jd_version}",
    )
    st.session_state["jd_draft_text"] = jd_text

    st.markdown("##### 🛡️ Bias & Inclusive-Language Scan")
    scan = bias_checker.check_jd_bias(jd_text)
    risk_colors = {"Low": "#15803d", "Moderate": "#b45309", "High": "#b91c1c"}
    risk_bg = {"Low": "#dcfce7", "Moderate": "#fef3c7", "High": "#fee2e2"}
    risk = scan["risk_level"]
    st.markdown(
        f'<span style="background:{risk_bg[risk]};color:{risk_colors[risk]};padding:4px 12px;'
        f'border-radius:20px;font-size:0.8rem;font-weight:700;">Bias risk: {risk} '
        f'({scan["term_count"]} flagged term{"s" if scan["term_count"] != 1 else ""} '
        f'across {scan["category_count"]} categor{"ies" if scan["category_count"] != 1 else "y"})</span>',
        unsafe_allow_html=True,
    )
    if scan["hits"]:
        with st.expander("View flagged wording & suggested rewrites", expanded=(risk != "Low")):
            for h in scan["hits"]:
                st.markdown(
                    f'- **"{h["term"]}"** _(× {h["count"]})_ — {h["category"]} '
                    f'→ suggested: *{h["suggestion"]}*'
                )
    else:
        st.caption("No flagged gendered, age-coded, ableist, or exclusionary wording detected.")

    with st.form("req_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        default_title = prefill_choice if prefill_choice != "— Blank —" else ""
        title = col1.text_input("Job Title", value=default_title)
        department = col2.selectbox("Department", DEPARTMENTS)

        c1, c2 = st.columns(2)
        min_years = c1.number_input("Minimum years of experience required", min_value=0, max_value=30, value=0)
        required_education = c2.selectbox("Minimum education required", EDU_OPTIONS)

        submitted = st.form_submit_button("Create Requisition", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not title or not jd_text.strip():
            st.error("Job title and job description are required.")
        else:
            parsed = parse_job_description(
                jd_text, min_years_override=min_years, extra_skills=st.session_state["custom_skills"]
            )
            rid = new_req_id()
            req_data = {
                "id": rid,
                "title": title,
                "department": department,
                "jd_text": jd_text,
                "required_skills": parsed["required_skills"],
                "min_years": float(min_years) if min_years else parsed["min_years"],
                "required_education": required_education if required_education != "Not detected" else parsed["required_education"],
                "weights": dict(st.session_state["default_weights"]),
                "created_at": dt.datetime.now(),
            }
            st.session_state["requisitions"][rid] = req_data
            db.save_requisition(req_data)
            st.session_state["active_req_id"] = rid
            st.success(f"Requisition **{title}** created with {len(parsed['required_skills'])} auto-detected required skills. "
                       f"Head to **Resume Upload** to start screening.")
            st.markdown("**Auto-detected required skills:**")
            st.markdown("".join(f'<span class="tiq-skill-chip chip-match">{s}</span>' for s in parsed["required_skills"]) or "_none detected — add skills manually via the JD text_",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def account_settings_card():
    user = current_user()
    if not user:
        return
    st.markdown('<div class="tiq-card"><h4>Account</h4>'
                f'<p style="color:#6b7290;">Signed in as <b>{html.escape(user["name"])}</b> '
                f'({html.escape(user["email"])}).</p>', unsafe_allow_html=True)
    with st.form("change_password_form", clear_on_submit=True):
        current_pw = st.text_input("Current password", type="password")
        pc1, pc2 = st.columns(2)
        new_pw = pc1.text_input("New password", type="password")
        confirm_pw = pc2.text_input("Confirm new password", type="password")
        change = st.form_submit_button("🔑 Change password", type="primary")
    if change:
        ok, message = auth.change_password(user["id"], current_pw, new_pw, confirm_pw)
        (st.success if ok else st.error)(message)
    if st.button("Sign out", key="settings_sign_out"):
        sign_out()
    st.markdown("</div>", unsafe_allow_html=True)


def page_settings():
    styling.topbar(st, "Settings")

    st.markdown('<div class="tiq-card"><h4>AI Scoring Pipeline</h4>'
                '<p style="color:#6b7290;">Every candidate score is a transparent, weighted blend of four evidence '
                'factors — never a black-box number. Adjust the default split below; each requisition can also '
                'override it individually from its <b>Manage this requisition</b> panel on the Resume Upload page.</p>',
                unsafe_allow_html=True)
    dw = st.session_state["default_weights"]
    wc1, wc2, wc3, wc4 = st.columns(4)
    d_sem = wc1.slider("Semantic Relevance %", 0, 100, round(dw["semantic_relevance"] * 100), key="dw_sem")
    d_skl = wc2.slider("Skill Alignment %", 0, 100, round(dw["skill_alignment"] * 100), key="dw_skl")
    d_exp = wc3.slider("Experience Evidence %", 0, 100, round(dw["experience_evidence"] * 100), key="dw_exp")
    d_edu = wc4.slider("Education Evidence %", 0, 100, round(dw["education_evidence"] * 100), key="dw_edu")
    d_total = d_sem + d_skl + d_exp + d_edu
    normalized = _normalized_weights(d_sem, d_skl, d_exp, d_edu)
    previous_weights = st.session_state.get("default_weights", dict(WEIGHTS))
    st.session_state["default_weights"] = normalized
    st.session_state["app_prefs"]["default_weights"] = normalized
    if normalized != previous_weights:
        if db.update_hr_settings(default_weights=normalized):
            st.session_state["cloud_hr_settings"]["default_weights"] = normalized
        else:
            st.warning("Scoring weights changed for this session, but Supabase could not save them.")
    st.caption(f"Sliders sum to {d_total}% and are normalized to 100% automatically. "
               "This becomes the default for newly created requisitions — existing ones keep their own weights "
               "until edited.")
    weights_display = {"Semantic Relevance": d_sem, "Skill Alignment": d_skl, "Experience Evidence": d_exp, "Education Evidence": d_edu}
    fig = go.Figure(go.Bar(x=list(weights_display.values()), y=list(weights_display.keys()), orientation="h", marker_color="#131b3d"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", xaxis_title="Weight (%)")
    st.plotly_chart(fig, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tiq-card"><h4>Custom Skills Taxonomy</h4>'
                '<p style="color:#6b7290;">The built-in taxonomy covers ~250 common skills. Add proprietary or '
                'newer tools (e.g. LangGraph, vLLM, SvelteKit, Mojo) here so resumes and job descriptions mentioning '
                'them get recognized — no need to wait for a code change.</p>', unsafe_allow_html=True)
    new_custom = st.text_input("Add a custom skill", key="settings_add_custom_skill")
    if st.button("➕ Add skill") and new_custom.strip():
        if new_custom.strip() not in st.session_state["custom_skills"]:
            st.session_state["custom_skills"].append(new_custom.strip())
            st.session_state["app_prefs"]["custom_skills"] = list(st.session_state["custom_skills"])
            if db.update_hr_settings(custom_skills=st.session_state["custom_skills"]):
                st.session_state["cloud_hr_settings"]["custom_skills"] = list(st.session_state["custom_skills"])
            else:
                st.warning("Custom skill added for this session, but Supabase could not save it.")
        st.rerun()
    if st.session_state["custom_skills"]:
        st.caption("Current custom skills (click ✖ to remove):")
        cs_cols = st.columns(4)
        for i, skill in enumerate(list(st.session_state["custom_skills"])):
            with cs_cols[i % 4]:
                if st.button(f"✖ {skill}", key=f"rm_custom_skill_{i}"):
                    st.session_state["custom_skills"].remove(skill)
                    st.session_state["app_prefs"]["custom_skills"] = list(st.session_state["custom_skills"])
                    if db.update_hr_settings(custom_skills=st.session_state["custom_skills"]):
                        st.session_state["cloud_hr_settings"]["custom_skills"] = list(st.session_state["custom_skills"])
                    else:
                        st.warning("Custom skill removed for this session, but Supabase could not save it.")
                    st.rerun()
    else:
        st.caption("No custom skills added yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tiq-card"><h4>Semantic Matching Engine</h4>'
                '<p style="color:#6b7290;">The Semantic Relevance factor above (50% of the score) can run on '
                'either TF-IDF (fast, zero extra install) or sentence-transformer embeddings (meaning-based, '
                'catches phrasing TF-IDF misses — e.g. "led a squad" vs. "managed a team" — at the cost of a '
                'one-time ~80MB model download on first use).</p>', unsafe_allow_html=True)
    have_embeddings = embeddings_available()
    st.caption(
        "✅ sentence-transformers is installed — embeddings are available."
        if have_embeddings else
        "⚠️ sentence-transformers isn't installed — install requirements-embeddings.txt to enable it. "
        "TF-IDF will be used regardless of the setting below until then."
    )
    engine_labels = {
        "auto": "Auto (embeddings if installed, otherwise TF-IDF) — recommended",
        "tfidf": "TF-IDF only (fastest, works fully offline)",
        "embeddings": "Sentence-transformer embeddings only",
    }
    current = st.session_state["semantic_backend"]
    choice = st.radio(
        "Engine", list(engine_labels.keys()), index=list(engine_labels.keys()).index(current),
        format_func=lambda k: engine_labels[k], label_visibility="collapsed",
    )
    if choice != st.session_state.get("semantic_backend", "auto"):
        st.session_state["semantic_backend"] = choice
        st.session_state["app_prefs"]["semantic_backend"] = choice
        if db.update_hr_settings(semantic_backend=choice):
            st.session_state["cloud_hr_settings"]["semantic_backend"] = choice
            st.toast("Semantic engine preference saved to Supabase.")
        else:
            st.warning("Semantic engine changed for this session, but Supabase could not save it.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="tiq-card"><h4>Fairness & Responsible Screening</h4>'
        '<p style="color:#6b7290;">TalentIQ\'s scoring logic never uses candidate name, gender-coded language, '
        'photo, or college prestige as a signal — only skills, experience, education level, and semantic fit to '
        'the role. TalentIQ provides decision-support signals and does not replace human hiring judgment; every '
        'AI-generated ranking should be reviewed by an HR recruiter before a hiring decision is made.</p>',
        unsafe_allow_html=True,
    )
    all_results = [r for lst in st.session_state["results"].values() for r in lst]
    audited = [r for r in all_results if r.fairness_audit]
    if audited:
        avg_delta = sum(abs(r.fairness_audit["delta"]) for r in audited) / len(audited)
        st.markdown(
            f'<p style="color:#6b7290;"><b>Blind-scoring audit:</b> across {len(audited)} screened candidate'
            f'{"s" if len(audited) != 1 else ""} this session, redacting name/email/phone before re-scoring '
            f'moved the semantic-match score by an average of <b>{avg_delta:.2f} points</b> — evidence that '
            f'identity fields are not driving the ranking, not just a policy statement.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Run a screening to see the blind-scoring audit (redacted-identity re-score vs. original) here.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tiq-card"><h4>Organization & Email Identity</h4>'
                '<p style="color:#6b7290;">Used to personalize candidate emails sent from Candidate Intelligence. '
                'These details are <b>saved</b> — they stay the same after a refresh or restart until you change '
                'them here.</p>',
                unsafe_allow_html=True)
    org = st.session_state["org_settings"]
    with st.form("org_identity_form"):
        oc1, oc2, oc3 = st.columns(3)
        new_company = oc1.text_input("Company name", value=org["company_name"])
        new_sender = oc2.text_input("Your name (email sign-off)", value=org["hr_sender_name"])
        new_title = oc3.text_input("Your title", value=org["hr_sender_title"])
        save_org = st.form_submit_button("💾 Save organization details", type="primary")
    if save_org:
        if not (new_company.strip() and new_sender.strip() and new_title.strip()):
            st.error("Company name, your name, and your title can't be left blank.")
        else:
            updated = {
                "company_name": new_company.strip(),
                "hr_sender_name": new_sender.strip(),
                "hr_sender_title": new_title.strip(),
            }
            org.update(updated)
            ok = db.update_hr_settings(**updated)
            if ok:
                st.session_state["cloud_hr_settings"].update(updated)
                st.success("Organization details saved to Supabase.")
            else:
                st.warning("Saved for this session, but Supabase could not save the settings.")
    st.markdown("</div>", unsafe_allow_html=True)

    can_save_smtp = settings_store.smtp_saving_enabled()
    smtp_intro = (
        "Connect your own mailbox so shortlist / interview / rejection emails can be sent directly from a "
        "candidate profile. Your email address and app password are <b>saved with your TalentIQ account</b> "
        "(on the machine running this app, outside the project folder, and only ever used for your own account) "
        "until you change or forget them. For Gmail/Outlook, use an <b>app "
        "password</b> rather than your normal login password."
        if can_save_smtp else
        "Connect your own mailbox so shortlist / interview / rejection emails can be sent directly from a "
        "candidate profile. This deployment does not store credentials — they last for this browser session "
        "only. For Gmail/Outlook, use an <b>app password</b> rather than your normal login password."
    )
    st.markdown('<div class="tiq-card"><h4>📧 Email Automation (SMTP)</h4>'
                f'<p style="color:#6b7290;">{smtp_intro}</p>', unsafe_allow_html=True)

    ss = st.session_state
    smtp = ss["smtp_config"]
    flash = ss.pop("_smtp_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

    preset_names = list(email_automation.SMTP_PRESETS.keys())
    current_provider = smtp.get("provider", "Gmail")
    preset = st.selectbox(
        "Provider", preset_names,
        index=preset_names.index(current_provider) if current_provider in preset_names else 0,
        key="smtp_provider_select",
    )
    preset_vals = email_automation.SMTP_PRESETS[preset]
    # Show the saved host/port/TLS only for the provider they were saved under;
    # picking a different provider loads that provider's standard settings.
    use_saved = preset == current_provider and bool(smtp.get("host"))
    host_default = smtp["host"] if use_saved else preset_vals["host"]
    port_default = int(smtp["port"]) if use_saved else preset_vals["port"]
    tls_default = bool(smtp["tls"]) if use_saved else preset_vals["tls"]

    configured = bool(smtp.get("host") and smtp.get("sender_email") and smtp.get("password"))
    saved_on_disk = settings_store.has_saved_smtp(current_user_id())

    with st.form("smtp_form"):
        sc1, sc2, sc3 = st.columns([2, 1, 1])
        host_in = sc1.text_input("SMTP host", value=host_default, key=f"smtp_host_{preset}")
        port_in = sc2.number_input("Port", min_value=1, max_value=65535, value=port_default, key=f"smtp_port_{preset}")
        tls_in = sc3.checkbox("Use STARTTLS", value=tls_default, key=f"smtp_tls_{preset}")
        sc4, sc5 = st.columns(2)
        email_in = sc4.text_input("Your email address", value=smtp.get("sender_email", ""), key="smtp_email_input")
        pw_in = sc5.text_input(
            "App password", value="", type="password", key="smtp_password_input",
            placeholder="Saved — leave blank to keep it" if smtp.get("password") else "Paste your app password",
        )
        remember = False
        if can_save_smtp:
            remember = st.checkbox(
                "Remember for my account (until I change or forget it)",
                value=(saved_on_disk or not configured),
                help="Untick to use these details for this browser session only.",
            )
        b1, b2, _ = st.columns([1.3, 1.3, 3])
        save_smtp = b1.form_submit_button("💾 Save email settings", type="primary")
        forget_smtp = b2.form_submit_button("🗑 Forget email & password")

    def _reset_smtp_widgets():
        for key in list(ss.keys()):
            if key in ("smtp_email_input", "smtp_password_input") or key.startswith(("smtp_host_", "smtp_port_", "smtp_tls_")):
                ss.pop(key, None)

    if save_smtp:
        new_email = email_in.strip()
        new_host = host_in.strip()
        new_pw = pw_in.strip() or smtp.get("password", "")
        email_changed = new_email != smtp.get("sender_email", "")
        if not (new_host and new_email):
            st.error("SMTP host and your email address are required.")
        elif "@" not in new_email or "." not in new_email.split("@")[-1]:
            st.error("That doesn't look like a valid email address.")
        elif not new_pw or (email_changed and not pw_in.strip()):
            st.error("Enter the app password for this email address.")
        else:
            new_cfg = {"provider": preset, "host": new_host, "port": int(port_in),
                       "tls": bool(tls_in), "sender_email": new_email, "password": new_pw}
            smtp.update(new_cfg)
            if remember and can_save_smtp:
                if settings_store.save_smtp_config(new_cfg, current_user_id()):
                    ss["_smtp_flash"] = ("success", "Email settings saved to your account. Change them here any time.")
                else:
                    ss["_smtp_flash"] = ("warning", "Applied for this session, but the credentials file couldn't be written, "
                                                    "so they won't survive a restart.")
            else:
                settings_store.clear_smtp_config(current_user_id())
                ss["_smtp_flash"] = ("success", "Email settings applied for this browser session only (nothing saved to disk).")
            _reset_smtp_widgets()
            st.rerun()

    if forget_smtp:
        settings_store.clear_smtp_config(current_user_id())
        smtp.clear()
        smtp.update(settings_store.DEFAULT_SMTP)
        ss["_smtp_flash"] = ("success", "Email address and app password removed" +
                             (" (from this session and from disk)." if can_save_smtp else " from this session."))
        _reset_smtp_widgets()
        st.rerun()

    if configured:
        st.caption("✅ Email automation is configured and saved to your account." if saved_on_disk
                   else "✅ Email automation is configured for this session only.")
    else:
        st.caption("⚠️ Not fully configured yet — candidate emails can still be downloaded as .eml files without this.")
    st.markdown("</div>", unsafe_allow_html=True)

    account_settings_card()

    st.markdown('<div class="tiq-card"><h4>Cloud Data & Persistence</h4>', unsafe_allow_html=True)
    if db.is_configured():
        st.caption("Requisitions, resumes, screened candidates, interview guides, email logs, and HR settings are synchronized with your Supabase database. SMTP credentials remain in the separate private credential store.")
    else:
        st.caption("Supabase is not configured — running in session-only mode.")
    prefs = st.session_state["app_prefs"]
    seed_demo = st.checkbox(
        "Start each new session with the demo requisition (Machine Learning Engineer)",
        value=prefs.get("seed_demo_requisition", True),
        help="Turn this off to open the app with zero active jobs.",
    )
    if seed_demo != prefs.get("seed_demo_requisition", True):
        prefs["seed_demo_requisition"] = seed_demo
        if db.update_hr_settings(seed_demo_requisition=seed_demo):
            st.session_state["cloud_hr_settings"]["seed_demo_requisition"] = seed_demo
            st.toast("Preference saved to Supabase — applies from the next new session.")
        else:
            st.warning("Preference changed for this session, but Supabase could not save it.")
    if st.button("🗑 Reset all data (Clear Supabase & Session)"):
        db.reset_all_data()
        for key in ["requisitions", "results", "candidate_actions", "active_req_id", "req_counter",
                    "last_run_at", "jd_draft_text", "interview_guides", "email_log",
                    "upload_req_selector", "cand_req_selector",
                    "prefill_choice", "last_prefill_applied", "jd_widget_version",
                    "default_weights", "custom_skills", "confirm_delete_req",
                    "dash_remove_req", "dash_remove_confirm", "_demo_seed_checked", "_synced_from_db"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="tiq-card"><h4>About</h4>'
        '<p style="color:#6b7290;">TalentIQ — AI Resume Screener & Job Matcher. Built for the '
        '"Smarter shortlisting through AI" hackathon brief: parses resumes, understands skills semantically, '
        'and ranks candidates against a job description with a transparent, explainable score.</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    # Nothing in the workspace renders until someone has signed in.
    if not current_user():
        auth_ui.render_auth_page()
        return
    init_state()
    page = sidebar_nav()

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "📁 Resume Upload":
        page_upload()
    elif page == "🧠 Candidate Intelligence":
        page_candidates()
    elif page == "➕ Create Job Requisition":
        page_requisition()
    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()