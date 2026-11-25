import json
import tempfile
from pathlib import Path

import streamlit as st
from ai_resume_analyzer import ROLES, analyze, read_resume, build_report

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.title("AI Resume Analyzer & Skill Gap Detector")
st.caption("Python-powered career guidance • Rule-based analysis")
st.info("Use a sample or a redacted resume. Uploaded files are processed on the server; reports may contain personal information.")
sample = st.checkbox("Use sample resume", value=True)
upload = None if sample else st.file_uploader("Upload resume", type=["txt", "pdf", "docx"])
mode = st.radio("Analysis mode", ["Best-fit roles", "Target role", "Job description"], horizontal=True)
role = st.selectbox("Target role", list(ROLES)) if mode == "Target role" else None
jd = st.text_area("Paste job description") if mode == "Job description" else None
if st.button("Analyze resume", type="primary"):
    st.session_state.pop("result", None)
    try:
        if mode == "Job description" and not (jd or "").strip():
            raise ValueError("Enter a job description.")
        if sample:
            path = Path(__file__).parent / "sample_resume.txt"
            content = read_resume(str(path))
            name = path.name
        elif upload is not None:
            name = Path(upload.name).name
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / name
                path.write_bytes(upload.getvalue())
                content = read_resume(str(path))
        else:
            raise ValueError("Upload a resume first.")
        if len(content.split()) < 30:
            raise ValueError("Resume text is too short or could not be extracted.")
        st.session_state.result = analyze(content, source=name, role=role, jd_text=jd)
    except Exception as exc:
        st.error(str(exc))

res = st.session_state.get("result")
if res:
    st.subheader("Analysis results")
    a, b, c = st.columns(3)
    a.metric("ATS score", f"{res['ats']['score']}/100")
    b.metric("Skills detected", len(res["skills"]))
    c.metric("Top role", res["ranking"][0]["role"])
    st.progress(res["ats"]["score"] / 100)
    st.caption("Heuristic score, not an actual employer ATS result.")
    tabs = st.tabs(["Career matches", "Skills", "ATS breakdown", "Skill gap"])
    with tabs[0]:
        for item in res["ranking"][:5]:
            st.write(f"{item['role']} — {item['score']}%")
            st.progress(item["score"] / 100)
    with tabs[1]:
        for category, skills in res["skills_by_category"].items():
            st.markdown(f"**{category}**")
            st.write(", ".join(skills))
    with tabs[2]:
        for label, (points, maximum) in res["ats"]["breakdown"].items():
            st.write(f"{label}: {points}/{maximum}")
        for tip in res["ats"]["tips"]:
            st.write("• " + tip)
    with tabs[3]:
        target = res["target"]
        if target:
            st.metric(target["name"], f"{target['score']}%")
            st.write(target["verdict"])
            st.write("Matched skills:", ", ".join(target["matched"]) or "None")
            st.write("Missing skills:", ", ".join(target["missing"]) or "None")
            if target["roadmap"]:
                st.dataframe(target["roadmap"], hide_index=True)
        else:
            st.write("No target skill gap available for this analysis.")
        if res["jd_similarity"] is not None:
            st.write(f"Job description text similarity: {res['jd_similarity']}%")
    a, b = st.columns(2)
    a.download_button("Download TXT report", build_report(res), "resume_report.txt", "text/plain")
    b.download_button("Download JSON report", json.dumps(res, indent=2), "resume_report.json", "application/json")
    if st.button("Clear results"):
        st.session_state.pop("result", None)
        st.rerun()
