import streamlit as st  # type: ignore
import pandas as pd
import os
import csv
import re
import urllib.parse

from src.parser import parse_resume
from src.match_score import get_fit_score
from src.bias_checker import detect_bias
from src.explain import get_shap_values

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'sans-serif' !important;
    }
    label, input, textarea, .stTextInput > div > div, .stTextArea > div > textarea, .stMarkdown, .stButton > button {
        font-family: 'sans-serif' !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='font-family:sans-serif; font-weight:bold;'>📄 AI-Powered Resume Screener</h2>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style='font-family:sans-serif; font-weight:bold;'>📌 Job Information</h3>
""", unsafe_allow_html=True)

job_title = st.text_input("🧠 Job Title", placeholder=" ", label_visibility="visible")
job_description = st.text_area("📋 Job Description", placeholder="Paste the full JD here...", label_visibility="visible")

st.markdown("""
    <h3 style='font-family:sans-serif;'>⚙️ Settings</h3>
""", unsafe_allow_html=True)

with st.expander("⚙️ App Settings"):
    score_threshold = st.slider("🎯 Fit Score Threshold for Zipping Resumes", min_value=0, max_value=100, value=60, step=5)
    file_types = st.multiselect("📂 Allowed File Types", ["pdf", "docx"], default=["pdf", "docx"])
    archive_logs = st.checkbox("🗄️ Auto-archive daily logs", value=True)

    reset_triggered = st.button("🔄 Reset Settings")
    if reset_triggered:
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()


st.markdown("""
    <h4 style='font-family:sans-serif;'>📁 Upload Resumes (PDF/DOCX)</h4>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag and drop files here (PDF or DOCX only)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    label_visibility="visible"
)

st.markdown("✅ <span style='font-family:sans-serif;'>Supported formats: PDF, DOCX</span>", unsafe_allow_html=True)

def extract_email(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return emails[0] if emails else "Not found"

results = []

import datetime
if archive_logs:
    try:
        if os.path.exists("data/logs.csv"):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.rename("data/logs.csv", f"data/logs_{timestamp}.csv")
            st.toast("🗃️ Logs archived on app start", icon="🧾")
    except Exception as e:
        st.warning(f"Log archiver error: {e}")
else:
    try:
        if os.path.exists("data/logs.csv"):
            os.remove("data/logs.csv")
            st.toast("🧹 Logs cleared on app start", icon="🧾")
    except Exception as e:
        st.warning(f"Log cleaner error: {e}")

if uploaded_files and job_description:
    os.makedirs("data", exist_ok=True)

    import pathlib
    from zipfile import ZipFile

    progress = st.progress(0, text="Processing resumes...")

    zip_path = "data/top_resumes.zip"
    with ZipFile(zip_path, 'w') as zipf:
        for i, file in enumerate(uploaded_files):
            filename = pathlib.Path(file.name)
            ext = filename.suffix.lower()

            if ext.strip(".") not in file_types:
                st.warning(f"❌ Unsupported file type: {ext}. Please upload PDF or DOCX.")
                continue

            file_path = f"data/{filename.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            resume_text = parse_resume(file_path)
            score = get_fit_score(resume_text, job_description)
            bias = detect_bias(resume_text)
            X, shap_values = get_shap_values(resume_text, job_description)
            matched_keywords = [col for col in X.columns if X[col][0] == 1]
            email_found = extract_email(resume_text)

            result = {
                "ResumeFile": filename.name,
                "Email": email_found,
                "FitScore": int(round(score * 100)),
                "MatchedKeywords": ", ".join(matched_keywords),
                "GenderedWords": ", ".join(bias["gendered_words"]) or "None",
                "AgeIndicators": ", ".join(bias["age_indicators"]) or "None"
            }

            results.append(result)

            log_path = "data/logs.csv"
            file_exists = os.path.isfile(log_path)
            with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)

            if result["FitScore"] >= score_threshold:
                zipf.write(file_path, arcname=filename.name)

            progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i + 1}/{len(uploaded_files)} resumes")

    df = pd.DataFrame(results).sort_values(by="FitScore", ascending=False).reset_index(drop=True)
    st.markdown("""
        <h3 style='font-family:sans-serif;'>🏆 Top Resume Matches</h3>
    """, unsafe_allow_html=True)
    st.dataframe(df.style.hide(axis="index"))

    st.markdown("""
        <h3 style='font-family:sans-serif;'>🔢 Select Top N Resumes</h3>
    """, unsafe_allow_html=True)
    top_n = st.slider("Choose how many top resumes to view:", min_value=1, max_value=len(df), value=5, label_visibility="visible")
    top_df = df.head(top_n).reset_index(drop=True)

    st.markdown("""
        <h3 style='font-family:sans-serif;'>📧 Emails of Top Resumes</h3>
    """, unsafe_allow_html=True)
    st.dataframe(top_df[["ResumeFile", "Email"]].style.hide(axis="index"))

    st.markdown("""
        <h3 style='font-family:sans-serif;'>📤 Mail Top Resumes via Gmail</h3>
    """, unsafe_allow_html=True)
    valid_emails = [email for email in top_df["Email"] if email != "Not found"]
    subject_text = f"Regarding Your Application for: {job_title.strip() or 'a recent opportunity'}"
    encoded_subject = urllib.parse.quote(subject_text)
    mailto_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={','.join(valid_emails)}&su={encoded_subject}"

    st.markdown(f"""
        <a href="{mailto_link}" target="_blank">
            <button style="
                background-color: #f63366;
                color: white;
                padding: 0.6em 1.2em;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s;
            ">📨 Email Top {top_n} Resumes via Gmail</button>
        </a>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    csv_all = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download All Resume Scores (CSV)", csv_all, "all_resume_scores.csv", "text/csv")
