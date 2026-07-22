import streamlit as st
from extract_text import extract_resume_text
from analyzer import get_resume_review, get_jd_match

if "resume_text" not in st.session_state:
    st.session_state.resume_text=None
if "resume_json" not in st.session_state:
    st.session_state.resume_json=None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = None

if "jd_match_result" not in st.session_state:
    st.session_state.jd_match_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("Resume Analyzer")
uploaded_file=st.sidebar.file_uploader("Upload your resume(PDF)",type=["pdf"])
if uploaded_file is not None:
    st.session_state.resume_text = extract_resume_text(uploaded_file)
    st.sidebar.success("Resume uploaded and text extracted!")
jd_input=st.sidebar.text_area("Paste Job Description here(optional)",height=200)   
if jd_input:
    st.session_state.jd_text=jd_input
else:
    st.session_state.jd_text = None
if st.session_state.jd_text:
    tab1 , tab2 = st.tabs(["Resume Score","Resume score + Job Match"])
else:
    tab1 = st.tabs(["Resume Score"])[0]

with tab1:
    if st.session_state.resume_text is None:
        st.info("Please upload a resume to get the score")
    else:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                review=get_resume_review(st.session_state.resume_text)
            
            if review is None:
                st.error("Something went wrong!")
            else:
                st.session_state.resume_json=review
                st.success("Resume analysis completed!")
                   

            if st.session_state.resume_json:
                review=st.session_state.resume_json
                st.metric("Resume Score",f"{review['score']}/100")

                st.subheader("Strengths")
                if review["strengths"]:
                 for item in review["strengths"]:
                    st.write(f"-{item}")

                st.subheader("Weaknesses")
                if review["weaknesses"]:
                    for item in review["weaknesses"]:
                        st.write(f"-{item}")

                st.subheader("Suggestions")
                if review["suggestions"]:
                    for item in review["suggestions"]:
                        st.write(f"-{item}")



if st.session_state.jd_text:
    with tab2:
        if st.session_state.resume_json is None:
            st.info("Please analyze your resume first")
        else:
            if st.button("Analyze JD Match"):
                 with st.spinner("Comparing your resume to the job description..."):
                    match = get_jd_match(
                        st.session_state.resume_text,
                        st.session_state.jd_text,
                        st.session_state.resume_json
                    )
                 if match is None:
                    st.error("Something went wrong parsing the AI's response. Try again.")
                 else:
                    st.session_state.jd_match_result = match

            if st.session_state.jd_match_result:
                match = st.session_state.jd_match_result
                review = st.session_state.resume_json

                col1,col2=st.columns(2)
                with col1:
                    st.metric("Resume Score",f"{review['score']}/100")

                with col2:
                    st.metric("JD Match Score",f"{match['match_score']}/100")
                st.subheader("Matched Keywords")
                if match["matched_keywords"]:
                   for item in match["matched_keywords"]:
                      st.write(f"- {item}")
                else:
                     st.write("No matched keywords found.")

                st.subheader("Missing Keywords")
                if match["missing_keywords"]:
                   for item in match["missing_keywords"]:
                    st.write(f"- {item}")
                else:
                   st.write("No missing keywords — great match!")

                st.subheader("JD-Specific Suggestions")
                if match["jd_suggestions"]:
                   for item in match["jd_suggestions"]:
                     st.write(f"- {item}")
                else:
                     st.write("No specific suggestions.")



