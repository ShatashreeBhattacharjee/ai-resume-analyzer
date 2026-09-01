import streamlit as st
from extract_text import extract_text
from analyzer import get_resume_review, get_jd_match, format_resume_summary, chat, format_jd_summary, CHAT_SYSTEM_PROMPT, get_interview_questions, get_answer_feedback
from rag import build_tagged_chunks, embed_chunks, build_faiss_index, retrieve_context, retrieve_context_per_candidate


def build_chat_messages():
    latest_question = st.session_state.chat_history[-1]["content"]
    retrieved_chunks = retrieve_context_per_candidate(
        latest_question,
        st.session_state.tagged_chunks,
        top_k_per_candidate=2
    )
    context_text = "\n\n".join(
        f"[{c['candidate']} - {c['section']}]\n{c['text']}" for c in retrieved_chunks
    )

    jd_text = ""
    if st.session_state.jd_text:
        jd_text = f"\n\nJob description being considered:\n{st.session_state.jd_text}"

    system_message = {
        "role": "system",
        "content": CHAT_SYSTEM_PROMPT + f"\n\nRelevant resume context for this question:\n{context_text}{jd_text}"
    }
    clean_history = [system_message] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
    ]
    return clean_history


if "jd_text" not in st.session_state:
    st.session_state.jd_text = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_resumes" not in st.session_state:
    st.session_state.uploaded_resumes = []

if "resumes" not in st.session_state:
    st.session_state.resumes = []

st.sidebar.title("Resume Analyzer")

uploaded_files = st.sidebar.file_uploader(
    "Upload resume(s) (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded_files:
    existing_filenames = {r["filename"] for r in st.session_state.uploaded_resumes}
    new_file_added = False
    for file in uploaded_files:
        if file.name not in existing_filenames:
            new_text = extract_text(file)
            st.session_state.uploaded_resumes.append({
                "filename": file.name,
                "resume_text": new_text,
            })
            new_file_added = True
        if new_file_added:
            st.session_state.resumes = []
            st.session_state.chat_history = []

if st.session_state.uploaded_resumes:
    st.sidebar.write(f"{len(st.session_state.uploaded_resumes)} resume(s) uploaded:")
    for resume in list(st.session_state.uploaded_resumes):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.write(resume["filename"])
        with col2:
            if st.button("✕", key=f"remove_{resume['filename']}"):
                removed_filename = resume["filename"]
                st.session_state.uploaded_resumes.remove(resume)
                st.session_state.resumes = [
                    r for r in st.session_state.resumes if r["filename"] != removed_filename
                ]

                if "tagged_chunks" in st.session_state:
                    st.session_state.tagged_chunks = [
                        c for c in st.session_state.tagged_chunks if c["candidate"] != removed_filename
                    ]
                    if st.session_state.tagged_chunks:
                        embeddings = embed_chunks(st.session_state.tagged_chunks)
                        st.session_state.faiss_index = build_faiss_index(embeddings)
                    else:
                        del st.session_state.tagged_chunks
                        if "faiss_index" in st.session_state:
                            del st.session_state.faiss_index

                st.rerun()


jd_input = st.sidebar.text_area("Paste Job Description here(optional)", height=200)
if jd_input:
    st.session_state.jd_text = jd_input
else:
    st.session_state.jd_text = None

if not st.session_state.uploaded_resumes:
    st.markdown(
        """
        <div style="
            background-color: #FFF4E0;
            padding: 40px 30px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
        ">
            <h1 style="margin: 0; font-size: 2.2rem; color: #1A1A1A;">
                Is your resume good enough?
            </h1>
            <p style="margin-top: 10px; font-size: 1.1rem; color: #4A4A4A;">
                Get an instant score, clear feedback, and answers to your questions — all in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    features = [
        ("⚡", "Instant score", "Get a numeric score and breakdown in seconds."),
        ("💡", "Clear suggestions", "See specific strengths, weaknesses, and fixes."),
        ("💬", "Ask questions", "Chat with the AI about your resume anytime."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="font-weight: 600; margin-top: 8px; font-size: 1.05rem;">{title}</div>
                    <div style="color: #666; font-size: 0.9rem; margin-top: 4px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div style="
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            font-size: 1rem;
            color: #4A4A4A;
        ">
            👈 Upload your resume in the sidebar to get started
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    if st.session_state.jd_text:
        tab1, tab2, tab3 = st.tabs(["Resume Score", "Resume score + Job Match", "Interview Prep"])
    else:
        tab1, tab3 = st.tabs(["Resume Score", "Interview Prep"])

    with tab1:
        if st.button("Analyze"):
            st.session_state.resumes = []
            failed_resumes = []
            with st.spinner("Analyzing..."):
                for resume in st.session_state.uploaded_resumes:
                    review = get_resume_review(resume["resume_text"])

                    if review.get("error"):
                        failed_resumes.append(resume["filename"])
                        continue
                    st.session_state.resumes.append({
                        "filename": resume["filename"],
                        "resume_text": resume["resume_text"],
                        "resume_json": review,
                        "jd_match_result": None,
                    })
            if failed_resumes:
                st.warning(f"Could not analyze: {', '.join(failed_resumes)}. Please try again.")

            with st.spinner("Preparing chat context..."):
                all_tagged_chunks = []
                for r in st.session_state.resumes:
                    chunks = build_tagged_chunks(r["resume_text"], r["filename"])
                    all_tagged_chunks.extend(chunks)

                if all_tagged_chunks:
                    embeddings = embed_chunks(all_tagged_chunks)
                    st.session_state.faiss_index = build_faiss_index(embeddings)
                    st.session_state.tagged_chunks = all_tagged_chunks
            st.success("Analysis completed!")

        if st.session_state.resumes:
            for r in st.session_state.resumes:
                st.subheader(r["filename"])
                review = r["resume_json"]
                st.metric("Resume Score", f"{review['score']}/100")

                st.write("**Strengths**")
                for item in review["strengths"]:
                    st.write(f"- {item}")

                st.write("**Weaknesses**")
                for item in review["weaknesses"]:
                    st.write(f"- {item}")

                st.write("**Suggestions**")
                for item in review["suggestions"]:
                    st.write(f"- {item}")

                st.divider()

    with tab3:
        if not st.session_state.resumes:
            st.info("Please analyze your resumes first")
        else:
            if st.session_state.jd_text:
                target_role_text = st.session_state.jd_text
            else:
                target_role_text = st.text_input("Target role (used since no job description is provided)")

            if st.button("Generate Questions"):
                if not target_role_text:
                    st.warning("Please paste a job description or enter a target role first.")
                else:
                    failed_questions = []
                    with st.spinner("Generating interview questions..."):
                        for r in st.session_state.resumes:
                            result = get_interview_questions(r["resume_text"], target_role_text)
                            if result.get("error"):
                                failed_questions.append(r["filename"])
                                r["interview_questions"] = None
                            else:
                                r["interview_questions"] = result["questions"]
                    if failed_questions:
                        st.warning(f"Could not generate questions for: {', '.join(failed_questions)}. Please try again.")

            for r in st.session_state.resumes:
                if r.get("interview_questions"):
                    st.subheader(r["filename"])
                    for i, question in enumerate(r["interview_questions"]):
                        st.write(f"**Q{i+1}:** {question}")
                        answer_key = f"answer_{r['filename']}_{i}"
                        answer = st.text_area("Your answer", key=answer_key, label_visibility="collapsed")

                        if "qa_pairs" not in r:
                            r["qa_pairs"] = {}
                        if i not in r["qa_pairs"]:
                            r["qa_pairs"][i] = {}
                        r["qa_pairs"][i]["question"] = question
                        r["qa_pairs"][i]["answer"] = answer

                        feedback_key = f"feedback_btn_{r['filename']}_{i}"
                        if st.button("Get Feedback", key=feedback_key):
                            if not answer.strip():
                                st.warning("Please write an answer first.")
                            else:
                                with st.spinner("Evaluating your answer..."):
                                    feedback_result = get_answer_feedback(r["resume_text"], question, answer)
                                if feedback_result.get("error"):
                                    st.warning("Could not generate feedback. Please try again.")
                                else:
                                    r["qa_pairs"][i]["feedback_result"] = feedback_result

                        if r["qa_pairs"][i].get("feedback_result"):
                            fb = r["qa_pairs"][i]["feedback_result"]
                            st.metric("Answer Score", f"{fb['score']}/100")
                            st.write(f"**Feedback:** {fb['feedback']}")
                            with st.expander("See a recommended answer"):
                                st.write(fb["recommended_answer"])

                        st.divider()

    if st.session_state.jd_text:
        with tab2:
            if not st.session_state.resumes:
                st.info("Please analyze your resumes first")
            else:
                if st.button("Analyze JD Match"):
                    failed_matches = []
                    with st.spinner("Comparing your resume to the job description..."):
                        for r in st.session_state.resumes:
                            match = get_jd_match(
                                r["resume_text"],
                                st.session_state.jd_text,
                                r["resume_json"]
                            )
                            if match.get("error"):
                                failed_matches.append(r["filename"])
                                r["jd_match_result"] = None
                            else:
                                r["jd_match_result"] = match
                    if failed_matches:
                        st.warning(f"JD match failed for: {', '.join(failed_matches)}. Please try again.")

                    st.session_state.resumes.sort(
                        key=lambda r: r["jd_match_result"]["match_score"] if r["jd_match_result"] else 0,
                        reverse=True
                    )

                any_matches = any(r["jd_match_result"] for r in st.session_state.resumes)

                if any_matches:
                    for r in st.session_state.resumes:
                        match = r["jd_match_result"]
                        review = r["resume_json"]

                        st.subheader(r["filename"])

                        if match is None:
                            st.warning("JD match failed for this resume.")
                            st.divider()
                            continue

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Resume Score", f"{review['score']}/100")
                        with col2:
                            st.metric("JD Match Score", f"{match['match_score']}/100")

                        st.write("**Matched Keywords**")
                        if match["matched_keywords"]:
                            for item in match["matched_keywords"]:
                                st.write(f"- {item}")
                        else:
                            st.write("No matched keywords found.")

                        st.write("**Missing Keywords**")
                        if match["missing_keywords"]:
                            for item in match["missing_keywords"]:
                                st.write(f"- {item}")
                        else:
                            st.write("No missing keywords — great match!")

                        st.write("**JD-Specific Suggestions**")
                        if match["jd_suggestions"]:
                            for item in match["jd_suggestions"]:
                                st.write(f"- {item}")
                        else:
                            st.write("No specific suggestions.")

                        st.divider()

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.divider()
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


if st.session_state.resumes:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask anything about your resume..")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Thinking..."):
            clean_history = build_chat_messages()
            reply = chat(clean_history)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.write(reply)

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()