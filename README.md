# AI Resume Analyzer & Career Copilot

An end-to-end resume analysis and interview-prep tool that scores resumes, compares multiple candidates against a single job description, and generates personalized mock interview questions with AI-graded feedback, all grounded in a custom-built RAG pipeline.

## Overview

Job seekers often don't know how their resume actually reads to an interviewer, or what they'll be asked once they land one. This tool closes that gap. It analyzes a resume's strengths and weaknesses, benchmarks it against a target job description, and generates role-specific interview questions grounded in the candidate's real experience, with AI feedback on their practice answers.

Built with Python, Streamlit, and the Groq API (gpt-oss-20b), using a custom RAG pipeline (sentence-transformers and FAISS) for accurate multi-resume retrieval and context-aware chat.

## Features

- **Resume Scoring** - a 0-100 score with specific, evidence-based strengths, weaknesses, and improvement suggestions, not generic advice.
- **Multi-Resume Comparison** - upload and analyze several resumes at once, with each candidate's full breakdown displayed in sequence for easy comparison.
- **Job Description Matching** - matched and missing keywords plus JD-specific suggestions, with resumes automatically ranked by match score.
- **Robust PDF Parsing** - a custom column-aware extraction pipeline correctly handles two-column sidebar resume templates, which naive text extraction typically scrambles. Uses a band-based, majority-vote algorithm to detect the column gap even when a centered header spans the full page width, something off-the-shelf PDF text extraction gets wrong.
- **RAG-Powered Chat** - ask follow-up questions about any candidate's resume, with answers grounded in retrieved, relevant chunks rather than the model guessing from memory.
- **AI-Generated Interview Prep** - generates interview questions tailored to the candidate's actual resume and the target role, from a pasted job description or a typed role. Questions probe both real accomplishments and gaps between the resume and the role's requirements.
- **AI-Graded Mock Answers** - practice answering generated questions and get a score, specific feedback, and a model answer grounded in your own resume content, not a generic template response.

## How It Works: The RAG Pipeline

Getting clean text out of real-world resumes turned out to be the hardest part of this project. Resume templates vary wildly, and a naive pdfplumber text extraction call breaks badly on common formats.

**Column-aware extraction.** Many resume templates use a two-column sidebar layout, with contact info and skills on one side and experience on the other. Extracting text top to bottom across the whole page interleaves these columns mid-sentence, producing garbage. To fix this, the extractor scans the page in horizontal bands rather than all at once, since a single full-width row, like a centered name, can span the gutter and hide the real column gap if you look at the page as a whole. Band by band, it finds the widest gap between text blocks, then takes a majority vote across all bands to find the gap position that recurs most consistently. This correctly separates the columns even on templates with mixed single and multi-column sections on the same page.

**Structural header detection.** Instead of matching section headers against a fixed word list, which breaks on any resume that phrases things unconventionally, headers are detected by comparing each line's font size and boldness against the page's own body-text baseline, with a fallback to structural signals such as all-caps text or multi-word title case for templates with no visual distinction between headers and body text. Lines containing commas are excluded from header detection entirely, since real section headers never contain commas, while job titles and addresses often do.

**Retrieval-augmented chat.** Extracted resume text is split into semantically meaningful chunks, by section and by individual job or project within the experience section, embedded with sentence-transformers, and indexed with FAISS. When comparing multiple resumes, retrieval is done per candidate rather than globally. This prevents one strong candidate's content from crowding out relevant chunks from a weaker candidate when the app searches for an answer, a real failure mode of naive multi-document RAG.

## Setup & Usage

**Requirements:** Python 3.14, a Groq API key (https://console.groq.com)

1. Clone the repo and install dependencies:

   ```
   git clone https://github.com/ShatashreeBhattacharjee/ai-resume-analyzer.git
   cd ai-resume-analyzer
   pip install -r requirements.txt
   ```

2. Add your Groq API key to a .env file in the project root:

   ```
   GROQ_API_KEY=your_key_here
   ```

3. Run the app:

   ```
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints, usually http://localhost:8501, upload a resume PDF, and click Analyze.