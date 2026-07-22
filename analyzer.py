import os
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT ="""You are an expert resume reviewer. You have access to the candidate's resume and a job description.

For structured analysis requests, respond in valid JSON only. No extra text, no markdown, no code fences.

For all follow-up questions, respond in plain conversational English.
No JSON, no code fences, no markdown."""

RESUME_REVIEW_PROMPT = """Review this resume and return your analysis as JSON in exactly this format:
{
  "score": <number between 0-100>,
  "strengths": ["<specific strength from the resume>", "<another strength>"],
  "weaknesses": ["<specific weakness from the resume>", "<another weakness>"],
  "suggestions": ["<specific actionable suggestion>", "<another suggestion>"]
}

Important rules:
- Score must reflect the ACTUAL quality of the resume
- Strengths must be specific to THIS resume only
- Weaknesses must be specific to THIS resume only
- Never copy the example values above

Resume:
"""

JD_MATCH_PROMPT = """Compare the resume against the job description below and return JSON in exactly this format:
{
  "match_score": <number between 0-100>,
  "matched_keywords": ["<keyword actually found in resume>"],
  "missing_keywords": ["<keyword in JD but missing from resume>"],
  "jd_suggestions": ["<specific suggestion based on gaps>"]
}

Important rules:
- match_score must reflect ACTUAL keyword overlap
- matched_keywords must only include keywords ACTUALLY in the resume
- missing_keywords must only include keywords ACTUALLY missing
- Never copy the example values above

Job Description:
"""

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(messages):            
    response=client.chat.completions.create(
      model= MODEL,
      messages=messages,
      temperature=0.2
    )
    return response.choices[0].message.content

def get_resume_review(resume_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME_REVIEW_PROMPT + resume_text},
    ]

    raw_review = chat(messages)

    try:
        review = json.loads(raw_review)
    except json.JSONDecodeError:
        return None

    return review

def get_jd_match(resume_text,jd_text,review):
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME_REVIEW_PROMPT + resume_text},
        {"role": "assistant", "content": json.dumps(review)},
        {"role": "user", "content": JD_MATCH_PROMPT + jd_text},
        ]
    raw_match=chat(messages)

    try:
        match = json.loads(raw_match)
    except json.JSONDecodeError:
        return None

    return match
    
