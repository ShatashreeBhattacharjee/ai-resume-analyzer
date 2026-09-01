import os
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT ="""You are an expert resume reviewer. You have access to the candidate's resume and a job description.

For structured analysis requests, respond in valid JSON only. No extra text, no markdown, no code fences.

For all follow-up questions, respond in plain conversational English.
No JSON, no code fences, no markdown."""

RESUME_REVIEW_PROMPT = """Review this resume and return your analysis as JSON in exactly this format:
{
  "reasoning": "<brief step-by-step evaluation: comment on formatting/structure, whether achievements are quantified with numbers/metrics, relevance and clarity of listed skills, and overall completeness. 3-5 sentences.>",
  "score": <number between 0-100>,
  "strengths": ["<specific strength from the resume>", "<another strength>"],
  "weaknesses": ["<specific weakness from the resume>", "<another weakness>"],
  "suggestions": ["<specific actionable suggestion>", "<another suggestion>"]
}

Here are examples of the level of specificity expected:

Example snippet A (strong): "Led migration of 12 microservices to AWS ECS, reducing deployment time by 40%. Built CI/CD pipelines using Jenkins and Docker, cutting release cycle from 2 weeks to 2 days."
- GOOD strength: "Quantifies impact clearly — 40% deployment time reduction and release cycle cut from 2 weeks to 2 days"
- BAD strength (too vague, avoid this style): "Has good DevOps experience"

Example snippet B (weak): "Responsible for various backend tasks. Worked with different technologies. Helped improve system performance."
- GOOD weakness: "No specific technologies named — 'different technologies' gives no verifiable detail"
- BAD weakness (too vague, avoid this style): "Resume needs more detail"

Always write strengths, weaknesses, and suggestions in the GOOD style shown above — specific and grounded in the actual resume text, never generic.

Important rules:
- Write the reasoning field FIRST, before deciding the score
- The score must be justified by what you wrote in reasoning — don't contradict your own reasoning
- Score must reflect the ACTUAL quality of the resume
- Strengths must be specific to THIS resume only
- Weaknesses must be specific to THIS resume only
- Never copy the example values above, and never copy the example snippets A or B — they are style references only, not content to reuse

Respond with ONLY the JSON object above, filled in. Do NOT:
- Add any text before the opening {
- Add any text after the closing }
- Wrap the JSON in markdown code fences (no ```json)
- Include comments or explanations within the JSON (outside the reasoning field)

Your entire response must start with { and end with }.
Before responding, mentally verify: is this a single, complete, valid JSON object with no other text?

Resume:
"""

JD_MATCH_PROMPT = """Compare the resume against the job description below and return JSON in exactly this format:
{
  "reasoning": "<brief step-by-step evaluation: list which key requirements/skills from the JD are present in the resume and which are missing, note how important each missing item seems to the role, then explain how that adds up to the match_score. 3-5 sentences.>",
  "match_score": <number between 0-100>,
  "matched_keywords": ["<keyword actually found in resume>"],
  "missing_keywords": ["<keyword in JD but missing from resume>"],
  "jd_suggestions": ["<specific suggestion based on gaps>"]
}

Here are examples of the level of specificity expected:

Example JD requirement: "3+ years experience with Python and REST API development"
Example resume line: "Built REST APIs in Python using Flask for 2 years at a startup"
- GOOD matched_keyword: "Python" (explicitly used)
- GOOD missing_keyword note: "REST API development" is present but the JD's 3+ years is not clearly met — resume shows 2 years
- BAD matched_keyword (too vague, avoid this style): "Has relevant experience"

Example JD requirement: "Experience with Kubernetes and container orchestration"
Example resume line: (no mention of Kubernetes or containers anywhere)
- GOOD missing_keyword: "Kubernetes"
- GOOD jd_suggestion: "Consider highlighting any container/orchestration exposure, even from coursework or personal projects, since Kubernetes is explicitly required"
- BAD jd_suggestion (too vague, avoid this style): "Add more skills to your resume"

Always write matched_keywords, missing_keywords, and jd_suggestions in the GOOD style shown above — specific and grounded in the actual resume and JD text, never generic.

Important rules:
- Write the reasoning field FIRST, before deciding the match_score
- The match_score must be consistent with matched_keywords and missing_keywords — don't contradict your own reasoning
- match_score must reflect ACTUAL keyword overlap
- matched_keywords must only include keywords ACTUALLY in the resume
- missing_keywords must only include keywords ACTUALLY missing
- Never copy the example values above, and never copy the example JD/resume lines — they are style references only, not content to reuse

Respond with ONLY the JSON object above, filled in. Do NOT:
- Add any text before the opening {
- Add any text after the closing }
- Wrap the JSON in markdown code fences (no ```json)
- Include comments or explanations within the JSON (outside the reasoning field)

Your entire response must start with { and end with }.
Before responding, mentally verify: is this a single, complete, valid JSON object with no other text?

Job Description:
"""

CHAT_SYSTEM_PROMPT = """You are a career assistant helping a job seeker. You have access to a review of their resume and, if available, how well it matches a job description.

Use this information as background context, but don't limit yourself to only resume critique — answer whatever the user actually asks, including questions about career direction, role fit, skill gaps, interview prep, or general advice.

Respond in plain conversational English. No JSON, no code fences, no markdown."""

INTERVIEW_QUESTIONS_PROMPT = """Based on this resume and the target role below, generate interview questions this candidate is likely to be asked. Return your response as JSON in exactly this format:
{
  "reasoning": "<brief step-by-step evaluation: identify the candidate's key experience areas, the target role's core requirements, and any notable gaps between them that an interviewer would likely probe. 3-5 sentences.>",
  "questions": ["<specific interview question>", "<another question>"]
}

Here are examples of the level of specificity expected:

Example resume snippet: "Built REST APIs in Python using Flask, reduced API response time by 30% through query optimization"
- GOOD question: "You mentioned reducing API response time by 30% — walk me through how you identified the bottleneck and what you changed"
- BAD question (too generic, avoid this style): "Tell me about your experience with APIs"

Example target role requirement: "Experience with Kubernetes and container orchestration" with no matching resume content
- GOOD question: "This role involves container orchestration with Kubernetes — what's your experience with containerized deployments, even outside of Kubernetes specifically?"
- BAD question (too generic, avoid this style): "Do you know Kubernetes?"

Always write questions in the GOOD style shown above — grounded in the actual resume content and target role, referencing specific projects, technologies, or claims where possible, never generic.

Important rules:
- Write the reasoning field FIRST, before generating questions
- Questions must be grounded in what's ACTUALLY in the resume or ACTUALLY required by the target role — don't invent experience the candidate doesn't have
- Include a mix: some questions probing specific resume claims, some probing gaps against the target role
- Generate 5-8 questions
- Never copy the example values above, and never copy the example snippets — they are style references only, not content to reuse

Respond with ONLY the JSON object above, filled in. Do NOT:
- Add any text before the opening {
- Add any text after the closing }
- Wrap the JSON in markdown code fences (no ```json)
- Include comments or explanations within the JSON (outside the reasoning field)

Your entire response must start with { and end with }.
Before responding, mentally verify: is this a single, complete, valid JSON object with no other text?

Resume:
"""

ANSWER_FEEDBACK_PROMPT = """Given this candidate's resume, an interview question they were asked, and their answer, evaluate the answer and return your response as JSON in exactly this format:
{
  "reasoning": "<brief step-by-step evaluation: what the question was probing for, whether the answer addresses it directly, whether it's specific and grounded in real experience versus vague, and what's missing. 3-5 sentences.>",
  "score": <number between 0-100>,
  "feedback": "<specific, actionable feedback on what was strong and what to improve>",
  "recommended_answer": "<a strong example answer to this specific question, grounded in the candidate's actual resume content>"
}

Here are examples of the level of specificity expected:

Example question: "You mentioned reducing API response time by 30% — walk me through how you identified the bottleneck and what you changed"
Example weak answer: "I looked at the code and made it faster by optimizing the database queries."
- GOOD feedback: "This answer states the fix (query optimization) but skips the diagnostic process — how did you identify the bottleneck was queries specifically? Interviewers want to see your debugging process, not just the outcome."
- BAD feedback (too vague, avoid this style): "Could be more detailed."

Always write feedback and the recommended_answer in the GOOD style shown above — specific, grounded in the actual resume and question, never generic filler.

Important rules:
- Write the reasoning field FIRST, before deciding the score
- The score must be justified by what you wrote in reasoning — don't contradict your own reasoning
- If the answer is empty or clearly off-topic, score it low and say so plainly in feedback — don't be falsely encouraging
- recommended_answer must be grounded in the candidate's ACTUAL resume content — don't invent experience they don't have
- Never copy the example values above — they are style references only, not content to reuse

Respond with ONLY the JSON object above, filled in. Do NOT:
- Add any text before the opening {
- Add any text after the closing }
- Wrap the JSON in markdown code fences (no ```json)
- Include comments or explanations within the JSON (outside the reasoning field)

Your entire response must start with { and end with }.
Before responding, mentally verify: is this a single, complete, valid JSON object with no other text?

Resume:
"""

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(messages):            
    response=client.chat.completions.create(
      model= MODEL,
      messages=messages,
      temperature=0.2,
      max_tokens=1024
    )
    return response.choices[0].message.content

def extract_json_substring(raw_text):
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or start > end:
        return None

    candidate = raw_text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def blind_retry(messages):
    raw_retry = chat(messages)

    try:
        return json.loads(raw_retry)
    except json.JSONDecodeError:
        return extract_json_substring(raw_retry)

def corrective_retry(messages, broken_raw_output):
    corrected_messages = messages + [
        {"role": "assistant", "content": broken_raw_output},
        {"role": "user", "content": "Your previous response was not valid JSON. Return ONLY a valid JSON object — no explanation, no markdown, no preamble. Start with { and end with }."}
    ]

    raw_final = chat(corrected_messages)

    try:
        return json.loads(raw_final)
    except json.JSONDecodeError:
        return extract_json_substring(raw_final)

def call_groq_with_json_retry(messages):
    raw_response = chat(messages)

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    result = extract_json_substring(raw_response)
    if result is not None:
        return result

    result = blind_retry(messages)
    if result is not None:
        return result
    result = corrective_retry(messages, raw_response)
    if result is not None:
        return result

    return {
        "error": True,
        "message": "Failed to get valid JSON after multiple attempts.",
        "raw": raw_response
    }



def get_resume_review(resume_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME_REVIEW_PROMPT + resume_text},
        ]

    review = call_groq_with_json_retry(messages)

    return review

def get_jd_match(resume_text,jd_text,review):
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME_REVIEW_PROMPT + resume_text},
        {"role": "assistant", "content": json.dumps(review)},
        {"role": "user", "content": JD_MATCH_PROMPT + jd_text},
        ]
    
    match = call_groq_with_json_retry(messages)

    return match

def format_resume_summary(resume_json):
    strengths="\n".join(f"-{item}" for item in resume_json["strengths"])
    weakenesses="\n".join(f"-{item}" for  item in resume_json["weaknesses"])
    suggestions="\n".join(f"-{item}" for item in resume_json["suggestions"])
    summary=f"""Resume Review Summary:
Score:{resume_json['score']}/100
Strengths:
{strengths}
Weaknesses:
{weakenesses}
Suggestions:
{suggestions}
"""
    return summary

def format_jd_summary(jd_match_json):
    matched="\n".join(f"-{item}" for item in jd_match_json["matched_keywords"])
    missing="\n".join(f"-{item}" for item in jd_match_json["missing_keywords"])
    suggestions="\n".join(f"-{item}" for item in jd_match_json["jd_suggestions"])
    summary = f"""JD Match Summary:
JD Match Score: {jd_match_json["match_score"]}/100

Matched Keywords:
{matched}

Missing Keywords:
{missing}

JD-Specific Suggestions:
{suggestions}
"""
    return summary

def get_interview_questions(resume_text, target_role_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INTERVIEW_QUESTIONS_PROMPT + resume_text + "\n\nTarget role:\n" + target_role_text},
    ]

    questions = call_groq_with_json_retry(messages)

    return questions
    
def get_answer_feedback(resume_text, question, answer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ANSWER_FEEDBACK_PROMPT + resume_text + f"\n\nInterview question:\n{question}\n\nCandidate's answer:\n{answer}"},
    ]

    feedback = call_groq_with_json_retry(messages)

    return feedback