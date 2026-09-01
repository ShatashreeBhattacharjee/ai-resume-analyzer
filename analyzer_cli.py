import os 
import json 

from dotenv import load_dotenv
from groq import Groq
import pdfplumber

load_dotenv()

MODEL = "openai/gpt-oss-20b"
RESUMES_DIR = "resumes"
JD_PATH = "jobs/job_description.txt"
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
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

resumes={}
for filename in os.listdir(RESUMES_DIR):
    if filename.endswith(".pdf"):
        filepath=os.path.join(RESUMES_DIR,filename)
        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        resumes[filename] = text
        print(f" Loaded: {filename}")

if os.path.exists(JD_PATH):
    with open(JD_PATH,"r") as f:
        jd_text=f.read().strip()

else:
    jd_text=None
    print("No job description found, proceeding with resume analysis only.\n")
        
def chat(messages):            
    response=client.chat.completions.create(
      model= MODEL,
      messages=messages,
      temperature=0.2
    )
    return response.choices[0].message.content

results = []

for filename, resume_text in resumes.items():
    print(f"\nANALYZING: {filename}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RESUME_REVIEW_PROMPT + resume_text},
    ]

    raw_review = chat(messages)
    try:
        review = json.loads(raw_review)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON for {filename}, skipping...")
        continue

    messages.append({"role": "assistant", "content": raw_review})
    match=None
    if jd_text:
        messages.append({ "role": "user", "content": JD_MATCH_PROMPT + jd_text })
        raw_match=chat(messages)
        try:
            match=json.loads(raw_match)
            messages.append({"role": "assistant", "content": raw_match})
        except json.JSONDecodeError:
             print(f"Warning: Could not parse JD match JSON for {filename}, skipping match...")
             match = None
    results.append({
        "filename": filename,
        "review": review,
        "match": match
    })

if jd_text:
    results.sort(key=lambda x: x["match"]["match_score"] if x["match"] else 0, reverse=True)

    print("\n" + "=" * 50)
    print("        RANKING BY JD MATCH SCORE")
    print("=" * 50)
    for i, r in enumerate(results):
        score = r["match"]["match_score"] if r["match"] else "N/A"
        print(f"#{i+1}  {r['filename']:<20} {score} / 100")
        

for r in results:
    print("\n" + "-" * 50)
    print(f" RESUME: {r['filename']}")
    print(f" RESUME SCORE: {r['review']['score']} / 100")
    print("-" * 50)

    print("\nSTRENGTHS:")
    for item in r["review"]["strengths"]:
        print(f"  • {item}")
    print("\nWEAKNESSES:")
    for item in r["review"]["weaknesses"]:
        print(f"  • {item}")
    print("\nSUGGESTIONS:")
    for item in r["review"]["suggestions"]:
        print(f"  • {item}")

    if r["match"]:
        print("\n" + "=" * 50)
        print(f" JD MATCH SCORE: {r['match']['match_score']} / 100")
        print("=" * 50)
        print("\nMATCHED KEYWORDS:")
        for item in r["match"]["matched_keywords"]:
            print(f"   {item}")
        print("\nMISSING KEYWORDS:")
        for item in r["match"]["missing_keywords"]:
            print(f"   {item}")
        print("\nJD-SPECIFIC SUGGESTIONS:")
        for item in r["match"]["jd_suggestions"]:
            print(f"  • {item}")


print("\n" + "-" * 50)
print("CHAT MODE - ask anything about the resumes or the job")
print("Type 'EXIT' to quit")
print("-" * 50 + "\n")


chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
for r in results:
    chat_messages.append({
        "role": "user",
        "content": f"Here is the analysis for {r['filename']}:\n{json.dumps(r['review'], indent=2)}"
    })
    chat_messages.append({
        "role": "assistant",
        "content": f"I have reviewed {r['filename']} and stored the analysis."
    })
    if r["match"]:
        chat_messages.append({
            "role": "user",
            "content": f"JD match for {r['filename']}:\n{json.dumps(r['match'], indent=2)}"
        })
        chat_messages.append({
            "role": "assistant",
            "content": f"I have the JD match results for {r['filename']}."
        })

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    if not user_input:
        continue

    chat_messages.append({"role": "user", "content": user_input})
    reply = chat(chat_messages)
    chat_messages.append({"role": "assistant", "content": reply})
    print(f"\nAI: {reply}\n")
