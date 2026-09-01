from hashlib import new

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from extract_text import HEADER_MARKER
model=SentenceTransformer("all-MiniLM-L6-v2")
SECTION_ALIASES = {
                   "SUMMARY":"SUMMARY",
                   "PROFILE":"SUMMARY",
                   "OBJECTIVE":"SUMMARY",
                   "EXPERIENCE":"EXPERIENCE",
                   "EMPLOYMENT HISTORY":"EXPERIENCE",
                   "WORK EXPERIENCE":"EXPERIENCE",
                   "SKILLS":"SKILLS",
                   "TECHNICAL SKILLS":"SKILLS",
                   "CORE SKILLS":"SKILLS",
                   "EDUCATION":"EDUCATION",
                   "PROJECTS":"PROJECTS",
                   "LEADERSHIP & EXTRACURRICULAR ACTIVITIES":"LEADERSHIP & EXTRACURRICULAR ACTIVITIES",
                   "EXTRACURRICULAR ACTIVITIES":"LEADERSHIP & EXTRACURRICULAR ACTIVITIES",
                   "EXTRACURRICULAR":"LEADERSHIP &EXTRACURRICULAR"}

def classify_header(raw_header_text):
    """Returns the cannonical section name if thi header is recongnized,
    else 'OTHER:<raw text>' so unfamiliar but real headers still get their own section."""
    cannonical=SECTION_ALIASES.get(raw_header_text.strip().upper())
    if cannonical:
        return cannonical
    return f"OTHER:{raw_header_text.strip()}"

def split_into_sections(text):
    """Splits extracted resume text into sections,keyed by cannonical section name
       or 'OTHER:<raw>'/PREAMBLE for unrecognized but real headers/ pre-header content.
       Relies on extract.py having tagged real headers with HEADER_MARKER via font-siz/
       boldness detection,rather tthan guessing from plain text vocabulary."""
       
    lines = text.split("\n")
    sections = {}
    current_section = None
    current_lines = []

    for line in lines:
        if line.startswith(HEADER_MARKER):
                header_text=line[len(HEADER_MARKER):].strip()
                if current_section:
                 existing=sections.get(current_section, "")
                 new_text="\n".join(current_lines).strip()
                 sections[current_section]=(existing +"\n" + new_text).strip() if existing else new_text
                else:
                    preamble_text="\n".join(current_lines).strip()
                    if preamble_text:
                        sections["PREAMBLE"]=preamble_text
                current_section=classify_header(header_text)
                current_lines=[]
        else:
                 current_lines.append(line)

    if current_section:
         existing=sections.get(current_section, "")
         new_text="\n".join(current_lines).strip()
         sections[current_section]=(existing +"\n" + new_text).strip() if existing else new_text
    else:
        preamble_text="\n".join(current_lines).strip()
        if preamble_text:
            sections["PREAMBLE"]=preamble_text

    return sections

def split_experiences_into_jobs(experience_text):
    lines=experience_text.split("\n")
    has_bullets=any(
        line.strip().startswith("-") or line.strip().startswith("*") or line.strip().startswith("•")
        for line in lines
    )
    if not has_bullets:
        return [experience_text.strip()] if experience_text.strip() else []
    jobs=[]
    current_job_lines=[]
    seen_bullet_in_current_job=False
    for line in lines:
        stripped=line.strip()
        if stripped=="":
          continue
        is_bullet=stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•")
        if is_bullet:
            current_job_lines.append(line)
            seen_bullet_in_current_job=True
            continue
        first_alpha=next((c for c in stripped if c.isalpha()),None)
        looks_like_wrapped_continuation=(
            seen_bullet_in_current_job and first_alpha is not None and first_alpha.islower()
        )
        if looks_like_wrapped_continuation:
            if current_job_lines:
             current_job_lines[-1]=current_job_lines[-1].rstrip()+" "+stripped
            else:
             current_job_lines.append(line)
        elif seen_bullet_in_current_job:
            if current_job_lines:
                jobs.append("\n".join(current_job_lines).strip())
            current_job_lines=[line]
            seen_bullet_in_current_job=False
        else:
            current_job_lines.append(line)


    if current_job_lines:
        jobs.append("\n".join(current_job_lines).strip())

    return jobs

def split_into_items(section_text):
    lines = section_text.split("\n")
    items = []
    current_item_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        is_bullet = stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•")
        is_header = "|" in stripped and not is_bullet

        if is_bullet:
            current_item_lines.append(line)
        elif is_header or not current_item_lines:
            
            if current_item_lines:
                items.append("\n".join(current_item_lines).strip())
            current_item_lines = [line]
        else:
            
            if current_item_lines:
                current_item_lines[-1] = current_item_lines[-1].rstrip() + " " + stripped

    if current_item_lines:
        items.append("\n".join(current_item_lines).strip())

    return items

def build_tagged_chunks(text,candidate_name):
    sections=split_into_sections(text)
    chunks=[]
    for section_name, section_text in sections.items():
        if section_name=="EXPERIENCE":
            jobs=split_experiences_into_jobs(section_text)
            for job in jobs:
                chunks.append({
                    "text":job,
                    "candidate":candidate_name,
                    "section":"EXPERIENCE"
                })
        elif section_name=="PROJECTS":
            projects=split_into_items(section_text)
            for proj in projects:
                chunks.append({
                    "text":proj,
                    "candidate":candidate_name,
                    "section":"PROJECTS"
                })
        else:
            chunks.append({
                "text": section_text,
                "candidate": candidate_name,
                "section": section_name
            })
    return chunks

def embed_chunks(tagged_chunks):
    texts=[chunk["text"]for chunk in tagged_chunks]
    embeddings=model.encode(texts)
    return embeddings

def build_faiss_index(embeddings):
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve_context(query,index,tagged_chunks,top_k=3):
    query_embedding=model.encode([query]).astype("float32")
    distances,indices=index.search(query_embedding,top_k)

    results=[]
    for i in indices[0]:
        results.append(tagged_chunks[i])
    return results

def retrieve_context_per_candidate(query,tagged_chunks,top_k_per_candidate=2):
    candidates=set(chunk["candidate"]for chunk in tagged_chunks)
    all_results=[]
    for candidate in candidates:
        candidate_chunks=[c for c in tagged_chunks if c["candidate"]==candidate]
        candidate_embeddings=embed_chunks(candidate_chunks)
        candidate_index=build_faiss_index(candidate_embeddings)
        results=retrieve_context(query,candidate_index,candidate_chunks,top_k_per_candidate)
        all_results.extend(results)
    return all_results