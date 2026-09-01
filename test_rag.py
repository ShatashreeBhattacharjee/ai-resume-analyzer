import os
from extract_text import extract_text, is_header_line

def explore_resume():
     from rag import build_tagged_chunks, split_into_sections,split_experiences_into_jobs,split_into_items, embed_chunks 
     from rag import build_faiss_index, retrieve_context
     filepath = "resumes/Alex Carter.pdf"
     candidate_name = os.path.splitext(os.path.basename(filepath))[0]

     text = extract_text(open("resumes/Alex Carter.pdf", "rb")) 
     sections = split_into_sections(text)

     for section, content in sections.items():
         print(f"--- {section} ---")
         print(content)
         print()
     jobs = split_experiences_into_jobs(sections.get("EXPERIENCE", ""))
     for i, job in enumerate(jobs):
         print(f"--- JOB {i+1} ---")
         print(job)
         print()
     projects = split_into_items(sections.get("PROJECTS", ""))
     for i, proj in enumerate(projects):
         print(f"--- PROJECT {i+1} ---")
         print(proj)
         print()
     tagged_chunks=build_tagged_chunks(text,candidate_name)
     for chunk in tagged_chunks:
      print(chunk)
      print()

     embeddings=embed_chunks(tagged_chunks)
     print(f"Number of chunks:{len(tagged_chunks)}")
     print(f"Number of embeddings:{len(embeddings)}")
     print(f"Shape of one embedding:{embeddings[0].shape}")
     print(f"First few numbers of first embedding:{embeddings[0][:5]}")

     index=build_faiss_index(embeddings)
     print(f"Number of vectors in index:{index.ntotal}")

     results=retrieve_context("What backend technologies does this candidate know?",index,tagged_chunks,top_k=3)
     for r in results:
         print(r)
         print()

def test_comma_excludes_header_detection():
    line = {
        "text": "Travel Agent, Amazon Travel, New York",
        "size": 12,
        "bold": True,
    }
    body_size = 10
    assert is_header_line(line, body_size) is False

def test_narrative_paragraph_job_stays_one_job():
    from rag import split_experiences_into_jobs

    experience_text = (
        "Senior Software Engineer\n"
        "TechCorp | Jan 2020 - Present\n"
        "Led a team of five engineers building a distributed backend platform "
        "that processes millions of transactions daily.\n"
        "- Reduced deployment time by 40% through CI/CD improvements\n"
        "- Migrated legacy services to Kubernetes\n"
        "- Mentored two junior engineers\n"
    )

    jobs = split_experiences_into_jobs(experience_text)

    assert len(jobs) == 1

def test_wrapped_bullet_continuation_stays_in_same_job():
    from rag import split_experiences_into_jobs

    experience_text = (
        "Backend Engineer\n"
        "Acme Inc | Jun 2019 - Dec 2021\n"
        "- Built and maintained internal tooling used by over 50 engineers "
        "across the organization to speed up\n"
        "deployment workflows and reduce manual review time\n"
        "- Owned the on-call rotation for three critical services\n"
    )

    jobs = split_experiences_into_jobs(experience_text)

    assert len(jobs) == 1


if __name__ == "__main__":
    explore_resume()

