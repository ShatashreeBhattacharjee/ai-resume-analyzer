from extract_text import extract_text
from rag import split_into_sections, build_tagged_chunks
import os

resume_files = os.listdir("resumes")

for filename in resume_files:
    if not filename.endswith(".pdf"):
        continue
    path = os.path.join("resumes", filename)
    print("=" * 60)
    print(filename)
    print("=" * 60)

    text = extract_text(path)
    sections = split_into_sections(text)
    print("Sections found:", list(sections.keys()))

    chunks = build_tagged_chunks(text, filename)
    print(f"Total chunks: {len(chunks)}")
    for c in chunks:
        preview = c["text"][:70].replace("\n", " / ")
        print(f"  [{c['section']}] {preview}...")
    print()