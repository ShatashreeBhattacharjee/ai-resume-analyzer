import pdfplumber
def extract_resume_text(uploaded_file):
    text=""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text


