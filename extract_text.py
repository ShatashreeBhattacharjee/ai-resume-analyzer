import pdfplumber
import re
from collections import Counter

HEADER_MARKER="\x01HEADER\x01"
MONTH_NAMES={"JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
             "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}

def looks_like_date_range(text):
    if re.search(r"\b\d{4}\b",text):
        return True
    words=re.findall(r"[A-Za-z]+", text.upper())
    return any(word in MONTH_NAMES for word in words)

def looks_like_address_fragment(text):
    if re.search(r",\s*[A-Z]{2}(\s+\d{5}(-\d{4})?)?", text):
        return True 
    if re.fullmatch(r"\d{5}(-\d{4})?", text):
        return True
    return False

def find_column_gap(page,min_gap_width=15,margin_frac=0.2,band_height=40):
    """Scan the page in horizontal bands rather than all at once,since a single full width row like a centered name can bridge the gutter and hide a real column gap.
    Take the gap position that recurs across the most bands.
    Returns(gap_start,gap_end) or None if this looks like a single column page."""
    words=page.extract_words()
    if not words:
        return None
    page_width=page.width
    page_height=page.height
    band_gaps=[]
    top=0
    while top<page_height:
        bottom=top+band_height
        band_words=[w for w in words if w["top"]<bottom]
        top=bottom
        if len(band_words)<2:
            continue
        intervals=sorted((w["x0"],w["x1"]) for w in band_words)
        merged=[]
        for x0,x1 in intervals:
            if merged and x0<=merged[-1][1]:
                merged[-1]=(merged[-1][0],max(merged[-1][1],x1))
            else:
                merged.append((x0,x1))
            if len(merged)<2:
                continue
            gaps=[]
            for i in range(len(merged)-1):
                gap_start,gap_end=merged[i][1],merged[i+1][0]
                gaps.append((gap_end-gap_start,gap_start,gap_end))
            gaps.sort(reverse=True)
            for width,gap_start,gap_end in gaps:
                if gap_start<page_width * margin_frac or gap_end>page_width*(1-margin_frac):
                    continue
                if width<min_gap_width:
                    break
                band_gaps.append((gap_start,gap_end))
                break
    if not band_gaps:
        return None

    bucket_size=20
    buckets=Counter()
    bucket_members={}
    for gap_start,gap_end in band_gaps:
        mid=(gap_start+gap_end)/2
        bucket=round(mid/bucket_size)
        buckets[bucket]+=1
        bucket_members.setdefault(bucket, []).append((gap_start, gap_end))
    best_bucket,count=buckets.most_common(1)[0]
    if count<3:
        return None

    members=bucket_members[best_bucket]
    avg_start=sum(g[0] for g in members)/len(members)
    avg_end=sum(g[1] for g in members)/len(members)
    return (avg_start,avg_end)

def group_words_into_lines(words,tolerance=3):
    """Group words into visual lines by y-position rather than relying on pdfplumber's default line breaks,so we can compute one font size per line."""
    if not words:
        return[]
    words_sorted=sorted(words,key=lambda w:w["top"])
    lines=[]
    current=[words_sorted[0]]
    current_top=words_sorted[0]["top"]
    for w in words_sorted[1:]:
        if abs(w["top"]-current_top)<=tolerance:
            current.append(w)

        else:
            lines.append(current)
            current=[w]
            current_top=w["top"]
    lines.append(current)
    return lines

def extract_lines_with_style(crop):
    words=crop.extract_words(extra_attrs=["fontname","size"])
    if not words:
        return []
    line_groups=group_words_into_lines(words)
    lines=[]
    for group in line_groups:
        group_sorted=sorted(group,key=lambda w:w["x0"])
        text=" ".join(w["text"] for w in group_sorted)
        avg_size=sum(w["size"] for w in group_sorted)/len(group_sorted)
        is_bold=any("bold" in w["fontname"].lower() and "semibold" not in w["fontname"].lower()
                    for w in group_sorted)
        lines.append({"text":text,"size":avg_size,"bold":is_bold})
    return lines


def body_font_size(lines):
    if not lines:
        return 10.0
    sizes=[round(l["size"]) for l in lines]
    return Counter(sizes).most_common(1)[0][0]

def is_header_line(line,body_size,size_margin=1.0,max_len=45):
    text=line["text"].strip()
    if not text or len(text)>max_len:
        return False
    if text.startswith("-") or text.startswith("*") or text.startswith("•"):
        return False
    if text[-1] in ".,;:":
        return False
    if "," in text:                
        return False
    if looks_like_date_range(text):
        return False
    if looks_like_address_fragment(text):
        return False
    is_larger=line["size"]>=body_size+size_margin
    font_signal=is_larger or line["bold"]
    words=text.split()
    is_all_caps=text.upper()==text and any (c.isalpha() for c in text)

    def word_is_titled(w):
        core=re.sub(r"[^A-Za-z]","",w)
        return (not core) or core[0].isupper()
    is_multiword_title=len(words)>=2 and all(word_is_titled(w) for w in words)
    structural_signal=is_all_caps or is_multiword_title 
    return font_signal or structural_signal

def process_crop(crop):
    lines=extract_lines_with_style(crop)
    if not lines:
        return ""
    body_size=body_font_size(lines)
    out=[]
    for line in lines:
        text=line["text"].strip()
        if not text:
            continue
        if is_header_line(line,body_size):
            out.append(f"{HEADER_MARKER}{text}")
        else:
            out.append(text)
    return "\n".join(out)

def extract_page_column_aware(page):
    gap=find_column_gap(page)
    if gap is None:
        return process_crop(page)

    gap_start,gap_end=gap
    split_x=(gap_start+gap_end)/2
    left_crop=page.crop((0,0,split_x,page.height))
    right_crop=page.crop((split_x,0,page.width,page.height))
    left_text=process_crop(left_crop)
    right_text=process_crop(right_crop)

    left_width=split_x
    right_width=page.width-split_x
    if left_width>=right_width:
        return left_text+"\n"+right_text
    return right_text+"\n"+left_text

def extract_text(pdf_path):
    """Extracts text from a resume PDF.Handles both single and two column layouts,and tags are real section headers via font size/boldness comaprison
       against the body text with HEADER_MARKER so rag.py's chunking logic doesn't need to guess headers from plain text vocabulary"""

    pages_text=[]
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(extract_page_column_aware(page))
    return "\n".join(pages_text)