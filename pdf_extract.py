import re
import fitz

def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def extract_abstract(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = []
    for i in range(min(5, doc.page_count)):
        page = doc.load_page(i)
        text.append(page.get_text("text"))
    doc.close()
    full = "\n".join(text)
    s = re.sub(r"\u00ad", "", full)
    idx = None
    for pat in [r"\babstract\b", r"\bsummary\b"]:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            idx = m.start()
            break
    if idx is None:
        parts = re.split(r"\n\s*\n", s)
        if parts:
            cand = parts[0]
            return clean_text(cand)
        return ""
    tail = s[idx:]
    stop = re.search(r"\bkeywords\b|\bindex terms\b|\bintroduction\b|\n\s*1\s*[\.| ]|\nI\.|\n1\s+Introduction", tail, flags=re.IGNORECASE)
    if stop:
        seg = tail[:stop.start()]
    else:
        seg = tail[:2000]
    seg = re.sub(r"^\s*abstract\s*[:\.-]*\s*", "", seg, flags=re.IGNORECASE)
    return clean_text(seg)
