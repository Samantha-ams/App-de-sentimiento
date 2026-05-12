"""
FilingLens AI  Backend V5 CLEAN
=================================
Bugs fixed vs original V5:

  [CRITICAL-1] SYSTEM_PROMPT defined twice  removed first definition
  [CRITICAL-2] build_prompt() defined twice (1800 vs 1200 char cap)  kept 1200, single def
  [CRITICAL-3] parse_output() defined twice  merged into one correct definition
  [CRITICAL-4] First parse_output had orphaned dead code: `raw_lower = raw.lower()`
               inside an if-block with no body, causing IndentationError at runtime
  [CRITICAL-5] CHUNK_SIZE=140 words  sentence fragments, not paragraphs.
               Fixed to 800 words (fits Qwen2-0.5B context safely)

  [HIGH-1] fetch_edgar_filing fetched the raw SGML .txt container file and passed
           is_html=False, so the XML/SGML wrapper was never stripped.
           Fixed: fetch the .htm filing index viewer instead, pass is_html=True
  [HIGH-2] Cutoff regex matched "SIGNATURES" anywhere mid-sentence.
           Fixed: re.MULTILINE + line-start anchor
  [HIGH-3] /api/upload route was missing entirely  frontend couldn't upload files

  [MEDIUM-1] detect_rule_based_sentiment threshold was 2 negative keyword hits.
             "risk" alone appears 50+ times in every 10-K Risk Factors section,
             so virtually every chunk classified as Negative regardless of content.
             Fixed: raised threshold to 4 hits AND required hits from 2 distinct
             keyword categories (legal, financial, operational) to avoid single-word bias
  [MEDIUM-2] aggregate() flagged Negative at negative_ratio>=0.25.
             Combined with the rule-based bias above, every filing  Negative.
             Fixed to 0.40 threshold with majority-vote fallback

  [LOW-1]  EDGAR .txt URL pattern (accession_raw.txt) returns 404 on most filings.
           Fixed: use the -index.htm approach to locate the actual primary document
"""

import os
import re
import time
import logging
import requests
import threading

from pathlib import Path
from collections import Counter

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup

# 
# Logging
# 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
)
logger = logging.getLogger("FilingLens")

# 
# Flask Config
# 

BASE_DIR      = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "html", "htm"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB
app.secret_key = os.urandom(32)

# 
# Pipeline Config
# 

# FIX [CRITICAL-5]: was 140  produced sentence fragments.
# 800 words fills ~75% of Qwen2-0.5B's effective context window safely.
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 40
MAX_CHUNKS    = 25

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# 
# Model Singleton
# 

_model      = None
_tokenizer  = None
_model_lock = threading.Lock()


def get_model():
    """Thread-safe lazy loader  torch is imported only when first needed."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _model_lock:
        if _model is not None:
            return _model, _tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading model: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        _model.eval()
        logger.info("Model ready.")
    return _model, _tokenizer


# 
# Financial Relevance Filter
# 

_FINANCIAL_SIGNALS: frozenset = frozenset([
    "revenue", "revenues", "income", "loss", "profit", "earnings",
    "growth", "decline", "increase", "decrease",
    "risk", "risks", "competition", "competitive",
    "litigation", "lawsuit", "claim", "market share",
    "uncertainty", "guidance", "forecast",
    "cash flow", "liquidity", "debt", "margin", "impairment",
    "inflation", "interest rate", "supply chain", "demand",
    "customer", "contract", "regulation", "compliance",
    "penalty", "settlement", "cybersecurity", "privacy",
    "antitrust", "artificial intelligence",
])


def is_financially_relevant(text: str) -> bool:
    """Return True if the chunk contains at least one financial narrative signal."""
    text_lower = text.lower()
    return any(signal in text_lower for signal in _FINANCIAL_SIGNALS)


# 
# Rule-Based Sentiment (pre-LLM shortcut)
# 

# Keywords organized into semantic CATEGORIES.
# FIX [MEDIUM-1]: threshold raised from 2  4 hits, AND must span 2 categories.
# This prevents "risk" alone (which appears constantly in Risk Factors boilerplate)
# from classifying every chunk as Negative.

_NEGATIVE_CATEGORIES: dict[str, list[str]] = {
    "legal":       ["litigation", "lawsuit", "antitrust", "penalty", "investigation",
                    "settlement", "regulatory action", "product liability", "recall"],
    "financial":   ["net loss", "operating loss", "impairment", "write-down", "write-off",
                    "declining revenue", "margin compression", "material adverse"],
    "operational": ["supply chain disruption", "cybersecurity breach", "security incident",
                    "privacy violation", "product defect", "manufacturing defect"],
    "macro":       ["inflation headwind", "interest rate risk", "foreign exchange risk",
                    "tariff", "geopolitical", "recession"],
}

_POSITIVE_CATEGORIES: dict[str, list[str]] = {
    "growth":     ["strong revenue growth", "revenue growth", "revenue expansion",
                   "record revenue", "double-digit growth"],
    "margins":    ["improving margins", "margin expansion", "gross margin improvement",
                   "operating leverage"],
    "position":   ["competitive advantage", "market leadership", "pricing power",
                   "strong demand", "market share gain"],
    "guidance":   ["raised guidance", "increased guidance", "raised outlook",
                   "profitability improvement"],
}


def _count_category_hits(text_lower: str, categories: dict) -> tuple[int, int]:
    """Return (total keyword hits, number of distinct categories hit)."""
    total_hits     = 0
    categories_hit = 0
    for kws in categories.values():
        hits = sum(1 for kw in kws if kw in text_lower)
        if hits:
            total_hits     += hits
            categories_hit += 1
    return total_hits, categories_hit


def detect_rule_based_sentiment(chunk: str) -> dict | None:
    """
    Fast rule-based pre-classifier.
    Returns a result dict if confident, None to fall through to LLM.

    FIX [MEDIUM-1]: Requires total_hits >= 4 AND categories_hit >= 2 to avoid
    single-word "risk" flooding everything with Negative labels.
    """
    lower = chunk.lower()

    neg_hits, neg_cats = _count_category_hits(lower, _NEGATIVE_CATEGORIES)
    pos_hits, pos_cats = _count_category_hits(lower, _POSITIVE_CATEGORIES)

    # Negative wins if it clearly dominates
    if neg_hits >= 4 and neg_cats >= 2:
        return {
            "sentiment":   "Negative",
            "explanation": "Material operational, legal, or financial risks identified.",
            "summary":     "Risk exposure confirmed across multiple categories.",
            "raw":         "RULE_BASED_NEGATIVE",
        }

    # Positive only if it clearly dominates AND no significant negative signal
    if pos_hits >= 3 and pos_cats >= 2 and neg_hits < 2:
        return {
            "sentiment":   "Positive",
            "explanation": "Strong business performance or competitive positioning.",
            "summary":     "Positive operational momentum confirmed.",
            "raw":         "RULE_BASED_POSITIVE",
        }

    # Ambiguous  let the LLM decide
    return None


# 
# Prompt  (single definition  FIX [CRITICAL-1,2])
# 

SYSTEM_PROMPT = (
    "You are a senior Wall Street risk analyst. Task: Detailed Business Risk Analysis.\n\n"
    "INSTRUCTIONS:\n"
    "- Provide a DEEP technical analysis of the excerpt.\n"
    "- DO NOT use filler phrases like 'The text mentions' or 'The SEC filing says'.\n"
    "- Focus on specific triggers: legal liabilities, supply chain bottlenecks, "
    "macroeconomic headwinds, or competitive pressures.\n"
    "- If POSITIVE: describe specific growth drivers and margin improvements.\n\n"
    "FORMAT:\n"
    "Sentiment: [Positive|Neutral|Negative]\n"
    "Explanation: [A detailed professional paragraph, at least 40 words, focusing on impact.]\n"
    "Summary: [A concise financial headline.]"
)


def build_prompt(chunk: str) -> str:
    # FIX [CRITICAL-2]: was defined twice with caps of 1800 and 1200.
    # Single definition, using 1200 chars to stay within 512 token input budget.
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"SEC EXCERPT:\n"
        f"{chunk[:1200]}\n\n"
        f"Analysis:"
    )


# 
# Output Parser  (single definition  FIX [CRITICAL-3,4])
# 

import re

def parse_output(raw: str) -> dict:
    """
    Versin Final Profesional: 
    1. Captura bloques multilnea.
    2. Elimina muletillas de IA ("The SEC filing states...").
    3. Rompe el efecto 'eco' saltando palabras iniciales en el resumen.
    """
    result = {
        "sentiment":   "Neutral",
        "explanation": "",
        "summary":     "",
        "raw":         raw,
    }

    if not raw:
        return result

    # 1. Extraer Sentimiento
    sent_match = re.search(r"Sentiment:\s*(\w+)", raw, re.I)
    if sent_match:
        val = sent_match.group(1).lower()
        if "negative" in val: result["sentiment"] = "Negative"
        elif "positive" in val: result["sentiment"] = "Positive"
        else: result["sentiment"] = "Neutral"

    # 2. Extraer Explicacin (Captura prrafos largos)
    exp_match = re.search(r"Explanation:\s*(.*?)(?=\s*Summary:|$)", raw, re.I | re.S)
    if exp_match:
        result["explanation"] = exp_match.group(1).strip().replace("\n", " ")

    # 3. Extraer Resumen (Key Insight)
    sum_match = re.search(r"Summary:\s*(.*)", raw, re.I | re.S)
    if sum_match:
        result["summary"] = sum_match.group(1).strip().replace("\n", " ")

    # --- LGICA DE PULIDO PARA INTERFAZ PROFESIONAL ---

    # A. Fallback para Explicacin
    if not result["explanation"]:
        fallback = re.sub(r"(sentiment|explanation|summary):", "", raw, flags=re.I).strip()
        result["explanation"] = fallback if fallback else "Detailed analysis pending..."

    # B. MEJORA RADICAL ANTI-ECO (Para el Key Insight)
    # Si el resumen es un espejo de la explicacin, forzamos un cambio de ngulo.
    if not result["summary"] or len(result["summary"]) < 15:
        source_text = result["explanation"] if result["explanation"] else raw
        
        # Limpieza de muletillas (The SEC filing says that...)
        clean = re.sub(r"^(the sec filing|this document|the text|according to|the excerpt|that).*?\s(states|shows|mentions|is|contains|was|that)\s", "", source_text, flags=re.I)
        clean = re.sub(r"^(that|which|is|was)\s+", "", clean, flags=re.I).strip()

        words = clean.split()
        if len(words) > 10:
            # ESTRATEGIA: Saltamos las primeras 2 palabras del texto limpio 
            # para que el titular empiece con 'Stock price...' en lugar de 'Tesla's stock...'
            # Esto rompe la repeticin visual en la interfaz.
            headline_words = words[2:9] if len(words) > 5 else words[:7]
            result["summary"] = " ".join(headline_words).capitalize()
        elif len(words) > 0:
            result["summary"] = " ".join(words[:7]).capitalize()
        else:
            result["summary"] = "Key Financial Risk Identified"

    # C. Normalizacin de Puntuacin
    for field in ["explanation", "summary"]:
        text = result[field]
        if text and len(text) > 5:
            text = text.rstrip(". ")
            if field == "explanation":
                result[field] = text + "." # Prrafo termina en punto
            else:
                result[field] = text + "..." # Titular termina en elipsis

    return result


# 
# LLM Inference
# 

def run_inference(chunk: str, model, tokenizer) -> dict:
    """
    Run rule-based shortcut first; fall through to LLM only when ambiguous.
    Localized torch import avoids CUDA init on Flask cold start.
    """
    import torch

    # Fast path  no LLM call needed
    rule_result = detect_rule_based_sentiment(chunk)
    if rule_result is not None:
        logger.info(f"RULE-BASED  {rule_result['sentiment']}")
        return rule_result

    # LLM path
    inputs = tokenizer(
        build_prompt(chunk),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.15,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    logger.info(f"MODEL OUTPUT:\n{raw}\n")
    return parse_output(raw)


# 
# Chunking
# 

def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# 
# Aggregation
# 

def aggregate(chunk_results: list[dict]) -> dict:
    """
    Majority-vote sentiment with calibrated thresholds.

    FIX [MEDIUM-2]: was negative_ratio>=0.25  too aggressive with the rule-based
    bias. Raised to 0.40. Also added majority-vote fallback so the label always
    reflects what the model actually found rather than a fixed threshold.
    """
    if not chunk_results:
        return {
            "overall_sentiment": "Neutral",
            "distribution":      {"Positive": 0.0, "Neutral": 100.0, "Negative": 0.0},
            "chunk_count":       0,
            "explanation":       "No relevant chunks found.",
            "summary":           "No signal.",
        }

    sentiments = [r["sentiment"] for r in chunk_results]
    counts     = Counter(sentiments)
    total      = len(sentiments)

    neg_ratio = counts["Negative"] / total
    pos_ratio = counts["Positive"] / total

    if neg_ratio >= 0.40:
        overall = "Negative"
    elif pos_ratio >= 0.50:
        overall = "Positive"
    else:
        # Fall back to strict majority vote; tie-break: Negative > Neutral > Positive
        overall = counts.most_common(1)[0][0]
        if counts.most_common(1)[0][1] == counts.most_common(2)[-1][1]:
            for label in ("Negative", "Neutral", "Positive"):
                if counts[label] == counts.most_common(1)[0][1]:
                    overall = label
                    break

    distribution = {
        k: round(counts.get(k, 0) / total * 100, 1)
        for k in ("Positive", "Neutral", "Negative")
    }

    matching = [r for r in chunk_results if r["sentiment"] == overall]
    best     = matching[0] if matching else chunk_results[0]

    return {
        "overall_sentiment": overall,
        "distribution":      distribution,
        "chunk_count":       total,
        "explanation":       best["explanation"],
        "summary":           best["summary"],
        "chunk_results":     chunk_results,
    }


# 
# SEC Section Targeting
# 

_TARGET_SECTION_RE = re.compile(
    r"item\s+1a[\s\S]{0,120}?risk\s+factors"
    r"|item\s+7[\s\S]{0,120}?management[\s\S]{0,60}?discussion"
    r"|item\s+7\b",
    re.IGNORECASE,
)

# FIX [HIGH-2]: Added re.MULTILINE so ^ anchors to start of each line,
# preventing mid-sentence matches from stopping extraction too early.
_STOP_SECTION_RE = re.compile(
    r"^\s*item\s+(1b|2|3|4|5|6|7a|8|9|10|11|12|13|14|15)\b",
    re.IGNORECASE | re.MULTILINE,
)


def extract_target_sections(text: str) -> str:
    """
    Extract Item 1A (Risk Factors) and Item 7 (MD&A).
    Falls back to full text if no section headers are found.
    Returns full text if extracted content is under 200 words (safety net).
    """
    matches = list(_TARGET_SECTION_RE.finditer(text))

    if not matches:
        logger.warning("Target sections not found  using full filing text.")
        return text

    parts: list[str] = []
    for match in matches:
        stop = _STOP_SECTION_RE.search(text, match.end())
        end  = stop.start() if stop else len(text)
        section = text[match.start():end].strip()
        if len(section) > 200:
            parts.append(section)
            logger.info(f"Captured: '{match.group()[:60].strip()}'")

    if not parts:
        logger.warning("Section headers matched but extracted no content  using full text.")
        return text

    combined = "\n\n".join(parts)
    if len(combined.split()) < 200:
        logger.warning(f"Extraction yielded only {len(combined.split())} words  using full text.")
        return text

    return combined


# 
# Text Cleaning
# 

# FIX [HIGH-2]: require cutoff markers to be at the start of a line.
_CUTOFF_RE = re.compile(
    r"^\s*(SIGNATURES|EXHIBIT\s+INDEX|PART\s+IV|POWER\s+OF\s+ATTORNEY)",
    re.IGNORECASE | re.MULTILINE,
)


def clean_financial_content(raw: str, is_html: bool = True) -> str:
    """
    Full cleaning pipeline.

    FIX [HIGH-1]: original code fetched the raw SGML .txt container and passed
    is_html=False, leaving XML/SGML markup in the text stream.
    This function now always receives HTML (is_html=True) because fetch_edgar_filing
    was fixed to retrieve the .htm viewer document instead of the .txt container.

    The is_html=False branch below handles plain-text .txt/.sgml uploads from disk.
    """
    if not is_html:
        # Plain-text SGML container (uploaded manually, not fetched from EDGAR)
        docs = re.findall(
            r"<DOCUMENT>(.*?)</DOCUMENT>",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )
        best = ""
        for doc in docs:
            type_m = re.search(r"<TYPE>(.*?)\n", doc, re.IGNORECASE)
            doc_type = type_m.group(1).strip().upper() if type_m else ""
            if doc_type in {"10-K", "10-Q", "8-K"}:
                text_m = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL | re.IGNORECASE)
                if text_m and len(text_m.group(1)) > len(best):
                    best = text_m.group(1)
        if best:
            raw = best

    if is_html or "<html" in raw.lower():
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "meta", "link", "noscript", "head"]):
            tag.decompose()
        # Unwrap inline XBRL tags  keeps their text content
        for tag in soup.find_all(
            lambda t: t.name and (
                t.name.startswith("ix:") or t.name.startswith("xbrli:")
            )
        ):
            tag.unwrap()
        # Tables contain numeric data, not sentiment narrative
        for table in soup.find_all("table"):
            table.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = raw

    # Strip any residual HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()

    logger.info(f"[After HTML clean]   {len(text.split()):,} words")

    # Truncate at end-of-document boilerplate (line-anchored  FIX [HIGH-2])
    cutoff = _CUTOFF_RE.search(text)
    if cutoff:
        text = text[: cutoff.start()].strip()
        logger.info(f"[After cutoff]       {len(text.split()):,} words ('{cutoff.group().strip()}')")
    else:
        logger.info(f"[After cutoff]       {len(text.split()):,} words (no cutoff found)")

    # Target high-signal sections
    text = extract_target_sections(text)
    logger.info(f"[After sections]     {len(text.split()):,} words (final)")

    return text.strip()


# 
# SEC EDGAR Fetcher
# 

EDGAR_HEADERS = {
    "User-Agent":      "FilingLens AI contact@filinglens.ai",
    "Accept-Encoding": "gzip, deflate",
}


def fetch_edgar_filing(ticker: str, filing_type: str) -> tuple[str, str]:
    """
    Fetch the latest filing for a ticker from SEC EDGAR.
    Returns (raw_html, filing_url).

    FIX [LOW-1]: original tried to construct accession_raw.txt URL which 404s on
    most modern filings. Fixed to use the -index.htm approach to find the actual
    primary document (same reliable strategy as V3.1).
    """
    # Step 1: Resolve ticker  CIK
    atom = requests.get(
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?company=&CIK={ticker}&type={filing_type}"
        f"&dateb=&owner=include&count=10"
        f"&search_text=&action=getcompany&output=atom",
        headers=EDGAR_HEADERS,
        timeout=15,
    )
    atom.raise_for_status()

    soup    = BeautifulSoup(atom.text, "xml")
    cik_tag = soup.find("cik")

    if not cik_tag:
        raise ValueError(f"Could not resolve CIK for ticker '{ticker}'.")

    cik = cik_tag.text.strip().zfill(10)
    logger.info(f"Resolved CIK: {cik}")

    # Step 2: Fetch submission history
    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=EDGAR_HEADERS,
        timeout=20,
    )
    sub.raise_for_status()
    recent   = sub.json()["filings"]["recent"]
    forms    = recent["form"]
    acc_nums = recent["accessionNumber"]

    idx = next((i for i, f in enumerate(forms) if f == filing_type), None)
    if idx is None:
        raise ValueError(f"No {filing_type} filing found for {ticker}.")

    acc_raw = acc_nums[idx]
    acc     = acc_raw.replace("-", "")
    cik_int = int(cik)

    # Step 3: Fetch filing index and locate primary document
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{acc_raw}-index.htm"
    )
    logger.info(f"Fetching index: {index_url}")

    index_r = requests.get(index_url, headers=EDGAR_HEADERS, timeout=20)
    index_r.raise_for_status()
    idx_soup = BeautifulSoup(index_r.text, "html.parser")

    doc_link = None

    # Primary: find the row where the type cell matches the filing type
    for row in idx_soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4 and filing_type in cells[3].get_text():
            a = cells[2].find("a", href=True)
            if a:
                doc_link = "https://www.sec.gov" + a["href"]
                break

    # Fallback: grab the first .htm link in the index
    if not doc_link:
        for a in idx_soup.find_all("a", href=True):
            href = a["href"].lower()
            if href.endswith((".htm", ".html")) and "index" not in href:
                doc_link = "https://www.sec.gov" + a["href"]
                break

    if not doc_link:
        raise ValueError("Could not locate primary document in EDGAR filing index.")

    # 
    # FIX [CRITICAL]: Si la URL apunta al visor /ix?doc=, extraemos la URL real.
    # El visor IXBRL es JS-driven y BeautifulSoup lo ve como "0 words".
    # 
    if "/ix?doc=" in doc_link:
        raw_path = doc_link.split("/ix?doc=")[1]
        doc_link = f"https://www.sec.gov{raw_path}"
        logger.info(f"iXBRL Viewer detected. Redirecting to raw path: {doc_link}")

    logger.info(f"Fetching document: {doc_link}")

    # Step 4: Download and return the HTML
    doc_r = requests.get(doc_link, headers=EDGAR_HEADERS, timeout=30)
    doc_r.raise_for_status()
    return doc_r.text, doc_link


# 
# Helpers
# 

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# 
# Routes
# 

@app.route("/")
def index():
    return render_template("index.html")


# FIX [HIGH-3]: /api/upload was completely missing from V5.
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file in request."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported type. Use .txt / .html / .htm"}), 400
    stem, ext = os.path.splitext(secure_filename(file.filename))
    filename  = f"{stem}_{int(time.time())}{ext}"
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    logger.info(f"Uploaded: {filename}")
    return jsonify({"filename": filename, "message": "File uploaded successfully."})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    t_start = time.time()
    data    = request.get_json(force=True)

    ticker      = data.get("ticker",      "").strip().upper()
    filing_type = data.get("filing_type", "10-K").strip()
    filename    = data.get("filename",    "").strip()
    section     = data.get("section",     "").strip()

    filing_url = None

    #  Ingest 
    try:
        if ticker:
            raw, filing_url = fetch_edgar_filing(ticker, filing_type)
            text   = clean_financial_content(raw, is_html=True)
            source = f"{ticker} {filing_type}"

        elif filename:
            path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
            if not os.path.exists(path):
                return jsonify({"error": "File not found  please re-upload."}), 404
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            is_html = Path(path).suffix.lower() in (".html", ".htm")
            text    = clean_financial_content(raw, is_html=is_html)
            source  = filename

        else:
            return jsonify({"error": "Provide ticker or filename."}), 400

    except (ValueError, requests.RequestException) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Ingest failed")
        return jsonify({"error": f"Ingest error: {e}"}), 500

    #  Optional manual section targeting 
    if section:
        idx = text.lower().find(section.lower())
        if idx == -1:
            return jsonify({"error": f"Section '{section}' not found."}), 400
        text = text[max(0, idx - 100): idx + 10_000]

    #  Minimum content guard 
    word_count = len(text.split())
    if word_count < 30:
        return jsonify({
            "error": (
                f"Extracted text too short ({word_count} words). "
                "Check server logs for [After HTML clean] / [After sections] "
                "word counts to identify which pipeline stage lost content."
            )
        }), 400

    text_preview = text[:800]

    #  Chunk  filter  infer 
    all_chunks = chunk_text(text)
    logger.info(f"Chunks: {len(all_chunks)} total  |  {word_count:,} words")

    try:
        model, tokenizer = get_model()
    except Exception as e:
        logger.exception("Model load failed")
        return jsonify({"error": f"Model failed to load: {e}"}), 500

    chunk_results: list[dict] = []
    rejected = 0

    for i, chunk in enumerate(all_chunks):

        if len(chunk_results) >= MAX_CHUNKS:
            logger.info(f"Reached {MAX_CHUNKS}-chunk cap at chunk {i}. Stopping.")
            break

        if not is_financially_relevant(chunk):
            rejected += 1
            continue

        logger.info(f"Chunk {i:03d}  {len(chunk.split())} words")

        try:
            result = run_inference(chunk, model, tokenizer)
        except Exception as e:
            logger.error(f"Chunk {i} error: {e}")
            result = {
                "sentiment": "Neutral", "explanation": "Inference error.",
                "summary": "Error.", "raw": "",
            }

        result["chunk_index"] = i
        result["word_count"]  = len(chunk.split())
        chunk_results.append(result)

    elapsed = round(time.time() - t_start, 1)
    logger.info(
        f"Done  {len(chunk_results)} analyzed, {rejected} rejected, {elapsed}s"
    )

    return jsonify({
        "source":       source,
        "ticker":       ticker or "N/A",
        "filing_type":  filing_type if ticker else "Uploaded File",
        "filing_url":   filing_url,
        "text_preview": text_preview,
        "word_count":   word_count,
        "elapsed_sec":  elapsed,
        **aggregate(chunk_results),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File exceeds 50 MB limit."}), 413


# 
# Entry Point
# 

if __name__ == "__main__":
    logger.info("FilingLens AI V5 CLEAN")
    # Using 0.0.0.0 makes it accessible on the local network
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)