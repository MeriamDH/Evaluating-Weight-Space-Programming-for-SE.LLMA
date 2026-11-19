import re
from collections import Counter
from typing import Any, Dict, List, Optional

from langdetect import DetectorFactory, detect_langs

# Ensure deterministic language detection
DetectorFactory.seed = 42

SWEDISH_STOPWORDS = {
    "och", "att", "det", "som", "en", "på", "är", "av", "för", "med",
    "till", "den", "har", "de", "inte", "om", "ett", "men", "var", "jag",
    "från", "kan", "man", "eller", "så", "vid", "ska", "också", "efter", "när",
}

ENGLISH_STOPWORDS = {
    "the", "and", "is", "to", "of", "in", "that", "it", "for", "on",
    "with", "as", "are", "this", "be", "or", "from", "by", "not", "an",
    "can", "will", "have", "which", "their", "more", "about", "into",
}


def detect_swedish_probability(text: str) -> float:
    cleaned = text.strip()
    if len(cleaned) < 12:
        return 0.0
    try:
        langs = detect_langs(cleaned)
        for item in langs:
            if item.lang == "sv":
                return float(item.prob)
        return 0.0
    except Exception:
        return 0.0


def compute_lexical_language_scores(text: str) -> Dict[str, float]:
    tokens = re.findall(r"[a-zåäöé]+", text.lower())
    if not tokens:
        return {"sv_stopword_rate": 0.0, "en_stopword_rate": 0.0, "swedish_char_rate": 0.0}
    sv_hits = sum(1 for t in tokens if t in SWEDISH_STOPWORDS)
    en_hits = sum(1 for t in tokens if t in ENGLISH_STOPWORDS)
    sv_chars = sum(1 for ch in text.lower() if ch in {"å", "ä", "ö"})
    return {
        "sv_stopword_rate": sv_hits / len(tokens),
        "en_stopword_rate": en_hits / len(tokens),
        "swedish_char_rate": sv_chars / max(len(text), 1),
    }


def compute_repetition_metrics(text: str, n: int = 4) -> Dict[str, float]:
    """Detect degenerate n-gram repetition loops (a central failure mode in Weight-Space Programming)."""
    tokens = text.split()
    if len(tokens) < n:
        return {"distinct_n": 1.0, "max_ngram_repeat": 1, "loop_detected": 0.0}
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    distinct_ratio = len(counts) / len(ngrams)
    max_repeat = max(counts.values())
    loop_detected = 1.0 if (distinct_ratio < 0.55 or max_repeat >= 4) else 0.0
    return {
        "distinct_n": float(distinct_ratio),
        "max_ngram_repeat": int(max_repeat),
        "loop_detected": float(loop_detected),
    }


def score_response(
    response_text: str,
    expected_lang: str = "sv",
    reference_keywords: Optional[List[str]] = None,
    max_new_tokens: int = 160,
    generated_token_count: int = 0,
) -> Dict[str, Any]:
    sv_prob = detect_swedish_probability(response_text)
    lex = compute_lexical_language_scores(response_text)
    rep = compute_repetition_metrics(response_text)

    # Swedish adherence: strong langdetect confidence OR clear Swedish lexical dominance without English drift
    is_swedish = (
        sv_prob >= 0.70
        or (lex["sv_stopword_rate"] > lex["en_stopword_rate"] * 1.5 and lex["sv_stopword_rate"] >= 0.08)
    )
    code_switched = (
        expected_lang == "sv"
        and (lex["en_stopword_rate"] >= 0.08 or (sv_prob < 0.50 and len(response_text.strip()) >= 20))
    )

    keyword_hit = None
    if reference_keywords:
        lower_resp = response_text.lower()
        keyword_hit = float(any(kw.lower() in lower_resp for kw in reference_keywords))

    hit_token_limit = float(generated_token_count >= max_new_tokens - 1)

    return {
        "sv_prob": sv_prob,
        "is_swedish": float(is_swedish),
        "code_switched": float(code_switched),
        "sv_stopword_rate": lex["sv_stopword_rate"],
        "en_stopword_rate": lex["en_stopword_rate"],
        "swedish_char_rate": lex["swedish_char_rate"],
        "distinct_4gram": rep["distinct_n"],
        "max_4gram_repeat": rep["max_ngram_repeat"],
        "loop_detected": rep["loop_detected"],
        "hit_token_limit": hit_token_limit,
        "keyword_hit": keyword_hit,
        "char_length": len(response_text.strip()),
    }


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = [
        "sv_prob",
        "is_swedish",
        "code_switched",
        "sv_stopword_rate",
        "en_stopword_rate",
        "distinct_4gram",
        "loop_detected",
        "hit_token_limit",
        "char_length",
    ]
    summary = {}
    for k in keys:
        vals = [r["metrics"][k] for r in records if r["metrics"].get(k) is not None]
        summary[f"mean_{k}"] = float(sum(vals) / len(vals)) if vals else 0.0

    kw_vals = [r["metrics"]["keyword_hit"] for r in records if r["metrics"].get("keyword_hit") is not None]
    if kw_vals:
        summary["keyword_accuracy"] = float(sum(kw_vals) / len(kw_vals))
    summary["n_examples"] = len(records)
    return summary
