import re
import numpy as np
import pandas as pd
from collections import Counter

'''
'''


# -----------------------------
# 1. Basic helpers
# -----------------------------

STOPWORDS = [
    "the","a","an","and","or","but","if","then","so","for","of","on","in","to",
    "with","without","by","at","from","as","is","are","was","were","be","been",
    "being","that","this","these","those","it","its","into","about","than",
    "such","which","who","whom","whose","what","when","where","why","how"
]

QUESTION_WORDS = {
    "what", "when", "where", "why", "how", "who", "which",
    "has", "have", "is", "are", "was", "were",
    "does", "do", "did", "can", "could", "should", "would",
}

def tokenize(text):
    return re.findall(r"\w+|\d+|\S", text.lower())

def content_tokens(text):
    toks = tokenize(text)
    return [t for t in toks if t.isalpha() and t not in STOPWORDS]

def jaccard_overlap(a, b):
    A = set(content_tokens(a))
    B = set(content_tokens(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def has_number(text):
    return bool(re.search(r"\d", text))

def count_numbers(text):
    return len(re.findall(r"\d+(?:\.\d+)?", text))

MONTH_PAT = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
DATE_PAT = rf"(\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{1,2}}-\d{{1,2}}|{MONTH_PAT})"

def count_dates(text):
    return len(re.findall(DATE_PAT, text.lower()))

def shannon_entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    return -sum(p*np.log(p + 1e-12) for p in probs)

def stopword_ratio(text):
    toks = tokenize(text)
    if not toks:
        return 0.0
    sw = sum(1 for t in toks if t in STOPWORDS)
    return sw / len(toks)

def classify_question_type(q):
    q_low = q.lower()
    if any(x in q_low for x in ["when", "date", "time", "year", "day", "month"]):
        return "time"
    if any(x in q_low for x in ["how many", "how much", "number", "count", "dose", "dosage", "mg", "mcg"]):
        return "number"
    return "other"



# -----------------------------
# 2. Rule implementations (per-row)
# -----------------------------

def rule1_length_anomaly(answer_text, length_threshold):
    """Gold span length is abnormally large."""
    return len(tokenize(answer_text)) > length_threshold

def rule2_multi_clause(answer_text):
    """Gold span looks like multiple clauses/list items."""
    clauses = re.split(r',|\n|;|\band\b|\bor\b', answer_text, flags=re.IGNORECASE)
    informative = [c for c in clauses if len(content_tokens(c)) >= 3]
    return len(informative) > 1

def rule3_low_question_similarity(answer_text, question, sim_threshold=0.05):
    """Gold span has very low lexical overlap with question."""
    overlap = jaccard_overlap(answer_text, question)
    return overlap < sim_threshold

def rule4_pred_inside_gold_better_alignment(pred, gold, question, margin=0.1):
    """
    Predicted span is inside gold span AND aligns better with the question.
    Strong hint gold is a noisy chunk.
    """
    gold_clean = gold or ""
    pred_clean = pred or ""
    if not pred_clean or pred_clean.strip() not in gold_clean:
        return False
    q_gold = jaccard_overlap(gold_clean, question)
    q_pred = jaccard_overlap(pred_clean, question)
    return q_pred > q_gold + margin

def rule5_question_type_mismatch(answer_text, question):
    """
    Question type vs answer structure mismatch:
    - time question but zero or multiple date-like expressions
    - number question but zero or many numeric expressions
    """
    qtype = classify_question_type(question)
    if qtype == "time":
        n_dates = count_dates(answer_text)
        return n_dates == 0 or n_dates > 1
    if qtype == "number":
        n_nums = count_numbers(answer_text)
        return n_nums == 0 or n_nums > 3
    return False

def rule6_multiple_occurrences_in_context(answer_text, context):
    """Gold span appears multiple times in the context: ambiguous / alignment suspect."""
    if not answer_text:
        return False
    return context.count(answer_text) > 1

def rule7_boundary_weirdness(answer_text):
    """Gold span starts/ends with separators or looks cut mid-clause."""
    txt = answer_text.strip()
    if not txt:
        return False
    if txt.startswith((",", ";", "and", "or")):
        return True
    if txt.endswith((",", ";", "and", "or")):
        return True
    if txt[0] in "/-:" or txt[-1] in "/-:":
        return True
    return False

def rule8_pred_answers_question_better(pred, gold, question, margin=0.1):
    """Pred span has much better question alignment than gold (even if not substring)."""
    q_gold = jaccard_overlap(gold or "", question)
    q_pred = jaccard_overlap(pred or "", question)
    return q_pred > q_gold + margin

def rule9_question_not_starting_with_qword(question):
    """Question doesn't start with a common question word - malformed question."""
    q = re.sub(r"\s+", " ", question)
    q = q.strip().lower()
    if not q:
        return True
    first = q.split()[0]
    return first not in QUESTION_WORDS


# -----------------------------
# 3. Apply rules to dataframe
# -----------------------------

def compute_dataset_error_flags(row, length_threshold):
    ctx = row.get("context", "") or ""
    q = row.get("question", "") or ""
    gold = row.get("answer", "") or ""
    pred = row.get("predicted_answer", "") or ""

    r1 = rule1_length_anomaly(gold, length_threshold)
    r2 = rule2_multi_clause(gold)
    r3 = rule3_low_question_similarity(gold, q)
    r4 = rule4_pred_inside_gold_better_alignment(pred, gold, q)
    r5 = rule5_question_type_mismatch(gold, q)
    r6 = rule6_multiple_occurrences_in_context(gold, ctx)
    r7 = rule7_boundary_weirdness(gold)
    r8 = rule8_pred_answers_question_better(pred, gold, q)
    r9 = rule9_question_not_starting_with_qword(q)

    flags = {
        "rule1_length_anomaly": r1,
        "rule2_multi_clause": r2,
        "rule3_low_q_similarity": r3,
        "rule4_pred_inside_gold_better": r4,
        "rule5_qtype_mismatch": r5,
        "rule6_multi_occurrences": r6,
        "rule7_boundary_weirdness": r7,
        "rule8_pred_better_q_alignment": r8,
        "rule9_question_not_starting_with_qword": r9
    }
    flags["dataset_error_score"] = sum(flags.values())
    flags["is_dataset_error"] = flags["dataset_error_score"] >= 3
    return pd.Series(flags)


def detect_dataset_errors(df, error_threshold=2, length_percentile=95):
    """
    Main function to detect dataset errors in a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns: 'context', 'question', 'answer', 'predicted_answer'
    error_threshold : int
        Minimum number of rules that must trigger to classify as dataset error
    length_percentile : float
        Percentile for length anomaly threshold (default: 95)
    
    Returns:
    --------
    pd.DataFrame : Original dataframe with added error flag columns
    pd.DataFrame : Summary statistics of rule triggers
    """
    # Check if 'answer' column exists, if not create it from 'answers' column
    if 'answer' not in df.columns:
        if 'answers' in df.columns:
            df['answer'] = [v['text'][0] for v in df["answers"].values]
        # else:
        #     df['answer'] = ""
    # Compute length threshold
    gold_lengths = df['answer'].fillna("").apply(lambda x: len(tokenize(x)))
    length_p95 = np.percentile(gold_lengths, length_percentile) if len(gold_lengths) > 0 else 100
    length_med = np.median(gold_lengths) if len(gold_lengths) > 0 else 0
    mad = np.median(np.abs(gold_lengths - length_med)) if len(gold_lengths) > 0 else 0
    length_threshold = max(length_p95, length_med + 2 * mad)
    
    # Apply rules
    error_flags = df.apply(lambda row: compute_dataset_error_flags(row, length_threshold), axis=1)
    df_with_flags = pd.concat([df, error_flags], axis=1)
    
    # Update threshold if different from default
    if error_threshold != 3:
        df_with_flags["is_dataset_error"] = df_with_flags["dataset_error_score"] >= error_threshold
    
    # Compute summary
    rule_cols = [c for c in df_with_flags.columns if c.startswith("rule")]
    summary = pd.DataFrame({
        "count": df_with_flags[rule_cols].sum(),
        "percent": df_with_flags[rule_cols].mean() * 100
    }).sort_values("count", ascending=False)
    
    return df_with_flags, summary