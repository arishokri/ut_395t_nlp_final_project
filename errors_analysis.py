import re
import numpy as np
import pandas as pd
from collections import Counter

"""
"""


# -----------------------------
# 1. Basic helpers
# -----------------------------

STOPWORDS = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "for",
    "of",
    "on",
    "in",
    "to",
    "with",
    "without",
    "by",
    "at",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "into",
    "about",
    "than",
    "such",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "when",
    "where",
    "why",
    "how",
]

QUESTION_WORDS = {
    "what",
    "when",
    "where",
    "why",
    "how",
    "who",
    "which",
    "has",
    "have",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "would",
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
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log(p + 1e-12) for p in probs)


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
    if any(
        x in q_low
        for x in [
            "how many",
            "how much",
            "number",
            "count",
            "dose",
            "dosage",
            "mg",
            "mcg",
        ]
    ):
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
    clauses = re.split(r",|\n|;|\band\b|\bor\b", answer_text, flags=re.IGNORECASE)
    informative = [c for c in clauses if len(content_tokens(c)) >= 3]
    return len(informative) > 1


def rule3_low_question_similarity(answer_text, question, sim_threshold=0.05):
    """Gold span has very low lexical overlap with question."""
    overlap = jaccard_overlap(answer_text, question)
    return overlap < sim_threshold





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


def rule10_question_too_vague(question):
    """Question is too vague/incomplete to have a specific answer."""
    q_clean = question.strip().lower()
    
    # Pattern 1: Single word or very short questions
    if len(q_clean.split()) <= 2 and '?' not in q_clean:
        # Exceptions: proper wh-questions like "When?", "Why?"
        if q_clean not in ['when?', 'where?', 'why?', 'how?', 'who?']:
            return True
    
    # Pattern 2: "Previous [medication]" pattern (very common in medical QA)
    if re.match(r'^previous\s+\w+', q_clean):
        return True
    
    # Pattern 3: Just medication/treatment names without question words
    vague_patterns = [
        r'^meds?\.?$',
        r'^medications?\.?$',
        r'^meds\s+record\.?$',
        r'^history\.?$',
        r'^allergies?\.?$',
        r'^diagnos[ie]s\.?$',
        r'^treatments?\.?$',
        r'^procedures?\.?$',
        r'^current\s+meds?\.?$',
        r'^home\s+meds?\.?$',
        r'^discharge\s+meds?\.?$'
    ]
    
    return any(re.match(pattern, q_clean) for pattern in vague_patterns)


def rule11_medical_semantic_mismatch(answer, question):
    """Enhanced medical domain-specific semantic mismatch detection."""
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Extract potential medication name from question
    # Common patterns: "How often does patient take [medication]"
    # "Has the patient taken [drug]"
    med_name = None
    if 'take' in q_lower:
        parts = q_lower.split('take')
        if len(parts) > 1:
            # Get words after 'take', remove common words
            words = parts[1].strip().split()
            if words:
                # Get the medication name (could be multiple words)
                med_name = words[0].strip('?,.')
    elif 'of' in q_lower and any(kw in q_lower for kw in ['dose', 'dosage', 'frequency']):
        # Pattern: "What is the dose of [medication]"
        parts = q_lower.split('of')
        if len(parts) > 1:
            words = parts[-1].strip().split()
            if words:
                med_name = words[0].strip('?,.')
    
    # Medication frequency questions - check if answer discusses the right medication
    if any(phrase in q_lower for phrase in ['how often', 'frequency', 'times per']):
        freq_indicators = ['daily', 'twice', 'three times', 'q.d.', 'b.i.d.', 't.i.d.', 
                          'q.i.d.', 'every', 'per day', 'once', 'morning', 'evening', 'qd', 'bid', 'tid',
                          'each day', 'each morning', 'each evening', 'per hour']
        has_freq_info = any(indicator in a_lower for indicator in freq_indicators)
        
        # If no frequency info at all, it's a mismatch
        if not has_freq_info:
            return True
        
        # If we found a medication name in question, check if it's in the answer
        # Only flag if medication name is substantial enough and clearly absent
        if med_name and len(med_name) > 3:
            # Check for the medication name or close variants (without hyphens, etc.)
            med_clean = med_name.replace('-', ' ').strip()
            a_clean = a_lower.replace('-', ' ')
            
            if med_clean not in a_clean:
                return True  # Answer talks about different medication
    
    # Dosage questions with non-dosage answers
    if any(phrase in q_lower for phrase in ['dose', 'dosage', 'how much', 'amount']):
        dosage_indicators = ['mg', 'ml', 'tablet', 'pill', 'unit', 'mcg', 'gram', 'dose']
        has_dosage_info = any(indicator in a_lower for indicator in dosage_indicators)
        if not has_dosage_info:
            return True
        
        # Check if medication name matches
        if med_name and len(med_name) > 3:
            med_clean = med_name.replace('-', ' ').strip()
            a_clean = a_lower.replace('-', ' ')
            if med_clean not in a_clean:
                return True
    
    # Answer contains instructions rather than requested data
    instruction_patterns = [
        r'follow.?up',
        r'assistance with',
        r'continue.*as directed',
        r'treating with.*as necessary',
        r'apply to affected areas',
        r'take.*as directed',
        r'vna for'
    ]
    if any(re.search(pattern, a_lower) for pattern in instruction_patterns):
        return True
    
    return False


def rule12_model_underextraction(pred, gold, question, margin=0.1):
    """
    Model predicted a substring that's too short/incomplete AND not better aligned.
    This excludes cases where the shorter prediction is actually better aligned with the question
    (which would be a dataset annotation issue, not a model error).
    Reuses rule8 for alignment checking to avoid code duplication.
    """
    if not pred or not gold:
        return False
    
    pred_clean = pred.strip()
    gold_clean = gold.strip()
    
    # Prediction must be a substring of gold (model stopped early)
    if pred_clean in gold_clean and pred_clean != gold_clean:
        pred_len = len(tokenize(pred_clean))
        gold_len = len(tokenize(gold_clean))
        
        # Model extracted less than 50% of gold answer length
        if pred_len < gold_len * 0.5:
            # Only flag as underextraction if pred is NOT significantly better aligned
            # Reuse rule8 - if it triggers, pred is better aligned (dataset issue, not model error)
            if not rule8_pred_answers_question_better(pred, gold, question, margin):
                return True
    
    return False


def rule13_model_wrong_entity_same_type(pred, gold, question):
    """Model extracted wrong entity but of same semantic type."""
    if not pred or not gold:
        return False
    
    pred_clean = pred.strip().lower()
    gold_clean = gold.strip().lower()
    
    # Must be different answers
    if pred_clean == gold_clean:
        return False
    
    # Medication confusion: both contain medication indicators
    med_indicators = ['mg', 'mcg', 'ml', 'tablet', 'pill', 'capsule', 'dose']
    pred_has_med = any(ind in pred_clean for ind in med_indicators)
    gold_has_med = any(ind in gold_clean for ind in med_indicators)
    
    if pred_has_med and gold_has_med:
        return True
    
    # Temporal confusion: both contain time/date indicators
    time_indicators = ['day', 'month', 'year', 'date', 'time', 'hour', 'week']
    pred_has_time = any(ind in pred_clean for ind in time_indicators)
    gold_has_time = any(ind in gold_clean for ind in time_indicators)
    
    if pred_has_time and gold_has_time:
        return True
    
    # Numeric confusion: both contain numbers but different values
    pred_has_num = has_number(pred_clean)
    gold_has_num = has_number(gold_clean)
    
    if pred_has_num and gold_has_num:
        pred_nums = set(re.findall(r'\d+(?:\.\d+)?', pred_clean))
        gold_nums = set(re.findall(r'\d+(?:\.\d+)?', gold_clean))
        if pred_nums != gold_nums:
            return True
    
    return False


def rule14_model_position_bias(pred, context, gold):
    """Model shows position bias - extracted answer from wrong location."""
    if not pred or not context or not gold:
        return False
    
    # Find positions of predicted and gold answers in context
    pred_pos = context.find(pred)
    gold_pos = context.find(gold)
    
    # Both must appear in context
    if pred_pos == -1 or gold_pos == -1:
        return False
    
    # Calculate relative positions (0 = start, 1 = end)
    ctx_len = len(context)
    pred_rel_pos = pred_pos / ctx_len
    gold_rel_pos = gold_pos / ctx_len
    
    # Model consistently picks earlier/later occurrence (position bias)
    # Flag if prediction is in significantly different position
    position_diff = abs(pred_rel_pos - gold_rel_pos)
    
    # If prediction is >30% away in document position, possible bias
    if position_diff > 0.3:
        return True
    
    return False


def rule15_model_stops_at_punctuation(pred, gold):
    """Model stops extraction at punctuation/boundary unnecessarily."""
    if not pred or not gold:
        return False
    
    pred_clean = pred.strip()
    gold_clean = gold.strip()
    
    # Prediction must be prefix of gold
    if not gold_clean.startswith(pred_clean):
        return False
    
    # Get the character right after prediction in gold
    if len(pred_clean) < len(gold_clean):
        next_char = gold_clean[len(pred_clean)]
        
        # Model stopped right before punctuation/connector
        stop_chars = [',', ';', ':', '-', '(', ')', 'and', 'or', 'with', 'for']
        remaining = gold_clean[len(pred_clean):].strip()
        
        if next_char in stop_chars or any(remaining.startswith(w) for w in stop_chars):
            # But there's substantial content after the stop point
            remaining_tokens = len(tokenize(remaining))
            if remaining_tokens >= 3:
                return True
    
    return False


# -----------------------------
# 3. Apply rules to dataframe
# -----------------------------


def compute_dataset_error_flags(row, length_threshold):
    """
    Enhanced compute function with improved medical domain awareness.
    """
    ctx = row.get("context", "") or ""
    q = row.get("question", "") or ""
    gold = row.get("answer", "") or ""
    pred = row.get("predicted_answer", "") or ""

    r1 = rule1_length_anomaly(gold, length_threshold)
    r2 = rule2_multi_clause(gold)
    r3 = rule3_low_question_similarity(gold, q)
    r5 = rule5_question_type_mismatch(gold, q)
    r6 = rule6_multiple_occurrences_in_context(gold, ctx)
    r7 = rule7_boundary_weirdness(gold)
    r8 = rule8_pred_answers_question_better(pred, gold, q)
    r9 = rule9_question_not_starting_with_qword(q)

    # Enhanced rule 10: Vague/incomplete questions (medical QA specific)
    r10 = rule10_question_too_vague(q)
    
    # Enhanced rule 11: Medical domain semantic mismatch
    r11 = rule11_medical_semantic_mismatch(gold, q)
    
    # Model-behavior-focused rules (12-15)
    r12 = rule12_model_underextraction(pred, gold, q)
    r13 = rule13_model_wrong_entity_same_type(pred, gold, q)
    r14 = rule14_model_position_bias(pred, ctx, gold)
    r15 = rule15_model_stops_at_punctuation(pred, gold)

    flags = {
        "rule1_length_anomaly": r1,
        "rule2_multi_clause": r2,
        "rule3_low_q_similarity": r3,
        "rule5_qtype_mismatch": r5,
        "rule6_multi_occurrences": r6,
        "rule7_boundary_weirdness": r7,
        "rule8_pred_better_q_alignment": r8,
        "rule9_question_not_starting_with_qword": r9,
        "rule10_question_too_vague": r10,
        "rule11_medical_semantic_mismatch": r11,
        "rule12_model_underextraction": r12,
        "rule13_model_wrong_entity_same_type": r13,
        "rule14_model_position_bias": r14,
        "rule15_model_stops_at_punctuation": r15,
    }
    
    # Enhanced scoring with medical domain weighting
    base_score = sum([r1, r2, r3, r5, r6, r7, r8, r9])
    medical_penalty = 1 if (r10 or r11) else 0  # Extra weight for medical-specific issues
    
    flags["dataset_error_score"] = base_score + medical_penalty
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
    if "answer" not in df.columns:
        if "answers" in df.columns:
            df["answer"] = [v["text"][0] for v in df["answers"].values]
        # else:
        #     df['answer'] = ""
    # Compute length threshold
    gold_lengths = df["answer"].fillna("").apply(lambda x: len(tokenize(x)))
    length_p95 = (
        np.percentile(gold_lengths, length_percentile) if len(gold_lengths) > 0 else 100
    )
    length_med = np.median(gold_lengths) if len(gold_lengths) > 0 else 0
    mad = np.median(np.abs(gold_lengths - length_med)) if len(gold_lengths) > 0 else 0
    length_threshold = max(length_p95, length_med + 2 * mad)

    # Apply rules
    error_flags = df.apply(
        lambda row: compute_dataset_error_flags(row, length_threshold), axis=1
    )
    df_with_flags = pd.concat([df, error_flags], axis=1)

    # Update threshold if different from default
    if error_threshold != 3:
        df_with_flags["is_dataset_error"] = (
            df_with_flags["dataset_error_score"] >= error_threshold
        )

    # Compute summary
    rule_cols = [c for c in df_with_flags.columns if c.startswith("rule")]
    summary = pd.DataFrame(
        {
            "count": df_with_flags[rule_cols].sum(),
            "percent": df_with_flags[rule_cols].mean() * 100,
        }
    ).sort_values("count", ascending=False)

    return df_with_flags, summary


def apply_rules_to_dataset(dataset_df, length_percentile=95):
    """
    Apply rule-based error detection to a dataset and return only rule columns.
    Designed for integration with other analysis results.

    Parameters:
    -----------
    dataset_df : pd.DataFrame
        Must contain columns: 'context', 'question', 'answer' (or 'answers')
        Optional: 'predicted_answer'
    length_percentile : float
        Percentile for length anomaly threshold (default: 95)

    Returns:
    --------
    pd.DataFrame : DataFrame with 'id' column and rule flag columns only
    """
    # Ensure we have an id column
    if "id" not in dataset_df.columns:
        raise ValueError("Dataset must have an 'id' column for proper merging")

    # Create a working copy
    df = dataset_df.copy()

    # Ensure answer column exists
    if "answer" not in df.columns:
        if "answers" in df.columns:
            df["answer"] = [
                v["text"][0] if isinstance(v, dict) and "text" in v else ""
                for v in df["answers"].values
            ]
        else:
            df["answer"] = ""

    # Add predicted_answer if missing (rules that need it will return False)
    if "predicted_answer" not in df.columns:
        df["predicted_answer"] = ""

    # Compute length threshold
    gold_lengths = df["answer"].fillna("").apply(lambda x: len(tokenize(x)))
    length_p95 = (
        np.percentile(gold_lengths, length_percentile) if len(gold_lengths) > 0 else 100
    )
    length_med = np.median(gold_lengths) if len(gold_lengths) > 0 else 0
    mad = np.median(np.abs(gold_lengths - length_med)) if len(gold_lengths) > 0 else 0
    length_threshold = max(length_p95, length_med + 2 * mad)

    # Apply rules
    error_flags = df.apply(
        lambda row: compute_dataset_error_flags(row, length_threshold), axis=1
    )

    # Combine id with rule flags
    rule_df = pd.DataFrame({"id": df["id"].values})
    rule_df = pd.concat([rule_df, error_flags], axis=1)

    return rule_df


# -----------------------------
# Composite Error Categories
# -----------------------------

def create_composite_categories(df):
    """
    Create meaningful error categories by combining related rules.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rule flags from detect_dataset_errors()
    
    Returns:
    --------
    pd.DataFrame : DataFrame with additional composite category columns
    """
    df = df.copy()
    
    # Semantic Mismatch Category (Most Important)
    df['semantic_mismatch'] = (
        df['rule3_low_q_similarity'] |  # Low question-answer overlap
        df['rule5_qtype_mismatch']      # Question type doesn't match answer
    )
    
    # Template/Generation Artifacts
    df['template_artifacts'] = (
        df['rule8_pred_better_q_alignment'] |  # Predicted answer aligns better
        df['rule9_question_not_starting_with_qword']  # Poorly formed questions
    )
    
    # Span Extraction Issues
    df['span_issues'] = (
        df['rule7_boundary_weirdness'] | # Boundary problems
        df['rule1_length_anomaly']       # Length anomalies
    )
    
    # Structural Problems
    df['structural_issues'] = (
        df['rule2_multi_clause'] |       # Multi-clause answers
        df['rule6_multi_occurrences']    # Multiple occurrences in context
    )
    
    return df


def create_medical_error_patterns(df):
    """
    Identify medical QA specific error patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rule flags and text columns (question, answer, context)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with additional medical error pattern columns
    """
    df = df.copy()
    
    # Medication-related errors (most common in medical QA)
    medication_questions = df['question'].str.contains(
        'medication|drug|dose|dosage|frequency|mg|tablet|pill|how often|take', 
        case=False, na=False
    )
    
    df['med_semantic_error'] = (
        medication_questions & 
        (df['rule3_low_q_similarity'] | df['rule5_qtype_mismatch'])
    )
    
    # Diagnostic/procedure errors
    diagnostic_questions = df['question'].str.contains(
        'diagnos|procedure|treatment|condition|disease|disorder', 
        case=False, na=False
    )
    
    df['diagnostic_error'] = (
        diagnostic_questions & 
        (df['rule3_low_q_similarity'] | df['rule2_multi_clause'])
    )
    
    # Temporal relationship errors (common in clinical notes)
    temporal_questions = df['question'].str.contains(
        'when|how long|previous|current|since|before|after|during|history', 
        case=False, na=False
    )
    
    df['temporal_error'] = (
        temporal_questions & 
        df['rule5_qtype_mismatch']
    )
    
    # Frequency-specific errors (subset of medication but important)
    # Use rule11 which detects wrong medication in frequency answers
    frequency_questions = df['question'].str.contains(
        r'how often|frequency|times per', 
        case=False, na=False
    )
    
    df['frequency_error'] = (
        frequency_questions & 
        df['rule11_medical_semantic_mismatch']
    )
    
    # Vague question errors - enhanced to catch "Previous [medication]" pattern
    # Use rule10 directly for consistency
    df['vague_question_error'] = df['question'].apply(rule10_question_too_vague)
    
    return df


def analyze_model_error_patterns(df):
    """
    Analyze model-specific behavioral patterns from rules 12-15.
    Focuses on what the model does wrong, not dataset quality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rule flags including rules 12-15
    
    Returns:
    --------
    pd.DataFrame : DataFrame with model behavior pattern columns
    """
    df = df.copy()
    
    # Model Underextraction Pattern
    # Specific: Model stopped at "aspirin 81" when answer is "aspirin 81 mg daily"
    # General Class: Incomplete span extraction
    df['model_underextraction_pattern'] = df['rule12_model_underextraction']
    
    # Entity Confusion Pattern  
    # Specific: Model said "Percocet" when asked about "Colace" frequency
    # General Class: Right semantic type, wrong specific entity
    df['model_entity_confusion_pattern'] = df['rule13_model_wrong_entity_same_type']
    
    # Position Bias Pattern
    # Specific: Model picked first medication mention, not the correct later one
    # General Class: Positional bias in span selection
    df['model_position_bias_pattern'] = df['rule14_model_position_bias']
    
    # Boundary Detection Failure Pattern
    # Specific: Model stopped at comma, missing "and as needed for pain"
    # General Class: Premature termination at punctuation
    df['model_boundary_failure_pattern'] = df['rule15_model_stops_at_punctuation']
    
    # Composite: Any model behavioral error
    df['has_model_behavior_error'] = (
        df['model_underextraction_pattern'] |
        df['model_entity_confusion_pattern'] |
        df['model_position_bias_pattern'] |
        df['model_boundary_failure_pattern']
    )
    
    return df


def enhanced_error_analysis(df):
    """
    Perform comprehensive error analysis with composite categories, medical patterns,
    and model behavioral patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rule flags from detect_dataset_errors()
    
    Returns:
    --------
    tuple : (df_enhanced, composite_summary, medical_summary, model_behavior_summary)
    """
    # Apply composite categories
    df_enhanced = create_composite_categories(df)
    df_enhanced = create_medical_error_patterns(df_enhanced)
    df_enhanced = analyze_model_error_patterns(df_enhanced)
    
    # Generate summaries
    composite_cols = ['semantic_mismatch', 'template_artifacts', 'span_issues', 'structural_issues']
    composite_summary = pd.DataFrame({
        'count': df_enhanced[composite_cols].sum(),
        'percent': df_enhanced[composite_cols].mean() * 100,
        'error_precision': [df_enhanced[df_enhanced[col]]['is_dataset_error'].mean() * 100 
                           for col in composite_cols]
    }).sort_values('count', ascending=False)
    
    medical_cols = ['med_semantic_error', 'diagnostic_error', 'temporal_error', 
                    'frequency_error', 'vague_question_error']
    medical_summary = pd.DataFrame({
        'count': df_enhanced[medical_cols].sum(),
        'percent': df_enhanced[medical_cols].mean() * 100,
        'error_precision': [df_enhanced[df_enhanced[col]]['is_dataset_error'].mean() * 100 
                           if df_enhanced[col].sum() > 0 else 0
                           for col in medical_cols]
    }).sort_values('count', ascending=False)
    
    # Model behavior patterns summary
    model_cols = ['model_underextraction_pattern', 'model_entity_confusion_pattern',
                  'model_position_bias_pattern', 'model_boundary_failure_pattern']
    model_behavior_summary = pd.DataFrame({
        'count': df_enhanced[model_cols].sum(),
        'percent': df_enhanced[model_cols].mean() * 100,
        'of_all_errors': [df_enhanced[df_enhanced[col]].shape[0] / df_enhanced[df_enhanced['predicted_answer'] != df_enhanced['answer']].shape[0] * 100
                         if 'predicted_answer' in df_enhanced.columns and df_enhanced[col].sum() > 0
                         else 0
                         for col in model_cols]
    }).sort_values('count', ascending=False)
    
    return df_enhanced, composite_summary, medical_summary, model_behavior_summary


# -----------------------------
# Convenience Functions
# -----------------------------

def analyze_medical_qa_errors(df, error_threshold=2, length_percentile=95, verbose=True):
    """
    Complete analysis pipeline for medical QA error detection with enhanced categorization
    and model behavioral pattern analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with columns: context, question, answer, predicted_answer (required for model patterns)
    error_threshold : int
        Minimum rules to trigger for dataset error classification
    length_percentile : float
        Percentile for length anomaly detection
    verbose : bool
        Whether to print summary statistics
    
    Returns:
    --------
    dict : Complete analysis results with all summaries and enhanced dataframe
    """
    # Step 1: Basic error detection
    df_with_flags, basic_summary = detect_dataset_errors(
        df, error_threshold=error_threshold, length_percentile=length_percentile
    )
    
    # Step 2: Enhanced analysis with composite categories and model patterns
    df_enhanced, composite_summary, medical_summary, model_behavior_summary = enhanced_error_analysis(df_with_flags)
    
    if verbose:
        total_examples = len(df_enhanced)
        flagged_errors = df_enhanced['is_dataset_error'].sum()
        print("=== MEDICAL QA ERROR ANALYSIS RESULTS ===")
        print(f"Total examples analyzed: {total_examples}")
        print(f"Flagged as dataset errors: {flagged_errors} ({flagged_errors/total_examples*100:.1f}%)")
        print()
        
        # Check if we have predictions for model behavior analysis
        has_predictions = 'predicted_answer' in df_enhanced.columns and df_enhanced['predicted_answer'].notna().any()
        if has_predictions:
            model_errors = df_enhanced['has_model_behavior_error'].sum()
            print("=== MODEL BEHAVIORAL PATTERNS ===")
            print(f"Examples with model behavior errors: {model_errors} ({model_errors/total_examples*100:.1f}%)")
            print(model_behavior_summary.head())
            print()
        
        print("=== TOP COMPOSITE ERROR CATEGORIES ===")
        print(composite_summary.head())
        print()
        
        print("=== MEDICAL DOMAIN-SPECIFIC ERRORS ===")
        print(medical_summary.head())
    
    return {
        'enhanced_dataframe': df_enhanced,
        'basic_rule_summary': basic_summary,
        'composite_summary': composite_summary,
        'medical_summary': medical_summary,
        'model_behavior_summary': model_behavior_summary,
        'total_examples': len(df_enhanced),
        'flagged_errors': df_enhanced['is_dataset_error'].sum(),
        'error_rate': df_enhanced['is_dataset_error'].mean()
    }


def print_error_examples(df_enhanced, category='semantic_mismatch', max_examples=3):
    """
    Print examples of specific error categories for manual inspection.
    
    Parameters:
    -----------
    df_enhanced : pd.DataFrame
        Enhanced dataframe from analyze_medical_qa_errors()
    category : str
        Error category to show examples for
    max_examples : int
        Maximum number of examples to display
    """
    if category not in df_enhanced.columns:
        print(f"Category '{category}' not found. Available categories:")
        error_cols = [col for col in df_enhanced.columns 
                     if any(x in col for x in ['semantic', 'template', 'span', 'structural', 'med_', 'diagnostic', 'temporal'])]
        print(error_cols)
        return
    
    examples = df_enhanced[df_enhanced[category] == True].head(max_examples)
    
    print(f"\n=== EXAMPLES OF {category.upper().replace('_', ' ')} ERRORS ===")
    for idx, row in examples.iterrows():
        print(f"\nExample {idx + 1}:")
        print(f"Question: {row.get('question', 'N/A')[:100]}...")
        print(f"Gold Answer: {row.get('answer', 'N/A')[:100]}...")
        if 'predicted_answer' in row and pd.notna(row['predicted_answer']):
            print(f"Predicted: {row.get('predicted_answer', 'N/A')[:100]}...")
        print(f"Error Score: {row.get('dataset_error_score', 'N/A')}")
        print("-" * 60)