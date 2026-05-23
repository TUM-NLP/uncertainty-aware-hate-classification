import pandas as pd
import re, json
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def flatten_to_dataframe(nested_dict):
    """
    Flattens a one- or two-level nested dictionary of annotated samples into a DataFrame.

    Each row includes:
    - comment_id: Unique sample identifier (e.g., 'AU-Reddit-0')
    - annotator_id: Inner key if two-level, else None
    - text, prompt, generation_text, prompt_length
    - timing info: batch_id, batch_seconds, rag_seconds
    - filters: expanded into annotator_gender, age, education, race
    - uncertainty metrics (flattened)

    Parameters:
        nested_dict (dict): One- or two-level nested dictionary of samples

    Returns:
        pd.DataFrame: Flattened sample data
    """
    rows = []

    def extract_filter_value(filters, key_name):
        """Extract the value for a given filter key, or return None if not found."""
        try:
            for f in filters.get("andAll", []):
                if "equals" in f and f["equals"]["key"] == key_name:
                    return f["equals"]["value"]
        except Exception:
            pass
        return None

    for outer_key, outer_val in nested_dict.items():
        # Two-level case
        if isinstance(outer_val, dict) and all(isinstance(v, dict) for v in outer_val.values()):
            for inner_key, sample in outer_val.items():
                if not isinstance(sample, dict) or "uncertainty" not in sample:
                    continue
                filters = sample.get("filters", {})
                row = {
                    "comment_id": f"{outer_key}",
                    "annotator_id": inner_key,
                    "text": sample.get("text"),
                    "prompt": sample.get("prompt"),
                    "generation_text": sample.get("generation_text"),
                    "prompt_length": sample.get("prompt_length"),
                    "batch_id": sample.get("timing", {}).get("batch_id"),
                    "batch_seconds": sample.get("timing", {}).get("batch_seconds"),
                    "rag_seconds": sample.get("timing", {}).get("rag_seconds"),
                    "annotator_gender": extract_filter_value(filters, "annotator_gender"),
                    "annotator_age": extract_filter_value(filters, "annotator_age"),
                    "annotator_education": extract_filter_value(filters, "annotator_education"),
                    "annotator_race": extract_filter_value(filters, "annotator_race"),
                }
                row.update(sample["uncertainty"])
                rows.append(row)

        # One-level case
        elif isinstance(outer_val, dict) and "uncertainty" in outer_val:
            filters = outer_val.get("filters", {})
            row = {
                "comment_id": outer_key,
                "annotator_id": None,
                "text": outer_val.get("text"),
                "prompt": outer_val.get("prompt"),
                "generation_text": outer_val.get("generation_text"),
                "prompt_length": outer_val.get("prompt_length"),
                "batch_id": outer_val.get("timing", {}).get("batch_id"),
                "batch_seconds": outer_val.get("timing", {}).get("batch_seconds"),
                "rag_seconds": outer_val.get("timing", {}).get("rag_seconds"),
                "annotator_gender": extract_filter_value(filters, "annotator_gender"),
                "annotator_age": extract_filter_value(filters, "annotator_age"),
                "annotator_education": extract_filter_value(filters, "annotator_education"),
                "annotator_race": extract_filter_value(filters, "annotator_race"),
            }
            row.update(outer_val["uncertainty"])
            rows.append(row)

    return pd.DataFrame(rows)

def make_serializable(obj):
    
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif hasattr(obj, "__dict__"):  # generic objects
        return {k: make_serializable(v) for k, v in obj.__dict__.items()}
    else:
        return obj
    
def normalize_uncertainty_columns(df, uncertainty_cols=None):
    """
    Normalizes the given uncertainty columns in the DataFrame using MinMaxScaler.

    Parameters:
        df (pd.DataFrame): Your flattened data
        uncertainty_cols (list): List of column names to normalize. If None, auto-detects.

    Returns:
        pd.DataFrame: Updated DataFrame with normalized uncertainty columns
    """
    if uncertainty_cols is None:
        # Auto-detect uncertainty columns by checking typical names
        uncertainty_cols = ['ccp', 'msp', 'perplexity', 'mte', 'mpmi', 'mcpmi', 'ptrue'] #TODO: To config
        uncertainty_cols = [col for col in uncertainty_cols if col in df.columns]

    scaler = MinMaxScaler()
    df[uncertainty_cols] = scaler.fit_transform(df[uncertainty_cols])

    return df



import re
import json

def extract_hate_speech_score(text: str):
    """
    Extracts the hate speech score (0/1/2) from model output.
    Priority:
      1) Last <output>...</output> JSON block (any tag casing).
      2) Otherwise, last textual mention like:
         - "hate_speech_score": 1
         - Hate Speech Score: 2
         - score = 0
    Returns:
      int in {0,1,2} or None if not found.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    def _clamp_score(val):
        try:
            iv = int(str(val).strip())
            return iv if iv in (0,1,2) else None
        except Exception:
            return None

    def _canon(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    # 1) Try the LAST <output>...</output> JSON block (case-insensitive tags)
    tag_blocks = re.findall(
        r"<\s*output\s*>(.*?)<\s*/\s*output\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if tag_blocks:
        for block in reversed(tag_blocks):
            candidate = block.strip()
            # If not a clean JSON object, try to grab the first {...} inside
            if not (candidate.lstrip().startswith("{") and candidate.rstrip().endswith("}")):
                m = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
                candidate = m.group(0).strip() if m else candidate
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    # Look for common score keys (canonicalized)
                    for k, v in obj.items():
                        if _canon(k) in {"hatespeechscore", "score"}:
                            sv = _clamp_score(v)
                            if sv is not None:
                                return sv
                    # Second pass for exact "hate_speech_score" style
                    for k, v in obj.items():
                        if _canon(k) == "hatespeechscore":
                            sv = _clamp_score(v)
                            if sv is not None:
                                return sv
            except Exception:
                pass

    # 1b) If there was an <output> opening without a closing tag, try to parse any JSON with score
    m_any_json = re.search(
        r'\{[^{}]*?(?:hate\s*[_ ]?\s*speech\s*[_ ]?\s*score|score)[^{}]*?\}',
        text, flags=re.IGNORECASE | re.DOTALL
    )
    if m_any_json:
        try:
            obj = json.loads(m_any_json.group(0))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if _canon(k) in {"hatespeechscore", "score"}:
                        sv = _clamp_score(v)
                        if sv is not None:
                            return sv
        except Exception:
            pass

    # 2) Fallback: find the LAST textual mention of a score 
    # Examples matched:
    #   "hate_speech_score": 1
    #   hate speech score : 2
    #   Hate Speech Score=0
    #   score: 1
    pattern = r'(?:hate\s*[_ ]?\s*speech\s*[_ ]?\s*score|score)\s*[:=]\s*([0-2])'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if matches:
        last = matches[-1]
        return _clamp_score(last.group(1))

    # Nothing found
    return None

def process_llm_output(df):
    # Create new columns if they don't exist
    if 'hate_speech_score' not in df.columns:
        df['hate_speech_score'] = None
    '''
    if 'justification' not in df.columns:
        df['hate_speech_reason'] = None
    '''
    # Iterate through the dataframe rows
    for index, record in df.iterrows():
        hate_speech_score = extract_hate_speech_score(record['generation_text'])
        df.at[index, 'hate_speech_score'] = hate_speech_score
        #df.at[index, 'hate_speech_reason'] = justification

    
    return df
