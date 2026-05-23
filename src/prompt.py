# ------- helpers to render prompts & build a flat batch with metadata -------

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import pandas as pd
from config import zero_shot_prompt_template, zero_shot_demographics_prompt_template, rag_template
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import pandas as pd
from .rag_utils import get_context, extract_example

'''
ex.
persona={
    "annotator_gender": "female",
    "annotator_race": "Asian",
    "annotator_age": "25–34",
    "annotator_education": "College",
}
examples="- Text: I hate you...\n  Label: 2\n- Text: That’s silly...\n  Label: 0"
'''

@dataclass
class PromptItem:
    comment_id: Any
    approach: str                    # "baseline" | "persona" | "context"
    prompt: str
    text: str
    annotator_id: Optional[Any] = None
    persona: Optional[Dict[str, str]] = None   # for persona
    examples: Optional[str] = None             # for RAG
    filters: Optional[Dict[str, Any]] = None # for RAG
    rag_seconds: Optional[float] = None # for RAG


def get_baseline_prompt(text: str) -> str:
    return zero_shot_prompt_template.format(text=text)

def get_persona_prompt(text: str, row: pd.Series) -> str:
    education = row.get("annotator_education", row.get("annotator_educ", ""))
    
    return zero_shot_demographics_prompt_template.format(
        text=text,
        annotator_gender=row.get("annotator_gender", ""),
        annotator_race=row.get("annotator_race", ""),
        annotator_age=row.get("annotator_age", ""),
        annotator_education=education,
    )

def get_annotation_grounded_few_shot_prompt(text: str, examples_block: str) -> str:
    return rag_template.format(text=text, examples=examples_block)

# ------- builders for each approach (returns a flat list[PromptItem]) -------

def build_prompts_baseline(test_df: pd.DataFrame) -> List[PromptItem]:
    items = []
    # one prompt per unique comment
    for _, row in test_df.groupby("comment_id", as_index=False).first().iterrows():
        p = get_baseline_prompt(row["text"])
        items.append(PromptItem(
            comment_id=row["comment_id"],
            approach="baseline",
            prompt=p,
            text=row["text"],
        ))
    return items

def build_prompts_persona(test_df: pd.DataFrame) -> List[PromptItem]:
    """
    Expect multiple rows per comment_id (one per annotator persona).
    """
    items = []
    # use all rows; each row corresponds to a persona view of the same comment
    for _, row in test_df.iterrows():
        p = get_persona_prompt(row["text"], row)
        persona = {
            "annotator_gender": row.get("annotator_gender", ""),
            "annotator_race": row.get("annotator_race", ""),
            "annotator_age": row.get("annotator_age", ""),
            "annotator_education": row.get("annotator_education", ""),
        }
        items.append(PromptItem(
            comment_id=row["comment_id"],
            annotator_id=row["annotator_id"],
            approach="persona",
            prompt=p,
            text=row["text"],
            persona=persona,
        ))
    return items


def build_prompts_context(
    test_df: pd.DataFrame,
    context_dict:dict,
    example_count:int
) -> List[PromptItem]:
    items: List[PromptItem] = []
    for _, row in test_df.iterrows():  # <-- include every annotator row
        comment_id=str(row['comment_id'])
        annotator_id=str(row['annotator_id'])
        
        contexts=context_dict[comment_id][annotator_id]
        
        ex_block = extract_example(contexts,example_count)
        
        prompt = get_annotation_grounded_few_shot_prompt(row["text"], ex_block)
        items.append(PromptItem(
            comment_id=row["comment_id"],
            annotator_id=row.get("annotator_id"),
            approach="context",
            prompt=prompt,
            text=row["text"],
            examples=ex_block
        ))
    return items



