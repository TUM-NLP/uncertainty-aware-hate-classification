import time, torch
from typing import List, Any, Dict
from .prompt import PromptItem, build_prompts_context, build_prompts_baseline, build_prompts_persona
from .rag_utils import get_context
from .helpers import make_serializable
import pandas as pd
import time, torch, json
from typing import Any, Callable
from sklearn.preprocessing import MinMaxScaler

def evaluate_prompt_items(pipeline, prompt_items: List[PromptItem], batch_size: int = 4):
    """
    Runs the pipeline over all PromptItem.prompt strings (batched) and returns:
    - a dict keyed by comment_id
      * baseline → results[comment_id] = payload
      * persona/annotation-grounded → results[comment_id][annotator_id] = payload
    - also attaches batch timing info in results["info"]
    """
    
    prompts = [it.prompt for it in prompt_items]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    ret = pipeline.get_uncertainty_scores(prompts, batch_size=4)
    if isinstance(ret, tuple) and len(ret) == 2:
        pipe_out, batch_meta = ret
    else:
        pipe_out, batch_meta = ret, []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    by_comment = {}
    for item, out in zip(prompt_items, pipe_out):
        payload = {
            "text": item.text,
            "prompt": item.prompt,
            "uncertainty": {
                "ccp": float(out["ccp"]),
                "msp": float(out["msp"]),
                "perplexity": float(out["perplexity"]),
                "mte": float(out["mte"]),
                "mpmi": float(out["mpmi"]),
                "mcpmi": float(out["mcpmi"]),
                "ptrue": float(out["ptrue"]),
            },
            "generation_text": out["output"],
            "prompt_length": len(item.prompt),
            "timing": {
                **out.get("timing", {}),                    # ← batch_id/batch_seconds from pipeline (if present)
            },
            "meta": {},
        }
        if item.persona:
            payload["meta"]["persona"] = item.persona
        if item.examples:
            payload["meta"]["examples_in_prompt"] = item.examples

        # store results
        if item.approach == "baseline":
            by_comment[item.comment_id] = payload
        else:
            bucket = by_comment.setdefault(item.comment_id, {})
            bucket[item.annotator_id] = payload   # now nested by annotator_id

    by_comment["info"] = {
        "total_runtime_seconds": round(t1 - t0, 3),
        "model": getattr(pipeline.model_adapter, "model_path", None),
        "approach": "batched",
        "test_path": None,
        "batches": batch_meta,  # may be empty if pipeline didn’t return meta
    }
    return by_comment


def run_baseline(pipeline, test_df: pd.DataFrame, model_path: str, test_df_path: str, batch_size: int = 4):
    items = build_prompts_baseline(test_df)

    results = evaluate_prompt_items(pipeline, items, batch_size=batch_size)
    results["info"] = {
        "approach": "baseline",
        "model": model_path,
        "test_path": test_df_path,
        "generation_params": pipeline.generation_params
    }
    results=make_serializable(results)
    return results

def run_persona(pipeline, test_df: pd.DataFrame, model_path: str, test_df_path: str, batch_size: int = 4):
    items = build_prompts_persona(test_df)
    results = evaluate_prompt_items(pipeline, items, batch_size=batch_size)
    results["info"] = {
        "approach": "persona_prompting",
        "model": model_path,
        "test_path": test_df_path,
        "generation_params": pipeline.generation_params
    }
    results=make_serializable(results)
    return results

def run_context_retrieval(
    test_df: pd.DataFrame,
    knowledge_base_id: str,
    number_of_results: int,
    filtering: int,
    region:str,
):
    examples_dict = {}
    for _, row in test_df.iterrows():  # include every annotator row
        ex_block, filters, rag_seconds = get_context(
            row,
            knowledge_base_id=knowledge_base_id,
            number_of_results=max(number_of_results*3,10),
            filtering=filtering,
            region=region,
        )

        # Initialize if the comment_id key does not exist yet
        if row["comment_id"] not in examples_dict:
            examples_dict[row["comment_id"]] = {}

        # Store example block
        examples_dict[row["comment_id"]][row["annotator_id"]] = {}
        examples_dict[row["comment_id"]][row["annotator_id"]]['examples'] = ex_block
        examples_dict[row["comment_id"]][row["annotator_id"]]['filters'] = filters
        examples_dict[row["comment_id"]][row["annotator_id"]]['rag_seconds'] = rag_seconds

    return examples_dict

def run_annotation_grounded(
    *,
    pipeline,
    test_df: pd.DataFrame,
    model_path: str,
    test_df_path: str,
    context_path: str,
    example_count: int,
    batch_size: int = 4,
):
    """
    Build annotation-grounded (few-shot) prompts per (comment_id, annotator_id),
    run the uncertainty pipeline in batches, and return a results dict.

    Output shape:
      results["info"] = {...}
      results[comment_id][annotator_id] = { text, prompt, uncertainty, generation_text, ... }
    """
    with open(context_path, "r") as f:
        context_dict = json.load(f)
        
    # 1) Build all annotation-grounded PromptItem entries (one per row/annotator)
    items = build_prompts_context(
        test_df=test_df,
        context_dict=context_dict,
        example_count=example_count
    )

    # 2) Evaluate in batches via your pipeline (returns dict keyed by comment_id, annotator_id)
    results = evaluate_prompt_items(pipeline, items, batch_size=batch_size)

    # 3) Stamp run metadata
    results["info"] = {
        "approach": "annotation_grounded_few_shot",
        "model": model_path,
        "test_path": test_df_path,
        "generation_params": pipeline.generation_params,
        # keep any batch timing table already added by evaluate_prompt_items in results["info"]["batches"]
        **{k: v for k, v in results.get("info", {}).items()},  # preserve timing fields if already present
    }

    results=make_serializable(results)

    return results
