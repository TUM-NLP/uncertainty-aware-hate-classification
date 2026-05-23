from typing import Any
import json
import pickle
import warnings

import config
import transformers
from huggingface_hub import login
from datetime import datetime
from pathlib import Path


from src import rag, rag_utils, evaluate, uncertainty, prompt


# ----------------------------
# Helpers
# ----------------------------

def load_test_df(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Any, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def setup_environment():
    """Configure warnings, HuggingFace login, and transformer logging."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to")
    transformers.logging.set_verbosity_error()

    # HF login
    login(token=config.hf_login_token)


# ----------------------------
# Main Workflow
# ----------------------------

def main():
    setup_environment()

    # --- Create output folder if it does not exist ---
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Test Data ---
    test_df = load_test_df(config.test_df_path)
    
    # --- Context Retrieval ---
    contexts = evaluate.run_context_retrieval(
        test_df=test_df,
        knowledge_base_id=config.knowledge_base_id,
        number_of_results=config.context_doc_count,
        filtering=1,   # set to 0 to disable metadata filters
        region=config.region
    )

    context_out = f"{config.output_path}/contexts.json"
    save_json(contexts, context_out)

    print(f"[INFO] Saved retrieved contexts ➝ {context_out}")

    

    flag = config.flag
    load_in_4bit = config.load_in_4bit

    # --- Model Initialization ---
    model_adapter, tokenizer = uncertainty.init_whitebox_model(
        model_path=config.model_path,
        load_in_4bit=load_in_4bit
    )

    pipeline = uncertainty.UncertaintyPipeline(
        model_adapter=model_adapter,
        tokenizer=tokenizer,
        device="cuda"
    )

    # --- Run Baseline ---

    baseline_results = evaluate.run_baseline(
        pipeline=pipeline,
        test_df=test_df[:15],
        model_path=config.model_path,
        test_df_path=config.test_df_path,
        batch_size=config.batch_size
    )

    print("[INFO] Baseline evaluation complete.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_out = f"{config.output_path}/baseline_results_{flag}_{timestamp}.json"
    save_json(baseline_results, baseline_out)

    # --- Run Persona ---
    persona_results = evaluate.run_persona(
        pipeline=pipeline,
        test_df=test_df[:15],
        model_path=config.model_path,
        test_df_path=config.test_df_path,
        batch_size=config.batch_size
    )
    print("[INFO] Persona prompting evaluation complete.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_out = f"{config.output_path}/persona_results_{flag}_{timestamp}.json"
    save_json(persona_results, persona_out)

    # --- Run Annotation-Grounded Few-Shot ---
    context_results = evaluate.run_annotation_grounded(
        pipeline=pipeline,
        test_df=test_df[:15],
        model_path=config.model_path,
        test_df_path=config.test_df_path,
        context_path=config.context_path,
        example_count=config.context_doc_count,
        batch_size=config.batch_size
    )
    print("[INFO] Annotation-Grounded Few-Shot prompting evaluation complete.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_out = f"{config.output_path}/annotation-grounded_results_{flag}_{timestamp}.json"

    save_json(context_results, context_out)


if __name__ == "__main__":
    main()
