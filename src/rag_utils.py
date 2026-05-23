from src import rag
import time
import csv, io
from io import StringIO
from collections import defaultdict, Counter


def extract_example(
    context_dict: dict,
    example_count: int
    ) -> str:
    """
    Parse the context dictionary into a human-readable example.

    Expects a dictionary where keys are Texts and values are hate speech scores.
    
    Returns a single string like:
        "Text: ... \nhate_speech_score: <score>"
    """
    example_str = f''
    counter = 0
    for text, score in context_dict.get('examples', {}).items():
        if counter < example_count:
            example_str += f'Text: {text} \nhate_speech_score: {score}\n\n'
            counter += 1
        else:
            break
    if len(example_str) == 0:
        print(f'Empty/malformed context_dict')

    return example_str


def create_filters(row):
    education = row.get('annotator_education', row.get('annotator_educ', None))

    return {
        "andAll": [
            {"equals": {"key": "annotator_gender", "value": row['annotator_gender']}},
            {"equals": {"key": "annotator_age", "value": row['annotator_age']}},
            {"equals": {"key": "annotator_race", "value": row['annotator_race']}},
            {"equals": {"key": "annotator_educ", "value": education}} #change to education for second dataset
            #{"equals": {"key": "annotator_education", "value": education}} #change to education for second dataset

        ]
    }

def majority_vote_by_text(docs):
    """
    Groups documents with identical text content and applies majority voting 
    on the 'hatespeech' field.

    Args:
        docs (list): List of dicts, each containing 'content'['text'] with a CSV string.

    Returns:
        dict: Mapping from text -> majority hatespeech label (int)
    """
    grouped = defaultdict(list)

    # Extract text and hatespeech score from each doc
    for doc in docs:
        text_data = doc["content"]["text"]
        # Parse CSV content
        reader = csv.DictReader(io.StringIO(text_data))
        for row in reader:
            text = row["text"].strip()
            hatespeech = int(row["hatespeech"])
            grouped[text].append(hatespeech)

    # Compute majority vote per text
    majority_votes = {}
    
    for text, labels in grouped.items():
        counts = Counter(labels)
        max_count = max(counts.values())
        tied_labels = [label for label, c in counts.items() if c == max_count] #labels with the max count, can be 1 or multiple in case of tie
        majority_label = max(tied_labels)   # <-- always picks higher label if tie
        majority_votes[text] = majority_label

    return majority_votes

# --- separate context retriever (per row) ---
def get_context(
    row,
    *,
    knowledge_base_id,
    number_of_results: int,
    filtering: int,
    region: str,
) -> str:
    """
    Retrieve docs for this row and return a single examples string.
    The returned string is already formatted like:
      "Text: ... hate_speech_score: 0\n\nText: ... hate_speech_score: 2\n\n..."
    """
    if filtering:
        filters = create_filters(row)
    
        kb_configs = {
            "vectorSearchConfiguration": {
                "numberOfResults": number_of_results,
                "filter": filters,
            }
        }
    else:
        filters = {}
        kb_configs = {
            "vectorSearchConfiguration": {
                "numberOfResults": number_of_results,
            }
        }
    t0 = time.time()
    r = rag.Rag(bedrock_region=region, kb_configs=kb_configs, rag_template='') # no need a rag template, no generation
    _ctx, docs = r.get_context(knowledge_base_id, row["text"])
    t1 = time.time()
    rag_seconds = round(t1 - t0, 2)
    
    if len(docs) == 0:
        comment_id = row['comment_id']
        annotator_id = row['annotator_id']
        print(f'No examples for ({comment_id},{annotator_id}). Removing filters..')
        kb_configs = {
        "vectorSearchConfiguration": {
            "numberOfResults": number_of_results,
            }
        }
        r = rag.Rag(bedrock_region=region, kb_configs=kb_configs, rag_template='')
        _ctx, docs = r.get_context(knowledge_base_id, row["text"])
        filters = {}

    majority_voted_docs = majority_vote_by_text(docs)

    '''
    examples = []
    for d in docs or []:
        raw = d["content"]["text"]
        ex = extract_example(raw)  # you already return "Text: ... hate_speech_score: ..."
        if ex:
            examples.append(ex)
    ex_block = "\n\n".join(examples).strip()
    
    if len(examples) == 0:
        comment_id = row['comment_id']
        annotator_id = row['annotator_id']
        print(f'No examples for ({comment_id},{annotator_id})')
    '''
        
    return majority_voted_docs, filters, rag_seconds

