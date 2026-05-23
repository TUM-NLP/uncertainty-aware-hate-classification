# ------- config-------

hf_login_token='' # TODO: Change this with your Huggingface login 

model_path = "meta-llama/Llama-3.1-8B-Instruct" # TODO: Change with your preferred HF model id 
load_in_4bit = False # TODO: Change this according to your needs
batch_size = 16 # TODO: Change this according to your needs

output_path = 'output/'
context_path = f'data/contexts.json' # Save your extracted context (few-shot examples here)
kb_df_path = f"data/df_knowledge_base_ds1.pkl" # Path to your Knowledge Base
test_df_path = f"data/df_test_ds1.pkl" # Path to test dataset DataFrame

knowledge_base_id="" # Change with your KB id (Amazon Bedrock Knowledge Bases)
data_source_id= "" # Change with your Data Source id  (OpenSearchServerless)
context_doc_count = 3 # Number of context documents retrieved from Knowledge Base 

region="us-east-1"

# Generation parameters (defaults) - greedy

flag = 'greedy'
generation_config = {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "do_sample": False,
    "num_beams": 1,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "stop_strings": None,
    "allow_newlines": True,
    "max_new_tokens": 100,
}

'''
# Alternative Generation parameters - top10_temp07
flag = 'top10_temp07'
generation_config = {
    "temperature": 0.7,        # slightly lower temp -> more focused but still diverse
    "top_k": 10,              # limit to top 10 tokens to encourage diversity
    "top_p": 1.0,             # nucleus sampling
    "do_sample": True,        # enables stochastic sampling
    "num_beams": 1,           # still single decoding path (no beam search)
    "presence_penalty": 0.0,  # discourages repetition a little
    "repetition_penalty": 1.0, # penalty for repeating tokens
    "stop_strings": None,
    "allow_newlines": True,
    "max_new_tokens": 100,
}
'''



zero_shot_prompt_template = (
    "You are an annotator.\n"
    "Rate the degree of hate speech in the text by generating a score 0, 1, or 2.\n"
    "0 = Not Hate Speech, 1 = Ambiguous, 2 = Hate Speech\n\n"
    "Provide a concise but complete justification INSIDE the JSON only (2–4 sentences).\n"
    "Do NOT output any text outside the required block.\n\n"
    "Text to evaluate (read-only):\n<<<\n{text}\n>>>\n\n"
    "OUTPUT REQUIREMENTS:\n"
    " - Return ONLY one <output> block containing valid JSON.\n"
    " - Include BOTH opening <output> and closing </output> tags exactly as shown.\n"
    " - The JSON must include only two fields: \"hate_speech_score\" and \"justification\".\n"
    " - Do NOT write anything before <output> or after </output>.\n"
    "<output>\n"
    "{{\n"
    "  \"hate_speech_score\": <generated_score>,\n"
    "  \"justification\": <explain_your_reasoning_in_2_to_4_sentences>\n"
    "}}\n"
    "</output>\n"
)

zero_shot_demographics_prompt_template = (
    "You are an annotator with the following background:\n\n"
    "Gender: {annotator_gender}\n"
    "Race: {annotator_race}\n"
    "Age: {annotator_age}\n"
    "Education Level: {annotator_education}\n\n"
    "Rate the degree of hate speech in the text by generating a score 0, 1, or 2.\n"
    "0 = Not Hate Speech, 1 = Ambiguous, 2 = Hate Speech\n\n"
    "Provide a concise but complete justification INSIDE the JSON only (2–4 sentences).\n"
    "Do NOT output any text outside the required block.\n\n"
    "Text to evaluate (read-only):\n<<<\n{text}\n>>>\n\n"
    "OUTPUT REQUIREMENTS:\n"
    " - Return ONLY one <output> block containing valid JSON.\n"
    " - Include BOTH opening <output> and closing </output> tags exactly as shown.\n"
    " - The JSON must include only two fields: \"hate_speech_score\" and \"justification\".\n"
    " - Do NOT write anything before <output> or after </output>.\n"
    "<output>\n"
    "{{\n"
    "  \"hate_speech_score\": <generated_score>,\n"
    "  \"justification\": <explain_your_reasoning_in_2_to_4_sentences>\n"
    "}}\n"
    "</output>\n"
)

rag_template = (
    "You are an annotator.\n"
    "You must NOT create, modify, or add any examples.\n\n"
    "<examples_read_only>\n{examples}\n</examples_read_only>\n\n"
    "Under consideration of the given examples, rate the degree of hate speech in the text by generating a score 0, 1, or 2.\n"
    "0 = Not Hate Speech, 1 = Ambiguous, 2 = Hate Speech\n\n"
    "Provide a concise but complete justification INSIDE the JSON only (2–4 sentences).\n"
    "Do NOT output any text outside the required block.\n\n"
    "Text to evaluate (read-only):\n<<<\n{text}\n>>>\n\n"
    "OUTPUT REQUIREMENTS:\n"
    " - Return ONLY one <output> block containing valid JSON.\n"
    " - Include BOTH opening <output> and closing </output> tags exactly as shown.\n"
    " - The JSON must include only two fields: \"hate_speech_score\" and \"justification\".\n"
    " - Do NOT write anything before <output> or after </output>.\n"
    " - Do NOT create any examples.\n"
    "<output>\n"
    "{{\n"
    "  \"hate_speech_score\": <generated_score>,\n"
    "  \"justification\": <explain_your_reasoning_in_2_to_4_sentences>\n"
    "}}\n"
    "</output>\n"
)
