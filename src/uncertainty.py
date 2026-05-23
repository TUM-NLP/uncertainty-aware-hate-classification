from torch.utils.data import DataLoader
from lm_polygraph.stat_calculators import GreedyProbsCalculator, GreedyAlternativesNLICalculator, EntropyCalculator
from lm_polygraph.estimators import ClaimConditionedProbability, MaximumSequenceProbability, MeanTokenEntropy, Perplexity
from lm_polygraph.estimators import MeanPointwiseMutualInformation,MeanConditionalPointwiseMutualInformation,PTrue
from lm_polygraph.stat_calculators import GreedyLMProbsCalculator, PromptCalculator
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.model import WhiteboxModel
import torch, time
from config import generation_config
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig


def init_whitebox_model(model_path,
                        cache_path="/home/ec2-user/SageMaker/huggingface_cache",
                       device='auto',
                       load_in_4bit=False):
    """
    Load a Hugging Face model + tokenizer and wrap it in a WhiteboxModel.

    Returns:
        model_adapter (WhiteboxModel), tokenizer
    """    
    # Convert config to GenerationParameters compatible format
    gen_params = {
        "temperature": generation_config.get("temperature", 1.0),
        "top_k": generation_config.get("top_k", 50),
        "top_p": generation_config.get("top_p", 1.0),
        "do_sample": generation_config.get("do_sample", False),
        "num_beams": generation_config.get("num_beams", 1),
        "presence_penalty": generation_config.get("presence_penalty", 0.0),
        "repetition_penalty": generation_config.get("repetition_penalty", 1.0),
    }
    generation_params = GenerationParameters(**gen_params)
    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map= device, #'cuda:0',
            torch_dtype=torch.float16,
            cache_dir=cache_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit, #True,
                bnb_4bit_compute_dtype=torch.float16
            ),
            trust_remote_code=True,
        )
        base_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_path,
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.pad_token_id = tokenizer.pad_token_id

        # Wrap in WhiteboxModel
        model = WhiteboxModel(
            base_model,
            tokenizer,
            model_path=model_path,
            generation_parameters=generation_params,
        )
    else:
        raise RuntimeError("No GPUs available.")

    return model, tokenizer


class UncertaintyPipeline:
    def __init__(self, model_adapter, tokenizer, device="cuda"):
        self.model_adapter = model_adapter
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = generation_config.get("max_new_tokens", 100)

        # Initialize calculators
        self.calc_greedy_probs = GreedyProbsCalculator(output_attentions=False)

        nli_model = Deberta(device=device)
        nli_model.setup()
        self.calc_nli = GreedyAlternativesNLICalculator(nli_model=nli_model)

        self.calc_entropy = EntropyCalculator()
        self.calc_greedylm = GreedyLMProbsCalculator()
        self.calc_prompt = PromptCalculator()
        
        # Convert config to GenerationParameters compatible format
        gen_params = {
            "temperature": generation_config.get("temperature", 1.0),
            "top_k": generation_config.get("top_k", 50),
            "top_p": generation_config.get("top_p", 1.0),
            "do_sample": generation_config.get("do_sample", False),
            "num_beams": generation_config.get("num_beams", 1),
            "presence_penalty": generation_config.get("presence_penalty", 0.0),
            "repetition_penalty": generation_config.get("repetition_penalty", 1.0),
        }
        self.generation_params = GenerationParameters(**gen_params)

        # Initialize estimators
        self.estimators = {
            "ccp": ClaimConditionedProbability(),
            "msp": MaximumSequenceProbability(),
            "perplexity": Perplexity(),
            "mte": MeanTokenEntropy(),
            "mpmi": MeanPointwiseMutualInformation(),
            "mcpmi": MeanConditionalPointwiseMutualInformation(),
            "ptrue": PTrue(),
        }
        
    def get_uncertainty_scores(self, prompts, batch_size: int = 4):
        results = []
        batch_meta = []

        # Wrap prompts with chat template
        '''
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        #chat_messages = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        chat_messages = [self.tokenizer.apply_chat_template(
                            m,
                            add_generation_prompt=True,
                            tokenize=False,
                            )
                        for m in messages
                        ]

        data_loader = DataLoader(chat_messages, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
        '''
        data_loader = DataLoader(prompts, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

        
        with torch.inference_mode():
            for batch_id, batch in enumerate(data_loader):
                if torch.cuda.is_available(): 
                    torch.cuda.synchronize()
                t0 = time.time()
    
                deps = {"input_texts": batch}

                deps.update(self.calc_greedy_probs(deps, texts=batch, model=self.model_adapter, max_new_tokens=self.max_new_tokens))
                deps.update(self.calc_nli(deps, texts=batch, model=self.model_adapter))
                deps.update(self.calc_entropy(deps, texts=batch, model=self.model_adapter))
                deps.update(self.calc_greedylm(deps, texts=batch, model=self.model_adapter))
                deps.update(self.calc_prompt(deps, texts=batch, model=self.model_adapter))
    
                scores = {name: est(deps) for name, est in self.estimators.items()}
                generated_texts = self.tokenizer.batch_decode(deps["greedy_tokens"])

                input_lens = [len(ids) for ids in deps.get("input_tokens", [[]] * len(batch))]
                gen_lens   = [len(ids) for ids in deps.get("greedy_tokens", [[]] * len(batch))]
    
                if torch.cuda.is_available(): 
                    torch.cuda.synchronize()
                t1 = time.time()
                batch_seconds = t1 - t0
    
                for i, prompt in enumerate(batch):
                    results.append({
                        "input": prompt,
                        "output": generated_texts[i],
                        "ccp": scores["ccp"][i],
                        "msp": scores["msp"][i],
                        "perplexity": scores["perplexity"][i],
                        "mte": scores["mte"][i],
                        "mpmi": scores["mpmi"][i],
                        "mcpmi": scores["mcpmi"][i],
                        "ptrue": scores["ptrue"][i],
                        "timing": {"batch_id": batch_id, "batch_seconds": round(batch_seconds, 4)},
                        "lengths": {"input_tokens": input_lens[i], "gen_tokens": gen_lens[i]},
                    })
    
                batch_meta.append({
                    "batch_id": batch_id,
                    "num_items": len(batch),
                    "seconds": round(batch_seconds, 4),
                    "total_input_tokens": int(sum(input_lens)),
                    "total_gen_tokens": int(sum(gen_lens)),
                })
                if batch_id % 10 == 0:
                    print(f"Batch id:{batch_id} is processed.")
    
        return results, batch_meta