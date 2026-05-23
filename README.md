# Uncertainty-Aware Hate Speech Classification

This repository contains the code for our EMNLP 2026 paper. We investigate how different prompting strategies affect both hate speech classification performance and prediction reliability in Large Language Models (LLMs), using white-box uncertainty quantification to measure when the model "knows what it doesn't know."

## Overview

We compare three prompting strategies for hate speech detection and analyze their uncertainty profiles:

| Approach | Description |
|---|---|
| **Baseline** | Zero-shot prompting — the model classifies hate speech without any additional context |
| **Persona Prompting** | The model is given a synthetic annotator persona (age, gender, race, education) before classifying |
| **Annotation-Grounded Few-Shot** | Semantically similar annotated examples are retrieved from Amazon Bedrock Knowledge Bases and provided as few-shot context |

For each prediction, the pipeline computes seven uncertainty scores via [LM-Polygraph](https://github.com/IINemo/lm-polygraph):

| Metric | Abbreviation |
|---|---|
| Claim-Conditioned Probability | `ccp` |
| Maximum Sequence Probability | `msp` |
| Perplexity | `perplexity` |
| Mean Token Entropy | `mte` |
| Mean Pointwise Mutual Information | `mpmi` |
| Mean Conditional Pointwise Mutual Information | `mcpmi` |
| Probability of being True | `ptrue` |

## Datasets

The project uses two hate speech corpora:

- **MHS** — [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (UC Berkeley)
- **CREHate** — [CRoss-cultural English Hate Speech](https://huggingface.co/datasets/Babelscape/CREHate)

Labels are three-class:

| Score | Meaning |
|---|---|
| `0` | Not hate speech |
| `1` | Ambiguous |
| `2` | Hate speech |

Each dataset is split into:
- **Test set** (`df_test_ds1.pkl` / `df_test_ds2.pkl`): 500 annotated comments for evaluation
- **Knowledge base** (`df_knowledge_base_ds1.pkl` / `df_knowledge_base_ds2.pkl`): 1,000 annotated comments with full annotator demographic metadata (age, gender, race, education) used for few-shot retrieval

Both splits are stored as Pandas DataFrames in `.pkl` format and contain text content, hate speech labels, annotator IDs, and annotator demographic attributes.

## Repository Structure

```
uncertainty-aware-hate-classification/
├── config.py               # All experiment configuration (model, paths, generation params)
├── main.py                 # Entry point — runs all three approaches end-to-end
├── requirements.txt
├── data/
│   ├── df_test_ds1.pkl          # MHS test set
│   ├── df_test_ds2.pkl          # CREHate test set
│   ├── df_knowledge_base_ds1.pkl
│   └── df_knowledge_base_ds2.pkl
├── src/
│   ├── prompt.py           # Prompt construction for all three approaches
│   ├── evaluate.py         # Evaluation runners (baseline, persona, annotation-grounded)
│   ├── uncertainty.py      # Model loading and UncertaintyPipeline (LM-Polygraph integration)
│   ├── rag.py              # RAG orchestration
│   ├── rag_utils.py        # Bedrock Knowledge Base retrieval utilities
│   └── helpers.py          # Serialization helpers
└── utils/
    └── bedrock.py          # AWS Bedrock client utilities
```

## Prerequisites

- Python 3.11
- AWS account with permissions for Amazon Bedrock and S3
- CUDA-capable GPU (development used an NVIDIA A10G on a `g5.2xlarge` EC2 instance)
- A [Hugging Face](https://huggingface.co/) account with access to the target model (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:TUM-NLP/uncertainty-aware-hate-classification.git
cd uncertainty-aware-hate-classification
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Configure the experiment

Edit `config.py` and fill in the required fields:

```python
hf_login_token = '<your-huggingface-token>'
model_path     = 'meta-llama/Llama-3.1-8B-Instruct'  # or any compatible HF model

test_df_path   = 'data/df_test_ds1.pkl'               # switch to ds2 for CREHate
kb_df_path     = 'data/df_knowledge_base_ds1.pkl'

knowledge_base_id = '<your-bedrock-knowledge-base-id>'
data_source_id    = '<your-opensearch-data-source-id>'
region            = 'us-east-1'
```

Optional generation settings (greedy decoding is the default):

```python
load_in_4bit = False   # set True to quantize the model and save GPU memory
batch_size   = 16
```

### 4. Set Up the Amazon Bedrock Knowledge Base

The annotation-grounded approach requires a Bedrock Knowledge Base for semantic retrieval.

**a. Upload the knowledge base dataset to S3**

```bash
aws s3 cp data/df_knowledge_base_ds1.pkl s3://<your-bucket>/kb/
```

**b. Create the Knowledge Base in the AWS Console**

1. Go to **Amazon Bedrock → Knowledge bases → Create knowledge base**
2. Select an embedding model (e.g., *Amazon Titan Embeddings*)
3. Connect your S3 bucket as the data source
4. Start ingestion and wait for the sync to complete

**c. Copy the IDs into `config.py`**

```python
knowledge_base_id = '<KB-ID-from-console>'
data_source_id    = '<DataSource-ID-from-console>'
```

For full documentation see the [AWS Bedrock Knowledge Bases guide](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html).

## Running the Experiments

```bash
python main.py
```

`main.py` executes the full pipeline in sequence:

1. **Context retrieval** — queries Bedrock for each test instance and saves `output/contexts.json`
2. **Baseline evaluation** — zero-shot prompting + uncertainty scoring
3. **Persona evaluation** — demographic persona prompting + uncertainty scoring
4. **Annotation-grounded evaluation** — few-shot prompting with retrieved examples + uncertainty scoring

Results are written to the `output/` directory as timestamped JSON files:

```
output/
├── contexts.json
├── baseline_results_greedy_<timestamp>.json
├── persona_results_greedy_<timestamp>.json
└── annotation-grounded_results_greedy_<timestamp>.json
```

Each result file maps `comment_id → annotator_id → {text, prompt, generation_text, uncertainty{ccp, msp, perplexity, mte, mpmi, mcpmi, ptrue}}`.

## Running on AWS EC2 (Recommended)

### Launch an instance

1. Go to **EC2 → Launch instance**
2. Choose **Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)**
3. Select instance type — `g5.2xlarge` was used in this work (1× NVIDIA A10G, 8 vCPUs, 32 GB RAM)
4. Create or select a key pair; download and secure the `.pem` file:
   ```bash
   chmod 400 your-key.pem
   ```
5. Configure inbound rules: SSH (port 22) and Jupyter (port 8888) restricted to your IP
6. Set storage to at least **100 GiB (gp3)**

### Connect and install Python

```bash
ssh -i /path/to/your-key.pem ec2-user@<instance-dns>

# On the instance:
sudo dnf install python3.11 -y
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Transfer the repository

From your **local machine**:

```bash
rsync -av -e "ssh -i /path/to/your-key.pem" \
  --exclude 'venv' --exclude 'output' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude '.git' \
  /path/to/uncertainty-aware-hate-classification/ \
  ec2-user@<instance-dns>:~/uncertainty-aware-hate-classification/
```

### Run the pipeline

```bash
cd ~/uncertainty-aware-hate-classification
source ~/venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Download results

From your **local machine**:

```bash
scp -i /path/to/your-key.pem -r \
  ec2-user@<instance-dns>:~/uncertainty-aware-hate-classification/output/ ./
```

> **Cost note:** Stop or terminate your EC2 instance when not in use. Monitor spending with [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/).

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{...,
  title     = {...},
  author    = {...},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
}
```

## License

This project is licensed under the MIT License.
