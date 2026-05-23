# uncertainty-aware-hate-classification

This project implements and compares three distinct approaches for hate speech detection using Large Language Models (LLMs), followed by uncertainty quantification analysis to measure prediction quality and reliability.

1. Basic Prompting: Direct zero-shot prompting without additional context or examples.
2. Pesona Prompting: Incorporates demographic information (age, gender, race, education) to provide annotator perspective context to the model.
3. Few-shot Context Aware prompting: Leverages Amazon Bedrock Knowledge Bases to retrieve semantically similar annotated examples, providing relevant context for improved classification.

Then, calculates Uncertainty Quantification Metrics as a quality measurement.

## Prerequisites
- AWS Account with appropriate permissions
- SSH client installed on your local machine
- Python == 3.11


## Installation
### 1. Clone the repository
```bash
git clone git@github.com:TUM-NLP/uncertainty-aware-hate-classification.git
```

### 2. Create Test and Knowledge Base Datasets
The project uses the Measuring Hate Speech dataset (MHS) and CRoss-cultural English Hate speech (CREHate) dataset from Hugging Face, formatted as follows:

Text: [comment text]
Hate Speech Score: [0, 1, or 2]
Where:

0: Not hate speech
1: Ambiguous
2: Hate speech

The dataset is curated in two parts, seperately for both MHS and CREHate datasets.
- Test Dataset: 500 annotated comments for evaluation
- Knowledge Base Dataset for Bedrock: 1000 annotated comments with annotator demographics metadata (age, gender, race, education)


### 3. Set Up Amazon Bedrock Knowledge Base
The Annotation-Grounded Few-Shot Prompting approach requires an Amazon Bedrock Knowledge Base for the semantic retrieval of similar annotated examples.

#### 1.Prepare your data source

- Upload your Knowledge Base dataset (1,000 samples) to Amazon S3
- Ensure proper formatting with text, labels, and metadata

#### 2.Create Knowledge Base
- Navigate to Amazon Bedrock Console → Knowledge bases
- Click "Create knowledge base"
- Configure embedding model (e.g., Amazon Titan Embeddings)
- Connect your S3 data source

#### 3.Data Source and index

- Initiate data sync to ingest and index your documents
- Wait for completion (typically several minutes)
- Test retrieval with sample queries

#### 4. Update config file
- Copy your Knowledge Base ID and Data Source ID
- Update config.py with relevant information
For detailed instructions, refer to the docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html


## GPU Setup Option
This repository requires a GPU instance and you can follow these steps to set up and AWS EC2 instance to run this code. 

### 1. Launch EC2 Instance
1. Navigate to the console.aws.amazon.com/ec2/
2. Click **"Launch instance"**
3. Give your instance a name (e.g., "ml-gpu-instance")

### 2. Select AMI
- Choose **Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)**
- This AMI comes pre-configured with CUDA toolkit and NVIDIA drivers

### 3. Choose Instance Type
- Select your preferred instance type. 
- **g5.2xlarge** instance used in the development of this work.
- Specifications: 1x NVIDIA A10G GPU, 8 vCPUs, 32 GB RAM

### 4. Configure Key Pair
1. Create a new key pair or select existing
2. Download the `.pem` file and store securely
3. Set proper permissions: `chmod 400 your-key.pem`

### 5. Network Settings (Security Group)
Configure inbound rules:
- **SSH (Port 22)**: Source = "My IP" (for secure access)
- **Jupyter (Port 8888)**: Source = "My IP" (for notebook access)

### 6. Configure Storage
- Change default 30 GiB to **100 GiB** (or more for large datasets)
- Keep volume type as **gp3** (cost-effective SSD)

### 7. Launch Instance
- Review settings and click **"Launch instance"**
- Wait 2-3 minutes for instance to initialize

## Connecting to Your Instance

### Fix Key Permissions (First Time Only)
```bash
chmod 400 /path/to/your-key.pem
```

### SSH Connection
```bash
ssh -i /path/to/your-key.pem ec2-user@your-instance-dns
```

### Verify GPU Access
Once connected, verify GPU is available:
```bash
nvidia-smi
```

## Instance Details
- **Instance Type**: g5.2xlarge
- **GPU**: NVIDIA A10G (24GB GPU memory)
- **AMI**: Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)
- **Default User**: ec2-user

## Cost Considerations
- Remember to **stop** or **terminate** your instance when not in use
- g5.2xlarge instances incur charges while running
- Use AWS Cost Explorer to monitor spending


# Environment Setup

## Install Python 3.11
```bash
sudo dnf install python3.11 -y
```

## Create and Activate Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

## Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

---

## Transfer Project to EC2 Instance

From your **local machine**, run this command to sync your code repository to the EC2 instance:

```bash
rsync -av -e "ssh -i /path/to/your-key.pem" \
  --exclude 'venv' \
  --exclude '.venv' \
  --exclude 'output' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  /path/to/your/uncertainty-aware-hate-classification/ \
  ec2-user@your-instance-dns:~/uncertainty-aware-hate-classification/
```

---

## Run Your Code on EC2 instance

```bash
# Navigate to project directory
cd ~/uncertainty-aware-hate-classification/

# Activate virtual environment
source ~/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

---

## Download Results from EC2 instance

From your **local machine**, run this command to retrieve the output saved into your EC2 instance:

```bash
scp -i /path/to/your-key.pem -r \
  ec2-user@your-instance-public-dns:~/uncertainty-aware-hate-classification/output/ \
  ./
```
