Markdown# Enterprise Financial Compliance RAG Pipeline

An asynchronous, privacy-first Retrieval-Augmented Generation (RAG) framework optimized for parsing and verifying dense corporate 10-K financial records (Apple Inc. & NVIDIA Corp.). 

This architecture is specifically engineered to deliver production-grade performance within strict, resource-constrained environments, running completely local on an **Intel i5 CPU** with a **6GB System RAM limit** and **$0.00 cloud infrastructure costs**.

---

## 🏗️ Technical Architecture Highlights

- **Multi-Threaded Asynchronous Retrieval:** Utilizes Python's `asyncio` engine to distribute heavy database lookups across background worker threads, preventing CPU stalling.
- **Dynamic Semantic Caching:** Integrates an fast cache lookup gate within Qdrant to intercept repeating compliance requests in under a millisecond.
- **Three-Tier Local AI Stack:** Decouples heavy financial text synthesis, query splitting routing, and dense token embeddings to maximize matrix computation speeds on minimal memory footprints.
- **Local Verification Suite:** Avoids memory-heavy deep-learning evaluation tools by utilizing a deterministic programmatic matrix to track factual precision.

---

## 📂 Complete Project Directory Map

Your local repository layout is organized using a strict decoupling pattern to separate ingestion data, system configuration mappings, processing layers, and evaluation outputs:

```text
financial_compliance_rag/
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Core configuration & local environment registry
│
├── data/
│   └── source_documents/          # Ingress storage folder for target 10-K PDFs
│       ├── apple_2025_10k.pdf
│       └── nvidia_2025_10k.pdf
│
├── src/
│   ├── __init__.py
│   │
│   ├── exceptions/
│   │   ├── __init__.py
│   │   └── governance.py          # Custom corporate compliance exception barriers
│   │
│   ├── ingestion/                 # Core processing & RAG architecture
│   │   ├── __init__.py
│   │   ├── generator.py           # Text synthesis, prompts, & cache synchronization
│   │   ├── query_validator.py     # Ingress validation & query deconstruction
│   │   ├── retriever.py           # Qdrant vector extraction & snippet stitching
│   │   └── semantic_cache.py      # Local vector-space caching layer
│   │
│   └── evaluation/                # Performance tracking metrics
│       ├── __init__.py
│       └── local_evaluate.py      # Custom CPU/RAM optimized validation script
│
├── .env                           # Local environmental variables & path definitions
├── requirements.txt               # Locked project dependency manifest
├── flush_cache.py                 # Administrative utility to clear stale Qdrant cache vectors
└── local_compliance_report.csv    # Generated evaluation report matrix output

🤖 Local AI Stack Requirements (Optimized for 6GB RAM)This system decouples processing stages into three specialized local micro-models to protect your system from memory thrashing:
Pipeline Component Layer    Model Target Specification      Size on Disk        Operational Role
Operational Generation      qwen2.5:1.5b                    ~986 MB             Handles complex text synthesis, prompt templates, and citation formatting.
Intelligent Routing         qwen2.5:0.5b                    ~390 MB             Powers lightning-fast query deconstruction and splitting on raw CPU cycles.
Vector Space Embeddings     BAAI/bge-small-en-v1.5          ~130 MB             Computes high-precision financial text embeddings natively.


Pulling the Weights via OllamaEnsure you have Ollama installed locally. 
Open your command shell terminal and download the required inference layers to your machine:
Bash
# Pull the core synthesis text generation engine
ollama pull qwen2.5:1.5b

# Pull the lightweight query-splitting router model
ollama pull qwen2.5:0.5b
Note: Your vector workspace natively handles the loading and optimization weights for the BAAI/bge-small-en-v1.5 transformer array during database indexing loops.

⚡ Step-by-Step Environment SetupFollow these steps to configure your virtual machine environment and run the pipeline locally:
1. Initialize the Workspace Scope & Virtual EnvironmentOpen your command terminal inside your project directory and run:
Bash
# Navigate to your local project directory path
cd /d :Project\financial_compliance_rag

# Initialize an isolated Python virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment frame
venv\Scripts\activate

2. Configure Local Settings Manifest (config/settings.py)Ensure your internal pipeline context loads your specific local hardware registry parameters:Python# Local AI Stack Components (Optimized for 6GB RAM)
OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_MODEL: str = "qwen2.5:1.5b"          # For heavy synthesis answering
ROUTER_MODEL: str = "qwen2.5:0.5b"       # For lightning-fast query splitting
EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

3. Install Project DependenciesVerify that your local requirements.txt file is saved in the root folder, then run the installation command:
Bash
python -m pip install --upgrade pip
pip install -r requirements.txt

4. Administrative Step: Purging/Flushing Stale Semantic Cache Records
If your evaluation suite returns outdated figures or you need to reset the system's warm memory vectors back to a clean state, execute the hardcoded cache eviction utility script:

Bash
python flush_cache.py

This command targets your local Qdrant collection database directly, deleting stale entry indices safely without impacting any structural source documents.

5. Running the Local Evaluation Suite
Execute the metrics tracking matrix using the terminal path shortcut:
Bash
python -m src.evaluation.local_evaluate

📊 Sample Execution Log ReportPlaintext🚀 Initializing 100% Local CPU Evaluation Stream...
📦 Hardware Profile: Optimized for 6GB RAM / Pure CPU Context

Evaluating Target Query: 'Extract the full consolidated total net sales reported by Apple Inc for the fiscal year 2025.'
--- Processing Ingress Query [Role: COMPLIANCE_AUDITOR] ---
Validator optimization framework fallback. Defaulting to raw pass-through.

🏆 LOCAL PIPELINE PERFORMANCE REPORT:
------------------------------------------------------------------------------------------------------------------------
| Question                                                                                     | Generated Answer                                                                                           | Factual Accuracy   |
|:----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:-------------------|
| Extract the full consolidated total net sales reported by Apple Inc for the fiscal year 2025. | The full consolidated total net sales reported by Apple Inc. for the fiscal year 2025 is $416,161 million. | 100.0%             |
| What is the complete total revenue reported by NVIDIA Corporation for the fiscal year 2025?   | The complete total revenue reported by NVIDIA Corporation for the fiscal year 2025 is $11,716 million.     | 66.7%              |
------------------------------------------------------------------------------------------------------------------------
📝 Metrics successfully locked to 'local_compliance_report.csv'!