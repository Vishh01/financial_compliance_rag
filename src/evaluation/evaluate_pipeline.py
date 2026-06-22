import os
import sys
import asyncio
import pandas as pd
from datasets import Dataset

# 1. Force strict absolute tracking of your repository directory boundaries
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. Complete clean Ragas evaluation imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper  # Added for local embedding override

# 3. Import Langchain's direct Ollama integration
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings  # Added to fetch local vectors

# 4. Import your pipeline's actual orchestrator class name
from src.ingestion.generator import AsyncComplianceRAGPipeline

# Set up test verification records 
EVALUATION_SET = [
    {
        "question": "Extract the full consolidated total net sales reported by Apple Inc for the fiscal year 2025.",
        "ground_truth": "$416,161 million reported for the fiscal year ending September 27, 2025."
    },
    {
        "question": "What is the complete total revenue reported by NVIDIA Corporation for the fiscal year 2025?",
        "ground_truth": "$130,497 million reported for the fiscal year ending January 26, 2025."
    }
]

async def run_local_ragas_evaluation():
    print("🚀 Initializing Ground Truth Evaluation Stream on Local Hardware...")
    print(f"📦 Workspace mapped dynamically to root scope: {PROJECT_ROOT}")
    
    # Initialize the local critic judge and local embedding models using LangChain's Ollama wrappers
    chat_ollama_model = ChatOllama(
        model="qwen2.5:0.5b", 
        base_url="http://localhost:11434"
    )
    # Set the model target to your newly pulled local nomic embedding model
    ollama_embed_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Wrap both models cleanly for the Ragas evaluator engine
    local_judge_llm = LangchainLLMWrapper(chat_ollama_model)
    local_judge_embeddings = LangchainEmbeddingsWrapper(ollama_embed_model)
    
    # Initialize your actual class layout
    pipeline = AsyncComplianceRAGPipeline()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # Process queries sequentially through your high-end pipeline
    for item in EVALUATION_SET:
        print(f"\nEvaluating Target Query: '{item['question']}'")
        print("⏳ Processing heavy-compute RAG matrix blocks (including Cross-Encoders and Stitching)...")
        
        # Extract the answer string
        generated_answer = await pipeline.query(
            user_question=item["question"], 
            user_role="compliance_auditor"
        )
        
        # Extract supporting context chunks independently via your retriever class 
        retrieval_response = pipeline.retriever.retrieve_context(
            query=item["question"], 
            user_role="compliance_auditor", 
            limit=2
        )
        
        retrieved_chunks = []
        if isinstance(retrieval_response, dict) and "context" in retrieval_response:
            retrieved_chunks = [node["text"] for node in retrieval_response["context"]]
        
        questions.append(item["question"])
        answers.append(generated_answer)
        contexts.append(retrieved_chunks)
        ground_truths.append(item["ground_truth"])
        
    # Bind into Hugging Face structural dictionary layers
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("\n📊 Calculating Ragas Scores Matrix via Local Ollama Critic...")
    
    # Pass both the local LLM wrapper AND the local Embeddings wrapper
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=local_judge_llm,
        embeddings=local_judge_embeddings
    )
    
    df = result.to_pandas()
    print("\n🏆 HIGH-END PIPELINE PERFORMANCE REPORT:")
    print("---")
    print(df[["question", "faithfulness", "answer_relevancy", "context_recall"]].to_markdown(index=False))
    print("---")
    
    df.to_csv("ragas_compliance_report.csv", index=False)
    print("\n📝 Metrics locked successfully to 'ragas_compliance_report.csv'")

if __name__ == "__main__":
    asyncio.run(run_local_ragas_evaluation())