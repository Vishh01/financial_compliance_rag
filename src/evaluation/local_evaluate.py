import os
import sys
import asyncio
import pandas as pd

# Force strict absolute tracking of your repository directory boundaries
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.generator import AsyncComplianceRAGPipeline

EVALUATION_SET = [
    {
        "question": "Extract the full consolidated total net sales reported by Apple Inc for the fiscal year 2025.",
        "expected_keywords": ["416,161", "2025", "apple"],
    },
    {
        "question": "What is the complete total revenue reported by NVIDIA Corporation for the fiscal year 2025?",
        "expected_keywords": ["130,497", "2025", "nvidia"],
    }
]

def calculate_local_accuracy(generated_answer, expected_keywords):
    """Calculates what percentage of critical financial data points were extracted successfully."""
    answer_lower = generated_answer.lower()
    matched_count = 0
    
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            matched_count += 1
            
    return matched_count / len(expected_keywords)

async def run_pure_local_evaluation():
    print("🚀 Initializing 100% Local CPU Evaluation Stream...")
    print(f"📦 Hardware Profile: Optimized for 6GB RAM / Pure CPU Context")
    
    pipeline = AsyncComplianceRAGPipeline()
    results = []
    
    for item in EVALUATION_SET:
        print(f"\nEvaluating Target Query: '{item['question']}'")
        
        # Force the pipeline to refresh its data context vectors
        generated_answer = await pipeline.query(
            user_question=item["question"], 
            user_role="compliance_auditor",
            bypass_cache=True  # Overwrites stale cache records with real runtime data
        )
        
        # Calculate local factual accuracy score
        accuracy_score = calculate_local_accuracy(generated_answer, item["expected_keywords"])
        
        results.append({
            "Question": item["question"],
            "Generated Answer": generated_answer,
            "Factual Accuracy": f"{accuracy_score * 100:.1f}%"
        })
        
    # Print results to terminal
    df = pd.DataFrame(results)
    print("\n🏆 LOCAL PIPELINE PERFORMANCE REPORT:")
    print("---")
    print(df.to_markdown(index=False))
    print("---")
    
    df.to_csv("local_compliance_report.csv", index=False)
    print("\n📝 Metrics successfully locked to 'local_compliance_report.csv'!")

if __name__ == "__main__":
    asyncio.run(run_pure_local_evaluation())