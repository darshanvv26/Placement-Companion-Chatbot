import os
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path="environment.env")

# Select a free GPU (GPU 1 in your case)
if torch.cuda.is_available():
    # Check available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"📊 Available GPUs: {gpu_count}")
    
    # List all GPUs
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Try to use GPU 1 if available, otherwise GPU 0
    gpu_id = 1 if gpu_count > 1 else 0
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"✅ Using GPU {gpu_id}: {torch.cuda.get_device_name(device)}")
    
    # Show GPU memory
    free_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    print(f"   Total Memory: {free_mem:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU.")

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "placement-companion-v5")
NAMESPACE = "placement-docs"
MODEL_NAME = 'Alibaba-NLP/gte-Qwen2-7B-instruct'

# Initialize Pinecone
print("\n🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load embedding model on the chosen GPU
print("🧠 Loading embedding model...")
model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    device=device
)
print(f"✅ Model loaded on {device}")

def query_pinecone(query_text, top_k=5, filters=None):
    """Query Pinecone with semantic search"""
    
    # Format query for GTE-Qwen2
    instruction = (
        "Given a placement document containing company information, "
        "job descriptions, eligibility criteria, compensation details, "
        "and selection processes, retrieve relevant information"
    )
    formatted_query = f"Instruct: {instruction}\nQuery: {query_text}"
    
    # Generate query embedding
    query_embedding = model.encode(
        formatted_query,
        normalize_embeddings=True,
        prompt_name="query"
    ).tolist()
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
        filter=filters
    )
    
    return results

# Test queries
test_queries = [
    "What is the eligibility criteria for Google?",
    "What companies offer more than 15 LPA CTC?",
    "Tell me about Amazon internship stipend",
    "Which companies visited in 2024?",
    "What is the selection process for Microsoft?"
]

print("\n" + "="*60)
print("TESTING PINECONE QUERIES")
print("="*60)

for i, query in enumerate(test_queries, 1):
    print(f"\n🔍 Query {i}: {query}")
    print("-" * 60)
    
    results = query_pinecone(query, top_k=3)
    
    if results.get('matches'):
        for j, match in enumerate(results['matches'], 1):
            meta = match.get('metadata', {})
            print(f"\n   Result {j} (Score: {match['score']:.4f}):")
            print(f"   • Company: {meta.get('company', 'N/A')}")
            print(f"   • Section: {meta.get('section', 'N/A')}")
            print(f"   • Year: {meta.get('year', 'N/A')}")
            if meta.get('ctc'):
                print(f"   • CTC: ₹{meta['ctc']} LPA")
            if meta.get('stipend'):
                print(f"   • Stipend: ₹{meta['stipend']}K/month")
            print(f"   • Content: {meta.get('content', '')[:150]}...")
    else:
        print("   ⚠️ No results found")

print("\n✅ Query testing complete!")