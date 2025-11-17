import os
import json
import numpy as np
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from datetime import datetime

# -------------------------
# Configuration
# -------------------------
load_dotenv(dotenv_path="environment.env")

# Paths - Updated to use chunks_all.json structure
EMBEDDINGS_PATH = "embeddings_store/embeddings.npy"
CHUNKS_PATH = "chunks_by_filetype_new_test/chunks_all_test.json"  # Changed from chunks.json
METADATA_PATH = "chunks_by_filetype_new/metadata_test.json"
STATS_PATH = "embeddings_store/embedding_stats.json"

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "placement-companion-v5")

# Upload settings
BATCH_SIZE = 100  # Pinecone recommends 100-200
NAMESPACE = "placement-docs"  # Optional: organize by namespace

# -------------------------
# Load Data
# -------------------------
def load_data():
    """Load embeddings, chunks, and metadata from new structure"""
    print("="*60)
    print("PINECONE UPLOAD PIPELINE")
    print("="*60)
    print("\n📂 Loading data...")
    
    # Validate files exist
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"❌ Embeddings not found: {EMBEDDINGS_PATH}\n   Run: python embeddings.py")
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"❌ Chunks not found: {CHUNKS_PATH}\n   Run: python chunks_generation.py")
    
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"✅ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D)")
    
    # Load chunks from chunks_all.json
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    # Flatten company-wise chunks into single list
    chunks = []
    chunks_dict = chunks_data.get("chunks", {})
    
    for company, chunk_list in chunks_dict.items():
        if isinstance(chunk_list, list):
            chunks.extend(chunk_list)
    
    print(f"✅ Loaded {len(chunks)} chunks from {len(chunks_dict)} companies")
    
    # Load global metadata
    global_metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            global_metadata = json.load(f)
        print(f"✅ Loaded global metadata")
        print(f"   • Total companies: {global_metadata.get('total_companies', 0)}")
        print(f"   • Recurring companies: {len(global_metadata.get('recurring_companies', []))}")
    
    # Load embedding stats
    embedding_stats = {}
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            embedding_stats = json.load(f)
        print(f"✅ Model: {embedding_stats.get('model_name', 'Unknown')}")
        print(f"✅ Dimension: {embedding_stats.get('embedding_dimension', embeddings.shape[1])}")
    
    # Sanity check
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"❌ Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks\n"
            f"   Delete embeddings_store/ and re-run: python embeddings.py"
        )
    
    return embeddings, chunks, global_metadata, embedding_stats

# -------------------------
# Prepare Metadata for Pinecone
# -------------------------
def prepare_metadata(chunk, chunk_idx):
    """
    Prepare metadata for Pinecone.
    Pinecone limitations:
    - Max 40KB per vector metadata
    - Only supports: string, number, boolean, list of strings
    """
    metadata = {
        # IDs
        "chunk_id": chunk.get("chunk_id", chunk_idx),
        "global_chunk_id": chunk.get("global_chunk_id", chunk_idx),
        
        # Basic info
        "company": chunk.get("company", "Unknown"),
        "role": chunk.get("role", "General"),
        "section": chunk.get("section", "Other"),
        "filename": chunk.get("filename", ""),
        "file_type": chunk.get("file_type", ""),
        
        # Content (truncate to stay under 40KB limit)
        "content": chunk.get("content", "")[:4000],  # ~4KB for content
        
        # Company details (from enhanced metadata)
        "year": chunk.get("company_year"),
        "cgpa": chunk.get("company_cgpa"),
        "is_recurring": chunk.get("is_recurring", False),
        
        # Compensation
        "stipend": chunk.get("company_stipend"),
        "ctc": chunk.get("company_ctc"),
        "salary_type": chunk.get("salary_type"),
        
        # Branch eligibility (convert dict/list to string for Pinecone)
        "bda_eligible": str(chunk.get("bda_eligible", {})) if chunk.get("bda_eligible") else None,
        "aiml_eligible": str(chunk.get("aiml_eligible", {})) if chunk.get("aiml_eligible") else None,
        
        # Years visited (convert to comma-separated string if it's a list)
        "years_visited": ",".join(map(str, chunk.get("years_visited", []))) if chunk.get("years_visited") else None,
        
        # Classification confidence
        "confidence": float(chunk.get("confidence", 0.0)),
        
        # Upload timestamp
        "upload_timestamp": datetime.now().isoformat()
    }
    
    # Remove None/empty values (Pinecone doesn't accept None)
    metadata = {
        k: v for k, v in metadata.items() 
        if v is not None and v != "" and v != [] and v != {}
    }
    
    return metadata

# -------------------------
# Initialize Pinecone
# -------------------------
def init_pinecone(embedding_dim):
    """Initialize Pinecone and create/connect to index"""
    print(f"\n🔌 Connecting to Pinecone...")
    
    if not PINECONE_API_KEY:
        raise ValueError("❌ PINECONE_API_KEY not found in environment.env file")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = {idx.name for idx in pc.list_indexes()}
    
    if INDEX_NAME not in existing_indexes:
        print(f"🆕 Creating new index: {INDEX_NAME}")
        print(f"   • Dimension: {embedding_dim}")
        print(f"   • Metric: cosine")
        print(f"   • Cloud: {PINECONE_CLOUD} ({PINECONE_REGION})")
        
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",  # Cosine similarity for normalized embeddings
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        print(f"✅ Index created successfully")
        time.sleep(5)  # Wait for index to be ready
    else:
        print(f"📡 Connecting to existing index: {INDEX_NAME}")
        
        # Verify dimension matches
        index_info = pc.describe_index(INDEX_NAME)
        existing_dim = index_info.dimension
        if existing_dim != embedding_dim:
            raise ValueError(
                f"❌ Dimension mismatch!\n"
                f"   Existing index: {existing_dim}D\n"
                f"   New embeddings: {embedding_dim}D\n"
                f"   Solution: Delete index or use different INDEX_NAME"
            )
    
    # Connect to index
    index = pc.Index(INDEX_NAME)
    
    # Get current stats
    stats = index.describe_index_stats()
    print(f"📊 Current index stats:")
    print(f"   • Dimension: {stats.get('dimension', 'N/A')}")
    print(f"   • Total vectors: {stats.get('total_vector_count', 0)}")
    
    # Optional: Clear existing data
    if stats.get('total_vector_count', 0) > 0:
        clear = input(f"\n⚠️  Index has {stats['total_vector_count']} vectors. Clear before upload? (y/N): ").strip().lower()
        if clear == 'y':
            print("🧹 Clearing existing vectors...")
            index.delete(delete_all=True, namespace=NAMESPACE)
            time.sleep(2)
            print("✅ Index cleared")
    
    return index

# -------------------------
# Upload to Pinecone
# -------------------------
def upload_to_pinecone(index, embeddings, chunks):
    """Upload embeddings and metadata to Pinecone in batches"""
    print(f"\n🚀 Uploading {len(embeddings)} vectors to Pinecone...")
    print(f"⚙️ Batch size: {BATCH_SIZE}")
    print(f"⚙️ Namespace: {NAMESPACE}")
    
    uploaded = 0
    failed = 0
    total_batches = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(embeddings), BATCH_SIZE), total=total_batches, desc="Uploading"):
        batch_end = min(i + BATCH_SIZE, len(embeddings))
        
        # Prepare batch
        vectors = []
        for idx in range(i, batch_end):
            # Use global_chunk_id for unique vector ID
            vector_id = f"chunk_{chunks[idx].get('global_chunk_id', idx)}"
            
            # Convert numpy array to list
            embedding = embeddings[idx].tolist()
            
            # Prepare metadata
            metadata = prepare_metadata(chunks[idx], idx)
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert batch to Pinecone
        try:
            index.upsert(
                vectors=vectors,
                namespace=NAMESPACE
            )
            uploaded += len(vectors)
        except Exception as e:
            failed += len(vectors)
            print(f"\n⚠️ Error in batch {i//BATCH_SIZE + 1}: {e}")
            # Continue with next batch instead of stopping
            continue
    
    print(f"\n✅ Upload complete!")
    print(f"   • Successfully uploaded: {uploaded}/{len(embeddings)}")
    if failed > 0:
        print(f"   • Failed: {failed} vectors")
    
    return uploaded, failed

# -------------------------
# Verify Upload
# -------------------------
def verify_upload(index):
    """Verify upload and run test query"""
    print("\n🔍 Verifying upload...")
    time.sleep(3)  # Wait for index to update
    
    stats = index.describe_index_stats()
    
    print(f"\n📊 Final Index Statistics:")
    print(f"   • Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"   • Dimension: {stats.get('dimension', 'N/A')}")
    
    # Show namespace stats
    if 'namespaces' in stats:
        for ns, ns_stats in stats['namespaces'].items():
            print(f"   • Namespace '{ns}': {ns_stats.get('vector_count', 0)} vectors")
    
    # Test query with dummy vector
    print("\n🧪 Testing query...")
    try:
        dim = stats.get('dimension', 3584)
        dummy_vector = [0.1] * dim  # Simple test vector
        
        results = index.query(
            vector=dummy_vector,
            top_k=3,
            namespace=NAMESPACE,
            include_metadata=True
        )
        
        if results.get('matches'):
            print(f"✅ Query successful! Found {len(results['matches'])} results")
            
            # Show top result
            match = results['matches'][0]
            print(f"\n📝 Sample result:")
            print(f"   • ID: {match.get('id')}")
            print(f"   • Score: {match.get('score', 0):.4f}")
            print(f"   • Company: {match.get('metadata', {}).get('company', 'N/A')}")
            print(f"   • Section: {match.get('metadata', {}).get('section', 'N/A')}")
            print(f"   • Year: {match.get('metadata', {}).get('year', 'N/A')}")
        else:
            print("⚠️ No results found in test query")
            
    except Exception as e:
        print(f"⚠️ Test query failed: {e}")

# -------------------------
# Main Execution
# -------------------------
def main():
    try:
        # Step 1: Load data
        embeddings, chunks, global_metadata, embedding_stats = load_data()
        
        # Step 2: Initialize Pinecone
        embedding_dim = embedding_stats.get('embedding_dimension', embeddings.shape[1])
        index = init_pinecone(embedding_dim)
        
        # Step 3: Upload
        uploaded, failed = upload_to_pinecone(index, embeddings, chunks)
        
        # Step 4: Verify
        verify_upload(index)
        
        # Summary
        print("\n" + "="*60)
        print("✅ PINECONE UPLOAD COMPLETE!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   • Total chunks: {len(chunks)}")
        print(f"   • Uploaded: {uploaded}")
        print(f"   • Failed: {failed}")
        print(f"   • Companies: {global_metadata.get('total_companies', 0)}")
        print(f"   • Recurring: {len(global_metadata.get('recurring_companies', []))}")
        
        print(f"\n📌 Next steps:")
        print(f"   1. Test queries: python test_pinecone_query.py")
        print(f"   2. Build chatbot: python chatbot.py")
        print(f"   3. Create UI: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
