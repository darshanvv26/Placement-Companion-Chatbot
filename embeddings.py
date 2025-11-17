import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Handles embedding generation for all chunk files.
    Reads from chunks_all.json (company-structured with metadata).
    """

    def __init__(self, model_name='Alibaba-NLP/gte-Qwen2-7B-instruct'):
        print(f"🧠 Loading embedding model: {model_name}")
        
        # GTE-Qwen2 requires specific configuration
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,  # Required for GTE-Qwen2
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Check model properties
        self.model_name = model_name
        self.is_gte_qwen = 'gte-qwen' in model_name.lower()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded on {self.model.device}")
        print(f"✅ Embedding dimension: {self.embedding_dim}")

        # Updated path to merged JSON file (from chunks_generation.py output)
        self.chunks_dir = r"chunks_by_filetype_new"  # Updated path
        self.combined_json = os.path.join(self.chunks_dir, "chunks_all.json")  # Changed filename
        self.metadata_json = os.path.join(self.chunks_dir, "metadata.json")

    # -------------------------
    # Load all chunks
    # -------------------------
    def load_all_chunks(self) -> tuple[List[Dict], Dict]:
        """Load chunks and metadata from chunks_all.json"""
        all_chunks: List[Dict] = []
        metadata = {}

        if not os.path.exists(self.combined_json):
            raise FileNotFoundError(f"❌ Combined chunks file not found: {self.combined_json}")

        try:
            with open(self.combined_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract metadata
            metadata = data.get("metadata", {})
            
            # The JSON is company-keyed (dict of lists)
            chunks_data = data.get("chunks", {})

            # Flatten company-wise chunks into one list with enhanced metadata
            chunk_id_counter = 0
            for company, chunks_list in chunks_data.items():
                if isinstance(chunks_list, list):
                    for chunk in chunks_list:
                        # Add global chunk ID for tracking
                        chunk["global_chunk_id"] = chunk_id_counter
                        chunk_id_counter += 1
                        
                        # Add company metadata if available
                        if company in metadata.get("company_details", {}):
                            company_meta = metadata["company_details"][company]
                            chunk["company_year"] = company_meta.get("year")
                            chunk["company_cgpa"] = company_meta.get("cgpa_requirement")
                            
                            # Add compensation info
                            comp = company_meta.get("compensation", {})
                            chunk["company_stipend"] = comp.get("stipend")
                            chunk["company_ctc"] = comp.get("ctc")
                            chunk["salary_type"] = comp.get("salary_type")
                            
                            # Add Excel info if available
                            excel_info = company_meta.get("excel_info", {})
                            if excel_info:
                                chunk["years_visited"] = excel_info.get("years_visited", [])
                                chunk["is_recurring"] = excel_info.get("is_recurring", False)
                                chunk["bda_eligible"] = excel_info.get("bda_eligible", {})
                                chunk["aiml_eligible"] = excel_info.get("aiml_eligible", {})
                        
                        all_chunks.append(chunk)

            print(f"📦 Loaded {len(all_chunks)} chunks from {self.combined_json}")
            print(f"📊 Metadata:")
            print(f"   • Total companies: {metadata.get('total_companies', 0)}")
            print(f"   • Recurring companies: {len(metadata.get('recurring_companies', []))}")
            if metadata.get('compensation', {}).get('avg_stipend'):
                print(f"   • Avg Stipend: {metadata['compensation']['avg_stipend']}")
            if metadata.get('compensation', {}).get('avg_ctc'):
                print(f"   • Avg CTC: {metadata['compensation']['avg_ctc']}")

        except Exception as e:
            print(f"⚠️ Error reading combined chunks: {e}")
            import traceback
            traceback.print_exc()

        return all_chunks, metadata

    # -------------------------
    # Enhanced Text Formatting
    # -------------------------
    def format_chunk_text(self, chunk: Dict, include_metadata: bool = True) -> str:
        """
        Format chunk text with enhanced context for better embeddings.
        """
        # Basic content
        section = chunk.get("section", "General")
        content = chunk.get("content", "")
        company = chunk.get("company", "Unknown")
        role = chunk.get("role", "General")
        
        # Build contextual text
        if include_metadata:
            # Add rich context
            context_parts = [f"Company: {company}"]
            
            # Add year if available
            if chunk.get("company_year"):
                context_parts.append(f"Year: {chunk['company_year']}")
            
            # Add role
            if role and role != "General":
                context_parts.append(f"Role: {role}")
            
            # Add section
            context_parts.append(f"Section: {section}")
            
            # Add compensation info if it's a compensation section
            if section == "Compensation":
                if chunk.get("company_stipend"):
                    context_parts.append(f"Stipend: ₹{chunk['company_stipend']}K/month")
                if chunk.get("company_ctc"):
                    context_parts.append(f"CTC: ₹{chunk['company_ctc']} LPA")
            
            # Add eligibility info if it's eligibility section
            if section == "Eligibility Criteria" and chunk.get("company_cgpa"):
                context_parts.append(f"CGPA: {chunk['company_cgpa']}")
            
            # Add recurring info
            if chunk.get("is_recurring"):
                years = chunk.get("years_visited", [])
                context_parts.append(f"Recurring: {len(years)} visits")
            
            context = " | ".join(context_parts)
            text = f"{context}\n\n{content}"
        else:
            text = f"{section}: {content}"
        
        return text

    # -------------------------
    # Embedding Generation
    # -------------------------
    def generate_embeddings(
        self,
        chunks: Optional[List[Dict]] = None,
        include_metadata: bool = True,
        batch_size: int = 8
    ) -> tuple:
        """
        Generate embeddings for text chunks with GTE-Qwen2.
        Enhanced with metadata context for better retrieval.
        """
        if chunks is None:
            chunks, _ = self.load_all_chunks()
        if not chunks:
            return [], np.array([])

        texts = []
        print("\n🔄 Formatting chunks with metadata...")
        for chunk in tqdm(chunks, desc="Formatting"):
            text = self.format_chunk_text(chunk, include_metadata)
            
            # GTE-Qwen2 instruct format for document embeddings
            if self.is_gte_qwen:
                # Enhanced instruction for placement documents
                instruction = (
                    "Given a placement document containing company information, "
                    "job descriptions, eligibility criteria, compensation details, "
                    "and selection processes, retrieve relevant information"
                )
                text = f"Instruct: {instruction}\nQuery: {text}"
            
            texts.append(text)

        print(f"\n🔄 Encoding {len(texts)} chunks with {self.model_name}...")
        print(f"⚙️ Batch size: {batch_size}")
        print(f"⚙️ Device: {self.model.device}")
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name="query" if self.is_gte_qwen else None
        )

        print(f"✅ Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Add embedding statistics
        embeddings_array = np.array(embeddings)
        print(f"\n📊 Embedding Statistics:")
        print(f"   • Shape: {embeddings_array.shape}")
        print(f"   • Mean norm: {np.linalg.norm(embeddings_array, axis=1).mean():.4f}")
        print(f"   • Size: {embeddings_array.nbytes / (1024*1024):.2f} MB")
        
        return chunks, embeddings_array

    # -------------------------
    # Save Embeddings
    # -------------------------
    def save_embeddings(
        self, 
        chunks: List[Dict], 
        embeddings: np.ndarray, 
        metadata: Dict = None,
        out_dir: str = "embeddings_store"
    ):
        """Save only embeddings locally. Chunks/metadata already in chunks_all.json"""
        os.makedirs(out_dir, exist_ok=True)
        
        # Save ONLY embeddings as numpy array
        embeddings_path = os.path.join(out_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings.astype(np.float32))
        
        # Save minimal stats (no chunk duplication)
        stats = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_chunks": len(chunks),
            "embeddings_size_mb": embeddings.nbytes / (1024 * 1024),
            "embedding_stats": {
                "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
                "std_norm": float(np.linalg.norm(embeddings, axis=1).std()),
                "min_value": float(embeddings.min()),
                "max_value": float(embeddings.max())
            },
            "source_files": {
                "chunks": os.path.join(self.chunks_dir, "chunks_all.json"),
                "metadata": os.path.join(self.chunks_dir, "metadata.json")
            },
            "note": "Full chunks and metadata available in chunks_by_filetype_new/"
        }
        
        stats_path = os.path.join(out_dir, "embedding_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
        print(f"\n💾 Saved to '{out_dir}':")
        print(f"   ├─ embeddings.npy ({stats['embeddings_size_mb']:.2f} MB)")
        print(f"   └─ embedding_stats.json (basic stats only)")
        print(f"\n📌 Source data:")
        print(f"   • Chunks: {stats['source_files']['chunks']}")
        print(f"   • Metadata: {stats['source_files']['metadata']}")

    # -------------------------
    # Load Saved Embeddings
    # -------------------------
    def load_saved_embeddings(self, in_dir="embeddings_store"):
        """Load embeddings, chunks, and stats from disk."""
        try:
            embeddings = np.load(os.path.join(in_dir, "embeddings.npy"))
            with open(os.path.join(in_dir, "chunks.json"), "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # Load stats if available
            stats_path = os.path.join(in_dir, "stats.json")
            stats = {}
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                print(f"📊 Model used: {stats.get('model_name', 'Unknown')}")
                print(f"📊 Dimension: {stats.get('embedding_dimension', 'Unknown')}")
                print(f"📊 Companies: {len(stats.get('companies', []))}")
            
            print(f"✅ Loaded {len(chunks)} chunks and embeddings from {in_dir}")
            return chunks, embeddings, stats
        except Exception as e:
            print(f"⚠️ Error loading saved embeddings: {e}")
            import traceback
            traceback.print_exc()
            return [], np.array([]), {}


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    print("="*60)
    print("EMBEDDING GENERATION WITH GTE-QWEN2-7B")
    print("="*60)
    
    # Initialize with best model
    eg = EmbeddingGenerator(model_name='Alibaba-NLP/gte-Qwen2-7B-instruct')
    
    # Load chunks and metadata
    print("\n📂 Step 1: Loading chunks with metadata...")
    chunks, metadata = eg.load_all_chunks()
    print(f"✅ Total chunks loaded: {len(chunks)}")
    
    # Show sample with enhanced metadata
    if chunks:
        print(f"\n📝 Sample chunk with metadata:")
        sample = chunks[0]
        print(f"   Company: {sample.get('company', 'N/A')}")
        print(f"   Year: {sample.get('company_year', 'N/A')}")
        print(f"   Section: {sample.get('section', 'N/A')}")
        print(f"   Role: {sample.get('role', 'N/A')}")
        print(f"   Recurring: {sample.get('is_recurring', False)}")
        if sample.get('company_stipend'):
            print(f"   Stipend: ₹{sample['company_stipend']}K/month")
        if sample.get('company_ctc'):
            print(f"   CTC: ₹{sample['company_ctc']} LPA")
        print(f"   Content preview: {sample.get('content', '')[:100]}...")
    
    # Generate embeddings with metadata context
    print("\n🔄 Step 2: Generating embeddings with metadata context...")
    chunks, embeddings = eg.generate_embeddings(
        chunks, 
        include_metadata=True,
        batch_size=8  # Adjust based on your GPU memory
    )
    
    # Save with metadata
    print("\n💾 Step 3: Saving embeddings with metadata...")
    eg.save_embeddings(chunks, embeddings, metadata)
    
    # Verify
    print("\n✅ Step 4: Verifying save/load...")
    loaded_chunks, loaded_embeddings, loaded_stats = eg.load_saved_embeddings()
    
    print("\n" + "="*60)
    print("✅ EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   • Chunks processed: {len(chunks)}")
    print(f"   • Embedding dimension: {embeddings.shape[1]}")
    print(f"   • Storage size: {embeddings.nbytes / (1024*1024):.2f} MB")
    print(f"   • Companies: {metadata.get('total_companies', 0)}")
    print(f"   • Recurring companies: {len(metadata.get('recurring_companies', []))}")
    print(f"\n📌 Next step: python pinecone_upsert.py")
    print(f"   ⚠️  Update Pinecone dimension to {embeddings.shape[1]}")
    print(f"   ⚠️  Ensure metadata fields are indexed properly")
