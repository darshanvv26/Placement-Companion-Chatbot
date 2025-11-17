import os
import json
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI


class QueryHelper:
    """Helper for querying Pinecone with company-based filtering and reranking."""

    def __init__(
        self,
        index_name: str | None = None,
        model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        chunks_dir: str = "chunks_by_filetype_new_test",
        use_reranker: bool = True,
        max_context_tokens: int = 3000,
        namespace: str = "placement-docs",
        use_llm: bool = True  # ADD THIS LINE
    ):
        # Load environment
        load_dotenv(dotenv_path="environment.env")
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("❌ PINECONE_API_KEY not set")

        # GPU Selection
        self._setup_gpu()

        # Pinecone initialization
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "placement-companion-v5")
        self.namespace = namespace
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(self.index_name)

        # Embedding model
        print(f"🧠 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=self.device
        )
        self.is_gte_qwen = "gte-qwen" in model_name.lower()
        print(f"✅ Model loaded on {self.device}")

        # Updated paths for new structure
        self.chunks_dir = chunks_dir
        self.combined_json = os.path.join(chunks_dir, "chunks_all_test.json")
        self.metadata_json = os.path.join(chunks_dir, "metadata_test.json")
        
        # Load chunks and metadata
        self.full_content, self.metadata = self._load_chunks_and_metadata()

        # Extract company names from chunks
        self.all_companies = sorted(set(
            chunk.get("company", "") 
            for chunks_list in self.full_content.values() 
            for chunk in chunks_list
            if chunk.get("company")
        ))
        print(f"📊 Loaded {len(self.all_companies)} companies")

        # Reranker
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            print("🔄 Loading re-ranker...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✅ Re-ranker loaded successfully")

        self.max_context_tokens = max_context_tokens

        # NEW: Initialize LLM client (ADD THIS SECTION)
        self.use_llm = use_llm
        self.llm_client = None
        if use_llm:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("⚠️ HF_TOKEN not found, LLM disabled")
                self.use_llm = False
            else:
                print("🤖 Initializing LLM client...")
                self.llm_client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
                print("✅ LLM client ready (openai/gpt-oss-20b)")

    def _setup_gpu(self):
        """Setup GPU device selection"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"📊 Available GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Use GPU 1 if available, else GPU 0
            gpu_id = 1 if gpu_count > 1 else 0
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"✅ Using GPU {gpu_id}: {torch.cuda.get_device_name(self.device)}")
            
            free_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            print(f"   Total Memory: {free_mem:.2f} GB")
        else:
            self.device = torch.device("cpu")
            print("⚠️ CUDA not available, using CPU")

    def _load_chunks_and_metadata(self) -> tuple[dict, dict]:
        """Load chunks and metadata from chunks_all.json and metadata.json"""
        chunks_data = {}
        metadata = {}
        
        # Load chunks_all.json
        try:
            if not os.path.exists(self.combined_json):
                raise FileNotFoundError(f"❌ Chunks file not found: {self.combined_json}")
            
            with open(self.combined_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract company-wise chunks (dict of company -> list of chunks)
            chunks_data = data.get("chunks", {})
            
            print(f"✅ Loaded chunks from {self.combined_json}")
            print(f"   Total companies: {len(chunks_data)}")
            total_chunks = sum(len(chunks) for chunks in chunks_data.values())
            print(f"   Total chunks: {total_chunks}")
            
        except Exception as e:
            print(f"⚠️ Error loading chunks: {e}")
            import traceback
            traceback.print_exc()
        
        # Load metadata.json
        try:
            if os.path.exists(self.metadata_json):
                with open(self.metadata_json, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                print(f"✅ Loaded metadata from {self.metadata_json}")
                print(f"   Recurring companies: {len(metadata.get('recurring_companies', []))}")
                if metadata.get('compensation', {}).get('avg_stipend'):
                    print(f"   Avg Stipend: {metadata['compensation']['avg_stipend']}")
                if metadata.get('compensation', {}).get('avg_ctc'):
                    print(f"   Avg CTC: {metadata['compensation']['avg_ctc']}")
            else:
                print(f"⚠️ Metadata file not found: {self.metadata_json}")
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
        
        return chunks_data, metadata

    def detect_company(self, query: str) -> str | None:
        """Extract company name if mentioned in the query."""
        query_lower = query.lower()
        for c in self.all_companies:
            # Try exact match first
            if c.lower() in query_lower:
                return c
            # Try partial match (first word of company name)
            company_token = c.lower().split("_")[0]
            if company_token in query_lower:
                return c
        return None

    def _is_aggregation_query(self, query: str) -> tuple[bool, dict | None]:
        """
        Detect if query requires aggregation/filtering on metadata.
        Returns (is_aggregation, filter_criteria)
        """
        query_lower = query.lower()
        
        # Aggregation keywords
        aggregation_keywords = [
            "how many", "list all", "which companies", "companies that",
            "all companies", "companies with", "companies offering"
        ]
        
        is_agg = any(keyword in query_lower for keyword in aggregation_keywords)
        
        if not is_agg:
            return False, None
        
        # Extract filter criteria
        criteria = {}
        
        # CTC filters
        if "ctc" in query_lower or "lpa" in query_lower or "package" in query_lower:
            # Extract threshold
            import re
            ctc_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lpa|ctc)', query_lower)
            if ctc_match:
                threshold = float(ctc_match.group(1))
                
                if any(word in query_lower for word in ["more than", "above", "greater", ">"]):
                    criteria["ctc_min"] = threshold
                elif any(word in query_lower for word in ["less than", "below", "under", "<"]):
                    criteria["ctc_max"] = threshold
                elif "between" in query_lower:
                    # Try to find range
                    range_match = re.findall(r'(\d+(?:\.\d+)?)', query_lower)
                    if len(range_match) >= 2:
                        criteria["ctc_min"] = float(range_match[0])
                        criteria["ctc_max"] = float(range_match[1])
    
        # Stipend filters
        if "stipend" in query_lower or "internship" in query_lower:
            import re
            stipend_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|thousand)', query_lower)
            if stipend_match:
                threshold = float(stipend_match.group(1))
                
                if any(word in query_lower for word in ["more than", "above", "greater", ">"]):
                    criteria["stipend_min"] = threshold
                elif any(word in query_lower for word in ["less than", "below", "under", "<"]):
                    criteria["stipend_max"] = threshold
    
        # Year filter
        import re
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        if year_match:
            criteria["year"] = year_match.group(1)
    
        # Role filter
        if "intern" in query_lower:
            criteria["role"] = "Intern"
        elif "full" in query_lower or "fte" in query_lower:
            criteria["role"] = "Full-Time"
        
        # Return empty dict instead of None if no criteria found
        # This allows "list all companies" queries to work
        return True, criteria

    def _aggregate_companies(self, criteria: dict | None) -> list[dict]:
        """
        Aggregate companies based on metadata criteria.
        Returns list of companies with their relevant data.
        """
        results = []
        
        # Handle None criteria (list all companies)
        if criteria is None:
            criteria = {}
        
        for company_name, chunks in self.full_content.items():
            # Get company metadata from first chunk (all chunks have same company metadata)
            if not chunks:
                continue
            
            sample_chunk = chunks[0]
            company_data = {
                "company": company_name,
                "ctc": sample_chunk.get("company_ctc"),
                "stipend": sample_chunk.get("company_stipend"),
                "year": sample_chunk.get("company_year"),
                "role": sample_chunk.get("role", "General"),
                "is_recurring": sample_chunk.get("is_recurring", False),
                "years_visited": sample_chunk.get("years_visited", [])
            }
            
            # Apply filters only if criteria exist
            match = True
            
            # CTC filters
            if "ctc_min" in criteria:
                if not company_data["ctc"] or company_data["ctc"] < criteria["ctc_min"]:
                    match = False
            if "ctc_max" in criteria:
                if not company_data["ctc"] or company_data["ctc"] > criteria["ctc_max"]:
                    match = False
            
            # Stipend filters
            if "stipend_min" in criteria:
                if not company_data["stipend"] or company_data["stipend"] < criteria["stipend_min"]:
                    match = False
            if "stipend_max" in criteria:
                if not company_data["stipend"] or company_data["stipend"] > criteria["stipend_max"]:
                    match = False
            
            # Year filter
            if "year" in criteria:
                if company_data["year"] != criteria["year"]:
                    match = False
            
            # Role filter
            if "role" in criteria:
                if criteria["role"].lower() not in company_data["role"].lower():
                    match = False
            
            if match:
                results.append(company_data)
        
        return results

    def _format_aggregation_results(self, companies: list[dict], criteria: dict, query: str) -> str:
        """Format aggregation results into a coherent context"""
        if not companies:
            return "No companies found matching the criteria."
        
        context_parts = [
            f"Found {len(companies)} companies matching the criteria:\n"
        ]
        
        for i, comp in enumerate(companies, 1):
            company_info = [f"{i}. {comp['company']}"]
            
            if comp.get("ctc"):
                company_info.append(f"CTC: ₹{comp['ctc']} LPA")
            if comp.get("stipend"):
                company_info.append(f"Stipend: ₹{comp['stipend']}K/month")
            if comp.get("year"):
                company_info.append(f"Year: {comp['year']}")
            if comp.get("role"):
                company_info.append(f"Role: {comp['role']}")
            if comp.get("is_recurring"):
                years = ", ".join(comp.get("years_visited", []))
                company_info.append(f"Recurring (Years: {years})")
            
            context_parts.append(" | ".join(company_info))
        
        # Add summary statistics
        ctc_values = [c["ctc"] for c in companies if c.get("ctc")]
        stipend_values = [c["stipend"] for c in companies if c.get("stipend")]
        
        if ctc_values:
            context_parts.append(f"\n📊 CTC Statistics:")
            context_parts.append(f"   • Average: ₹{sum(ctc_values)/len(ctc_values):.2f} LPA")
            context_parts.append(f"   • Highest: ₹{max(ctc_values)} LPA")
            context_parts.append(f"   • Lowest: ₹{min(ctc_values)} LPA")
        
        if stipend_values:
            context_parts.append(f"\n📊 Stipend Statistics:")
            context_parts.append(f"   • Average: ₹{sum(stipend_values)/len(stipend_values):.2f}K/month")
            context_parts.append(f"   • Highest: ₹{max(stipend_values)}K/month")
            context_parts.append(f"   • Lowest: ₹{min(stipend_values)}K/month")
        
        return "\n".join(context_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4

    def _condense_chunks(self, chunks_data: list, query_text: str) -> list:
        """Condense chunks if context exceeds token limit"""
        total_tokens = sum(self._estimate_tokens(c["content"]) for c in chunks_data)
        if total_tokens <= self.max_context_tokens:
            print(f"📊 Total tokens: {total_tokens} (within limit)")
            return chunks_data

        print(f"⚠️ Context too large ({total_tokens} tokens). Condensing...")

        # Re-rank if available
        if self.reranker:
            pairs = [(query_text, c["content"]) for c in chunks_data]
            scores = self.reranker.predict(pairs)
            for i, c in enumerate(chunks_data):
                c["relevance_score"] = float(scores[i])
            chunks_data = sorted(chunks_data, key=lambda x: x["relevance_score"], reverse=True)

        condensed, current_tokens = [], 0
        for chunk in chunks_data:
            chunk_tokens = self._estimate_tokens(chunk["content"])
            if current_tokens + chunk_tokens > self.max_context_tokens:
                break
            condensed.append(chunk)
            current_tokens += chunk_tokens

        print(f"✂️ Condensed to {len(condensed)} chunks (~{current_tokens} tokens)")
        return condensed

    def hybrid_query(self, query_text: str, top_k: int = 10, force_semantic: bool = False) -> tuple[str | None, list]:
        """
        🚀 Enhanced Hybrid Flow:
        1. Check if aggregation query → filter companies by metadata
        2. Detect company from query → load all chunks for that company + condense
        3. Else → semantic Pinecone retrieval with reranking
        """
        # Check for aggregation query first
        is_agg, criteria = self._is_aggregation_query(query_text)
        
        if is_agg:
            print(f"📊 Aggregation query detected!")
            print(f"   Criteria: {criteria}")
            
            companies = self._aggregate_companies(criteria)
            print(f"✅ Found {len(companies)} matching companies")
            
            # Format as context
            context_text = self._format_aggregation_results(companies, criteria, query_text)
            
            # Return as pseudo-chunks
            return None, [{
                "chunk_id": "aggregation_result",
                "company": "Multiple",
                "section": "Aggregation",
                "content": context_text,
                "score": 1.0,
                "aggregation_data": companies
            }]
        
        # Regular company detection
        company = self.detect_company(query_text) if not force_semantic else None

        # Company-specific retrieval
        if company and not force_semantic:
            print(f"🏢 Company detected: {company}")
            print(f"📦 Loading ALL chunks for {company}")

            # Get all chunks for this company from the loaded data
            company_chunks_raw = self.full_content.get(company, [])
            
            # Format chunks with enhanced metadata
            company_chunks = []
            for chunk in company_chunks_raw:
                formatted_chunk = {
                    "chunk_id": chunk.get("global_chunk_id", chunk.get("chunk_id")),
                    "company": chunk.get("company"),
                    "section": chunk.get("section", "Unknown"),
                    "role": chunk.get("role", "General"),
                    "filename": chunk.get("filename", "Unknown"),
                    "file_type": chunk.get("file_type", "Unknown"),
                    "year": chunk.get("company_year", chunk.get("year", "N/A")),
                    "ctc": chunk.get("company_ctc", chunk.get("ctc")),
                    "stipend": chunk.get("company_stipend", chunk.get("stipend")),
                    "content": chunk.get("content", ""),
                    "score": 1.0
                }
                
                # Add Excel-derived metadata if available
                if chunk.get("is_recurring"):
                    formatted_chunk["is_recurring"] = True
                    formatted_chunk["years_visited"] = chunk.get("years_visited", [])
                
                company_chunks.append(formatted_chunk)

            print(f"✅ Found {len(company_chunks)} chunks for {company}")
            
            if not company_chunks:
                print("⚠️ No chunks found, falling back to semantic search")
                results = self.query(query_text, top_k=top_k, auto_detect_company=False)
                return None, results
            
            condensed = self._condense_chunks(company_chunks, query_text)
            return company, condensed

        # Semantic search via Pinecone
        else:
            print("🔍 No company detected → Semantic search via Pinecone")
            results = self.query(query_text, top_k=top_k, auto_detect_company=False)
            return None, results

    def query(self, query_text: str, top_k: int = 5, company: str | None = None,
              section: str | None = None, role: str | None = None,
              auto_detect_company: bool = True, apply_reranker: bool = True):
        """Query Pinecone with optional filters and reranking"""
        
        if auto_detect_company and not company:
            detected = self.detect_company(query_text)
            if detected:
                print(f"🔍 Auto-detected company filter: {detected}")
                company = detected

        # Build filter
        filt = {}
        if company:
            filt["company"] = company
        if section:
            filt["section"] = section
        if role:
            filt["role"] = role

        # Format query for GTE-Qwen2
        if self.is_gte_qwen:
            instruction = (
                "Given a placement document containing company information, "
                "job descriptions, eligibility criteria, compensation details, "
                "and selection processes, retrieve relevant information"
            )
            query_emb_text = f"Instruct: {instruction}\nQuery: {query_text}"
        else:
            query_emb_text = query_text

        # Generate embedding
        query_emb = self.model.encode(
            query_emb_text,
            normalize_embeddings=True,
            prompt_name="query" if self.is_gte_qwen else None
        )

        # Fetch more results if reranking
        fetch_k = top_k * 3 if (apply_reranker and self.reranker) else top_k
        
        results = self.index.query(
            vector=query_emb.tolist(),
            top_k=fetch_k,
            include_metadata=True,
            filter=filt or None,
            namespace=self.namespace
        )

        # Enrich with full content from local cache
        enriched = []
        for match in getattr(results, "matches", []):
            md = match.metadata or {}
            chunk_id = md.get("chunk_id")
            company_name = md.get("company", "Unknown")
            
            # Try to find full content from loaded chunks
            full_chunk = None
            if company_name in self.full_content:
                for chunk in self.full_content[company_name]:
                    if chunk.get("global_chunk_id") == chunk_id or chunk.get("chunk_id") == chunk_id:
                        full_chunk = chunk
                        break
            
            if full_chunk:
                enriched.append({
                    "score": match.score,
                    "chunk_id": chunk_id,
                    "company": full_chunk.get("company", company_name),
                    "section": full_chunk.get("section", md.get("section", "Unknown")),
                    "role": full_chunk.get("role", md.get("role", "General")),
                    "filename": full_chunk.get("filename", md.get("filename", "Unknown")),
                    "file_type": full_chunk.get("file_type", md.get("file_type", "Unknown")),
                    "year": full_chunk.get("company_year", md.get("year", "N/A")),
                    "ctc": full_chunk.get("company_ctc", md.get("ctc")),
                    "stipend": full_chunk.get("company_stipend", md.get("stipend")),
                    "content": full_chunk.get("content", md.get("content", "")),
                })
            else:
                # Fallback to metadata only
                enriched.append({
                    "score": match.score,
                    "chunk_id": chunk_id,
                    "company": company_name,
                    "section": md.get("section", "Unknown"),
                    "role": md.get("role", "General"),
                    "filename": md.get("filename", "Unknown"),
                    "file_type": md.get("file_type", "Unknown"),
                    "year": md.get("year", "N/A"),
                    "ctc": md.get("ctc"),
                    "stipend": md.get("stipend"),
                    "content": md.get("content", "")
                })

        # Apply reranking
        if apply_reranker and self.reranker and enriched:
            print(f"🔄 Re-ranking {len(enriched)} results...")
            pairs = [(query_text, r["content"]) for r in enriched]
            rerank_scores = self.reranker.predict(pairs)
            for i, r in enumerate(enriched):
                r["rerank_score"] = float(rerank_scores[i])
            enriched = sorted(enriched, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        else:
            enriched = enriched[:top_k]

        return enriched

    def get_company_context(self, query_text: str, top_k: int = 10) -> tuple[str | None, str]:
        """Get context for a query with company detection"""
        company, results = self.hybrid_query(query_text, top_k=top_k)
        context = "\n\n".join([r["content"] for r in results])
        return company, context

    # NEW: LLM-powered answer generation
    def generate_answer(
        self, 
        query: str, 
        top_k: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 800
    ) -> dict:
        """
        Generate an answer using LLM based on retrieved context.
        
        Returns:
            dict with keys: 'query', 'answer', 'company', 'context', 'sources'
        """
        # Get context using existing retrieval - FIX: use hybrid_query directly
        company, results = self.hybrid_query(query, top_k=top_k)
        
        # If LLM disabled, return context only
        if not self.use_llm or not self.llm_client:
            context = "\n\n".join([r["content"] for r in results])
            return {
                "query": query,
                "answer": "LLM disabled. Showing raw context.",
                "company": company,
                "context": context,
                "sources": []
            }
        
        # Check if this is an aggregation query with special formatting
        is_aggregation = any(r.get("aggregation_data") for r in results)
        
        if is_aggregation:
            # For aggregation queries, use the formatted context directly
            context = results[0]["content"]
            
            system_prompt = """You are a helpful placement assistant for MSIS (Manipal School of Information Science).
Answer questions based ONLY on the provided placement data.
Format lists clearly with bullet points or numbered lists.
Be specific with numbers, packages, and company names."""

            user_prompt = f"""Placement Data for MSIS:
{context}

Question: {query}

Provide a clear, well-formatted answer based on the data above."""

        else:
            # For regular queries, build context from chunks
            context_parts = []
            sources = []
            
            for i, result in enumerate(results, 1):
                comp = result.get("company", "Unknown")
                section = result.get("section", "Unknown")
                content = result.get("content", "")
                
                if comp not in sources:
                    sources.append(comp)
                
                context_parts.append(f"--- Source {i}: {comp} - {section} ---\n{content}")
            
            context = "\n\n".join(context_parts)
            
            system_prompt = """You are a helpful placement assistant for MSIS (Manipal School of Information Science).
Answer questions based ONLY on the provided context from placement documents.
If the context doesn't contain the answer, say "I don't have information about this in the placement records."
Be specific with eligibility criteria, packages, processes, and dates.
Use bullet points for lists and processes."""

            user_prompt = f"""Context from MSIS Placement Records:
{context}

Question: {query}

Answer based on the context above:"""

        try:
            print("🤖 Generating answer with LLM...")
            print(f"📏 Context length: {len(context)} chars")
            
            completion = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-20b:groq",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = completion.choices[0].message.content.strip()
            
            # Build sources list
            if is_aggregation:
                final_sources = ["All Companies"]
            elif company:
                final_sources = [company]
            else:
                final_sources = list(set(r.get("company", "Unknown") for r in results))
            
            return {
                "query": query,
                "answer": answer,
                "company": company,
                "context": context,
                "sources": final_sources
            }
        
        except Exception as e:
            print(f"❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            
            # Build context for error case
            context = "\n\n".join([r.get("content", "") for r in results])
            
            return {
                "query": query,
                "answer": f"Error generating answer: {e}",
                "company": company,
                "context": context,
                "sources": []
            }


if __name__ == "__main__":
    helper = QueryHelper(use_llm=True)  # Enable LLM
    
    test_queries = [
        "What is the eligibility criteria for Amazon in msis?",
        "How many companies offer more than 15 LPA CTC in msis?",
        "Tell me about Novartis internship stipend",
        "Which companies visited msis?",
        "List all firms with packages exceeding 20 lakhs",  # Paraphrased
        "Companies offering internships above 50K per month",
        "What is the selection process for intel?"
    ]

    print("\n" + "="*60)
    print("TESTING QUERY HELPER WITH LLM - MSIS PLACEMENT ASSISTANT")
    print("="*60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 60)
        
        # Get LLM-generated answer
        result = helper.generate_answer(query, top_k=5)
        
        print(f"🏢 Company: {result['company'] or 'Multiple'}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        print(f"\n📝 Answer:\n{result['answer']}\n")
        # print(f"💬 Context preview: {result['context'][:]}...")