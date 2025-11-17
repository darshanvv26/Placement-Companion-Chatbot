import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
from typing import Dict, List


class LLMReasoner:
    """
    Post-processes RAG answers with:
    1. Format detection (table vs points vs prose)
    2. Answer refinement
    3. Hallucination checking
    4. Citation addition
    """
    
    def __init__(self):
        load_dotenv(dotenv_path="environment.env")
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("❌ HF_TOKEN not found")
        
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        print("✅ LLM Reasoner initialized")
    
    def detect_answer_format(self, query: str) -> str:
        query_lower = query.lower()
        table_keywords = [
            "list", "compare", "which companies", "all companies",
            "companies that", "companies with", "how many companies",
            "what are the", "show me"
        ]
        bullet_keywords = [
            "process", "steps", "rounds", "criteria", "requirements",
            "skills", "what is", "tell me about", "describe"
        ]
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference between"]):
            return "table"
        if any(kw in query_lower for kw in table_keywords):
            if any(word in query_lower for word in ["companies", "all", "list"]):
                return "table"
            return "bullet_points"
        if any(kw in query_lower for kw in bullet_keywords):
            return "bullet_points"
        return "prose"
    
    def refine_answer(
        self,
        query: str,
        draft_answer: str,
        context: str,
        sources: List[str],
        format_type: str = None
    ) -> Dict:
        if format_type is None:
            format_type = self.detect_answer_format(query)
        
        print(f"📊 Detected format: {format_type.upper()}")
        
        if format_type == "table":
            refinement_prompt = self._build_table_prompt(query, draft_answer, context, sources)
        elif format_type == "bullet_points":
            refinement_prompt = self._build_bullet_prompt(query, draft_answer, context, sources)
        else:
            refinement_prompt = self._build_prose_prompt(query, draft_answer, context, sources)
        
        # ---------------------- NEW SYSTEM PROMPT HEADER ----------------------
        system_prompt_header = """
You are the **MSIS Placement Bot**, designed to assist master's students from **Manipal School of Information Science (MSIS), MAHE**.

### 🏫 HEADER CONTEXT
1. You have access to **extensive placement data** from years **2023, 2024, and 2025**.
2. MSIS stands for **Manipal School of Information Science**, which offers **Master’s programs** such as M.Tech/MS in Data Science, AI, and related fields.
   - Learn more: [MSIS Official Site](https://www.manipal.edu/sois.html)
3. **MAHE (Manipal Academy of Higher Education)** is the parent university of MSIS.  
   - Learn more: [MAHE Official Website](https://manipal.edu/)
4. Your users are **postgraduate students** preparing for placements and internships.

### 💡 BODY INSTRUCTIONS
1. Always interpret the **user query carefully** and answer **based on retrieved context** (eligibility, stipend, company, year, etc.).
2. Present structured data such as **eligibility, stipend, CTC, job role, and duration** in a **tabular form** wherever possible.
3. Write **prolonged, professional, and factual answers** — as a **placement expert**, ensure clarity, accuracy, and depth.

### 🔗 TAIL INSTRUCTIONS
1. At the end of every answer, include **followable external links** (LinkedIn, Naukri, Glassdoor) for further company insights., e.g.: (https://www.linkedin.com/company/intel-corporation/)(https://www.glassdoor.co.in/Search/results.htm?keyword={company_name}).
2. If some information is missing, instruct the user to **reach out to current company employees** via LinkedIn and provide a generic search link, e.g.:  
   👉 [Search on LinkedIn](https://www.linkedin.com/company/intel-corporation/people/)
        """

        # ----------------------------------------------------------------------

        try:
            print("🔄 Refining answer...")
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-20b:groq",
                messages=[
                    {"role": "system", "content": system_prompt_header},
                    {"role": "system", "content": "You are an expert at formatting placement information clearly and accurately for MSIS students."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            refined_answer = completion.choices[0].message.content.strip()
            
            validation_result = self._validate_answer(refined_answer, context)
            
            return {
                "query": query,
                "refined_answer": refined_answer,
                "format_type": format_type,
                "validation": validation_result,
                "sources": sources
            }
        
        except Exception as e:
            print(f"⚠️ Refinement failed: {e}")
            return {
                "query": query,
                "refined_answer": draft_answer,
                "format_type": "prose",
                "validation": {"passed": False, "issues": [str(e)]},
                "sources": sources
            }
    
    def _build_table_prompt(self, query: str, draft: str, context: str, sources: List[str]) -> str:
        return f"""You are formatting placement data for MSIS students.

Original Question: {query}

Draft Answer:
{draft}

Source Context (verify facts):
{context[:2000]}...

Sources: {', '.join(sources)}

**Task**: Reformat this answer as a **clean Markdown table** with these requirements:

1. **Table columns should match the query** 
2. **Sort by most relevant column** (usually CTC or Stipend, highest first)
3. **Add a summary** at the end with statistics
4. **Include source citation** after table
5. **Only include information present in the context**
6. **Use proper formatting**: ₹ for money, "N/A" for missing data

Example format:
| Company | 
|---------|
| ABC     | 
| XYZ     | 

**Summary**: 2 companies visited MSIS in 2025
**Source**: MSIS Placement Records 2025

Now format the answer:"""

    def _build_bullet_prompt(self, query: str, draft: str, context: str, sources: List[str]) -> str:
        return f"""You are formatting placement data for MSIS students.

Original Question: {query}

Draft Answer:
{draft}

Source Context (verify facts):
{context[:2000]}...

Sources: {', '.join(sources)}

**Task**: Reformat this answer as **clear bullet points** with:

1. **Main heading** with company name (e.g., "# Amazon - Eligibility Criteria")
2. **Organized sections** with emoji headers:
   - 📋 **Eligibility** (CGPA, branch, year requirements)
   - 🔄 **Selection Process** (rounds, tests, interview stages)
   - 💰 **Compensation** (CTC/stipend with exact amounts)
   - 🎯 **Key Requirements** (skills, qualifications)
3. **Sub-bullets** for details
4. **Bold** all numbers, dates, and key terms
5. **Source citation** at end in format: `(Source: [Company_Year])`

Example format:
# Amazon - Eligibility Criteria

📋 **Eligibility**
- Minimum CGPA: **8.0** in PG (Master's)
- Branch: All branches eligible
- Year: **2026** batch

💰 **Compensation**
- Stipend: **₹70,000/month**
- Duration: **6 months**
- Starting: **June 2025**

(Source: Amazon_MTech_2026)

Now format the answer:"""

    def _build_prose_prompt(self, query: str, draft: str, context: str, sources: List[str]) -> str:
        return f"""You are formatting placement data for MSIS students.

Original Question: {query}

Draft Answer:
{draft}

Source Context (verify facts):
{context[:2000]}...

Sources: {', '.join(sources)}

**Task**: Rewrite this answer as **clear, concise prose** with:

1. **Direct answer first** (lead with the key fact in the first sentence)
2. **Supporting details** in 2-3 sentences
3. **Bold all numbers, dates, and amounts**
4. **Source citation** at end: `(Source: [Company_MTech_2026])`
5. **No hallucinated information** - only use context data

Example format:
The eligibility criteria for Amazon at MSIS is **8.0 CGPA** in the Master's program. The company offers a stipend of **₹70,000/month** for a **6-month** internship starting in **June 2025**. All branches are eligible to apply. (Source: Amazon_MTech_2026)

Now format the answer:"""

    def _validate_answer(self, answer: str, context: str) -> Dict:
        issues = []
        passed = True
        numbers_in_answer = re.findall(r'\d+(?:\.\d+)?', answer)
        missing_count = 0
        for num in numbers_in_answer:
            if num not in context:
                missing_count += 1
        if numbers_in_answer and missing_count / len(numbers_in_answer) > 0.5:
            issues.append(f"Warning: Some numbers may not be in source context")
        hallucination_phrases = [
            "according to my knowledge", "i believe", "probably", "might be", "it seems", "i think"
        ]
        answer_lower = answer.lower()
        for phrase in hallucination_phrases:
            if phrase in answer_lower:
                issues.append(f"Uncertain language detected: '{phrase}'")
                passed = False
        return {
            "passed": passed,
            "issues": issues if issues else ["All checks passed"]
        }

    def format_final_output(self, result: Dict) -> str:
        output = []
        output.append("=" * 70)
        output.append(f"📝 Query: {result['query']}")
        output.append("=" * 70)
        output.append(f"\n{result['refined_answer']}\n")
        output.append("-" * 70)
        output.append(f"📊 Format: {result['format_type'].upper()}")
        output.append(f"📚 Sources: {', '.join(result['sources'])}")
        validation = result['validation']
        if validation['passed']:
            output.append("✅ Validation: PASSED")
        else:
            output.append("⚠️ Validation: ISSUES DETECTED")
            for issue in validation['issues']:
                output.append(f"   • {issue}")
        return "\n".join(output)


# -------------------------
# Integration Example
# -------------------------
if __name__ == "__main__":
    from query_helper import QueryHelper
    
    print("\n" + "=" * 70)
    print("🚀 INITIALIZING ADVANCED RAG WITH LLM REASONER")
    print("=" * 70)
    
    print("\n1️⃣ Loading QueryHelper...")
    helper = QueryHelper(use_llm=True)
    
    print("\n2️⃣ Loading LLM Reasoner...")
    reasoner = LLMReasoner()
    
    test_queries = [
        "Which companies visited MSIS in 2026?",
        "What is the eligibility criteria for Amazon?",
        "What is the selection process for Intel?",
        "Tell me about Novartis internship stipend",
    ]
    
    print("\n" + "=" * 70)
    print("🧪 TESTING LLM REASONER - ADVANCED RAG")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*70}")
        print(f"🔍 Query {i}/{len(test_queries)}: {query}")
        print("="*70)
        
        try:
            print("\n📝 Step 1: Getting draft answer from RAG...")
            draft_result = helper.generate_answer(query, top_k=5)
            print(f"   Draft answer length: {len(draft_result['answer'])} chars")
            print(f"   Sources: {', '.join(draft_result['sources'])}")
            
            print("\n🔄 Step 2: Refining with LLM Reasoner...")
            refined = reasoner.refine_answer(
                query=query,
                draft_answer=draft_result['answer'],
                context=draft_result['context'],
                sources=draft_result['sources']
            )
            
            print("\n" + "="*70)
            print("📋 FINAL RESULT")
            print("="*70)
            print(reasoner.format_final_output(refined))
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE")
    print("=" * 70)
