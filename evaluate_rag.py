import json
from test_query_helper import QueryHelper
from test_llm_reasoner import LLMReasoner
from typing import List, Dict

# Test dataset with expected answers
EVAL_QUERIES = [
    {
        "query": "What is the eligibility criteria for Amazon?",
        "expected_keywords": ["8.0", "CGPA", "PG", "Master"],
        "expected_company": "Amazon_MTech_2026",
        "category": "eligibility"
    },
    {
        "query": "Which companies visited MSIS in 2026?",
        "expected_count_min": 40,  # At least 40 companies
        "category": "aggregation"
    },
    {
        "query": "What is Intel selection process?",
        "expected_keywords": ["round", "interview", "test"],
        "expected_company": "Intel_MTech_2026",
        "category": "process"
    },
    {
        "query": "Tell me about Novartis stipend",
        "expected_keywords": ["35", "35000", "stipend"],
        "expected_company": "Novartis_MTech_2026",
        "category": "compensation"
    },
    {
        "query": "Companies with CTC above 15 LPA",
        "expected_count_min": 5,
        "category": "aggregation"
    },
]


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.helper = QueryHelper(use_llm=True)
        self.reasoner = LLMReasoner()
    
    def evaluate_single_query(self, test_case: Dict) -> Dict:
        """Evaluate one query"""
        query = test_case["query"]
        
        # Get draft answer
        draft_result = self.helper.generate_answer(query, top_k=5)
        
        # Refine answer
        refined = self.reasoner.refine_answer(
            query=query,
            draft_answer=draft_result['answer'],
            context=draft_result['context'],
            sources=draft_result['sources']
        )
        
        # Evaluate
        result = {
            "query": query,
            "category": test_case.get("category"),
            "draft_answer": draft_result['answer'],
            "refined_answer": refined['refined_answer'],
            "sources": refined['sources'],
            "format": refined['format_type'],
            "validation_passed": refined['validation']['passed'],
            "scores": {}
        }
        
        # Check keyword presence
        if "expected_keywords" in test_case:
            answer_lower = refined['refined_answer'].lower()
            found_keywords = sum(
                1 for kw in test_case["expected_keywords"] 
                if kw.lower() in answer_lower
            )
            result["scores"]["keyword_match"] = found_keywords / len(test_case["expected_keywords"])
        
        # Check company detection
        if "expected_company" in test_case:
            result["scores"]["company_correct"] = (
                test_case["expected_company"] in refined['sources']
            )
        
        # Check aggregation count
        if "expected_count_min" in test_case:
            # Count numbers in answer
            import re
            numbers = re.findall(r'\d+', refined['refined_answer'])
            if numbers:
                max_count = max(int(n) for n in numbers)
                result["scores"]["count_valid"] = max_count >= test_case["expected_count_min"]
        
        return result
    
    def evaluate_all(self) -> Dict:
        """Run full evaluation"""
        print("\n" + "="*70)
        print("📊 STARTING RAG EVALUATION")
        print("="*70)
        
        results = []
        for i, test_case in enumerate(EVAL_QUERIES, 1):
            print(f"\n🔍 [{i}/{len(EVAL_QUERIES)}] {test_case['query']}")
            
            try:
                result = self.evaluate_single_query(test_case)
                results.append(result)
                
                # Print scores
                for metric, score in result["scores"].items():
                    if isinstance(score, bool):
                        print(f"   {'✅' if score else '❌'} {metric}: {score}")
                    else:
                        print(f"   📊 {metric}: {score:.2%}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    "query": test_case['query'],
                    "error": str(e),
                    "scores": {}
                })
        
        # Calculate overall metrics
        total_scores = {}
        for result in results:
            for metric, score in result.get("scores", {}).items():
                if metric not in total_scores:
                    total_scores[metric] = []
                total_scores[metric].append(float(score) if isinstance(score, bool) else score)
        
        summary = {
            "total_queries": len(EVAL_QUERIES),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "average_scores": {
                metric: sum(scores) / len(scores) 
                for metric, scores in total_scores.items()
            }
        }
        
        # Print summary
        print("\n" + "="*70)
        print("📈 EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Queries: {summary['total_queries']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"\n📊 Average Scores:")
        for metric, score in summary['average_scores'].items():
            print(f"   {metric}: {score:.2%}")
        
        # Save results
        with open("evaluation_results.json", "w") as f:
            json.dump({
                "summary": summary,
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\n💾 Saved to evaluation_results.json")
        
        return summary


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.evaluate_all()