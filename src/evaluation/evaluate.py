"""
Comprehensive Evaluation Framework for RAG System

Based on unified architecture recommendations:
- RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- Custom metrics for question-specific evaluation
- Comparison with golden dataset
- Detailed reporting
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None
    context_recall = None

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.retrieval.rag_agent import RAGAgent


class RAGEvaluator:
    """Comprehensive evaluation framework for RAG system."""
    
    def __init__(self, agent: RAGAgent):
        """Initialize evaluator with RAG agent."""
        self.agent = agent
        self.results: List[Dict[str, Any]] = []
    
    def evaluate_question(
        self,
        question: str,
        expected_output: str,
        question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single question against expected output.
        
        Returns comprehensive metrics including:
        - Exact match
        - Keyword coverage
        - Number match (for calculations)
        - Custom question-specific checks
        """
        # Get answer from agent
        resp = self.agent.answer(question)
        generated_answer = resp.get("answer", "")
        query_category = resp.get("query_category", "unknown")
        
        # Extract contexts for evaluation
        contexts = self._extract_contexts(resp.get("sources", []))
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            question=question,
            generated_answer=generated_answer,
            expected_output=expected_output,
            contexts=contexts,
            question_id=question_id
        )
        
        result = {
            "question_id": question_id or "unknown",
            "question": question,
            "expected_output": expected_output,
            "generated_answer": generated_answer,
            "query_category": query_category,
            "contexts": contexts,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        return result
    
    def _extract_contexts(self, sources: List[Any]) -> List[str]:
        """Extract context strings from sources."""
        contexts = []
        for src in sources:
            if isinstance(src, dict):
                if "content" in src:
                    contexts.append(str(src["content"]))
                elif src.get("type") == "sql":
                    contexts.append(str(src.get("content", "")))
                elif "page_content" in src:
                    contexts.append(str(src["page_content"]))
        return contexts
    
    def _calculate_metrics(
        self,
        question: str,
        generated_answer: str,
        expected_output: str,
        contexts: List[str],
        question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # 1. Exact match
        metrics["exact_match"] = self._exact_match(generated_answer, expected_output)
        
        # 2. Keyword coverage
        metrics["keyword_coverage"] = self._keyword_coverage(generated_answer, expected_output)
        
        # 3. Number match (for calculations)
        metrics["number_match"] = self._number_match(generated_answer, expected_output)
        
        # 4. Question-specific checks
        if question_id:
            question_specific = self._question_specific_checks(
                question_id, generated_answer, expected_output
            )
            metrics.update(question_specific)
        
        # 5. Answer completeness
        metrics["answer_completeness"] = self._answer_completeness(
            generated_answer, expected_output
        )
        
        # 6. Source citation quality
        metrics["source_citation"] = self._source_citation_quality(generated_answer)
        
        return metrics
    
    def _exact_match(self, generated: str, expected: str) -> bool:
        """Check if generated answer exactly matches expected (case-insensitive)."""
        gen_norm = " ".join(generated.lower().split())
        exp_norm = " ".join(expected.lower().split())
        return gen_norm == exp_norm
    
    def _keyword_coverage(self, generated: str, expected: str) -> float:
        """Calculate percentage of important keywords from expected that appear in generated."""
        import re
        
        # Extract keywords (capitalized phrases, numbers, important terms)
        def extract_keywords(text: str) -> set:
            keywords = set()
            # Capitalized phrases
            keywords.update(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text))
            # Numbers
            keywords.update(re.findall(r'\d+\.?\d*%?', text))
            # Important terms (after common stop words removal)
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            stop_words = {'this', 'that', 'the', 'and', 'for', 'with', 'from', 'are', 'was', 'were'}
            keywords.update([w for w in words if w not in stop_words])
            return keywords
        
        exp_keywords = extract_keywords(expected)
        gen_keywords = extract_keywords(generated)
        
        if not exp_keywords:
            return 1.0
        
        matched = len(exp_keywords & gen_keywords)
        return matched / len(exp_keywords)
    
    def _number_match(self, generated: str, expected: str) -> float:
        """Extract and compare numeric values."""
        import re
        
        def extract_numbers(text: str) -> set:
            # Extract numbers (including percentages, dollar amounts)
            numbers = re.findall(r'[\$]?(\d+\.?\d*)%?', text)
            return set(numbers)
        
        exp_numbers = extract_numbers(expected)
        gen_numbers = extract_numbers(generated)
        
        if not exp_numbers:
            return 1.0
        
        matched = len(exp_numbers & gen_numbers)
        return matched / len(exp_numbers)
    
    def _question_specific_checks(
        self,
        question_id: str,
        generated: str,
        expected: str
    ) -> Dict[str, Any]:
        """Question-specific evaluation checks."""
        checks = {}
        gen_lower = generated.lower()
        exp_lower = expected.lower()
        
        if question_id == "EF_1":
            # List question: count rules found
            import re
            # Extract rule names from expected
            exp_rules = set(re.findall(r'\*\s+([^\n*]+)', expected))
            gen_rules = set(re.findall(r'\*\s+([^\n*]+)', generated))
            # Also check for "Rule C-X" patterns
            exp_rule_nums = set(re.findall(r'rule\s+c-?\d+', exp_lower))
            gen_rule_nums = set(re.findall(r'rule\s+c-?\d+', gen_lower))
            
            total_rules = len(exp_rules) + len(exp_rule_nums)
            found_rules = len(gen_rules & exp_rules) + len(gen_rule_nums & exp_rule_nums)
            
            checks["rules_found"] = found_rules
            checks["total_rules"] = total_rules
            checks["rules_coverage"] = found_rules / total_rules if total_rules > 0 else 0.0
        
        elif question_id == "EF_2":
            # Territory/GRG question: check specific facts
            checks["has_yes"] = "yes" in gen_lower and "no" not in gen_lower[:200]
            checks["has_territory_118"] = "territory 118" in gen_lower or ("118" in gen_lower and "territory" in gen_lower)
            checks["has_territory_117"] = "territory 117" in gen_lower or ("117" in gen_lower and "territory" in gen_lower)
            checks["has_rate_0_305"] = "0.305" in gen_lower or "0.305%" in gen_lower
            checks["has_rate_neg_0_133"] = "-0.133" in gen_lower or "-0.133%" in gen_lower
            checks["has_grg_51"] = "51" in gen_lower or "051" in gen_lower
            checks["has_grg_15"] = "15" in gen_lower or "015" in gen_lower
            checks["has_difference_36"] = "36" in gen_lower or ("exceed" in gen_lower and "30" in gen_lower)
            
            # Calculate score
            fact_checks = [
                checks["has_yes"],
                checks["has_territory_118"],
                checks["has_territory_117"],
                checks["has_rate_0_305"],
                checks["has_rate_neg_0_133"],
                checks["has_grg_51"],
                checks["has_grg_15"],
                checks["has_difference_36"]
            ]
            checks["fact_score"] = sum(fact_checks) / len(fact_checks)
        
        elif question_id == "EF_3":
            # Calculation question: check for correct answer
            checks["has_604"] = "604" in gen_lower or "$604" in generated
            checks["has_293"] = "293" in gen_lower or "$293" in generated
            checks["has_2_061"] = "2.061" in gen_lower
            checks["has_calculation"] = "×" in generated or "*" in generated or "multipl" in gen_lower
            checks["cannot_calculate"] = any(phrase in gen_lower for phrase in [
                "cannot calculate", "cannot compute", "not possible", "missing",
                "not available", "not provided", "unable to", "do not have enough"
            ])
            checks["is_correct"] = checks["has_604"] and not checks["cannot_calculate"]
        
        return checks
    
    def _answer_completeness(self, generated: str, expected: str) -> float:
        """Estimate how complete the answer is compared to expected."""
        # Simple heuristic: compare lengths and key information density
        gen_words = len(generated.split())
        exp_words = len(expected.split())
        
        if exp_words == 0:
            return 1.0
        
        # Base score on word count ratio (capped at 1.0)
        length_score = min(gen_words / exp_words, 1.0)
        
        # Combine with keyword coverage
        keyword_score = self._keyword_coverage(generated, expected)
        
        return (length_score + keyword_score) / 2
    
    def _source_citation_quality(self, generated: str) -> float:
        """Check if answer includes source citations."""
        # Look for source indicators
        source_indicators = [
            "source:", "table:", "page", "document", "pdf",
            "according to", "based on", "from"
        ]
        
        gen_lower = generated.lower()
        found_indicators = sum(1 for indicator in source_indicators if indicator in gen_lower)
        
        # Score based on number of source indicators found
        return min(found_indicators / 3.0, 1.0)
    
    def evaluate_from_csv(
        self,
        questions_csv: str,
        results_csv: Optional[str] = None,
        skip_contexts: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate questions from CSV file.
        
        If results_csv is provided, uses existing results without re-running agent.
        Otherwise, runs agent on questions and evaluates.
        
        Args:
            questions_csv: Path to questions CSV
            results_csv: Path to existing results CSV (optional)
            skip_contexts: If True, skip context extraction when using existing results (faster)
        """
        questions_df = pd.read_csv(questions_csv)
        
        if results_csv and Path(results_csv).exists():
            # Use existing results
            results_df = pd.read_csv(results_csv)
            # Merge with questions to get expected outputs
            merged = questions_df.merge(
                results_df,
                on="question",
                how="left",
                suffixes=("_q", "_r")
            )
            print(f"Using existing results from: {results_csv}")
            print(f"Evaluating {len(merged)} questions (without re-running agent)...")
        else:
            # Run agent and evaluate
            merged = questions_df
            print(f"Running agent on {len(merged)} questions (this will take longer)...")
        
        evaluation_results = []
        
        for idx, row in merged.iterrows():
            question_id = row.get("id", "")
            question = row.get("question", "")
            expected = row.get("expected_output", "")
            
            if results_csv and Path(results_csv).exists():
                # Use existing answer - NO need to re-run agent!
                generated = row.get("answer", "")
                query_category = row.get("category", "unknown")
                
                # Only get contexts if explicitly needed (for RAGAS or context-based metrics)
                if skip_contexts:
                    contexts = []  # Empty contexts - not needed for most metrics
                else:
                    # Only if contexts are really needed (e.g., for RAGAS)
                    print(f"  [{idx+1}/{len(merged)}] Getting contexts for {question_id}...")
                    resp = self.agent.answer(question)
                    contexts = self._extract_contexts(resp.get("sources", []))
            else:
                # Run agent
                print(f"  [{idx+1}/{len(merged)}] Processing {question_id}...")
                resp = self.agent.answer(question)
                generated = resp.get("answer", "")
                query_category = resp.get("query_category", "unknown")
                contexts = self._extract_contexts(resp.get("sources", []))
            
            # Calculate metrics (most don't need contexts)
            metrics = self._calculate_metrics(
                question=question,
                generated_answer=generated,
                expected_output=expected,
                contexts=contexts,
                question_id=question_id
            )
            
            evaluation_results.append({
                "question_id": question_id,
                "question": question,
                "expected_output": expected,
                "generated_answer": generated,
                "query_category": query_category,
                **metrics
            })
        
        return pd.DataFrame(evaluation_results)
    
    def run_ragas_evaluation(
        self,
        questions_csv: str,
        results_csv: Optional[str] = None,
        use_cached_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run RAGAS evaluation if available.
        
        Requires ragas package to be installed.
        
        Args:
            questions_csv: Path to questions CSV
            results_csv: Path to existing results CSV (optional)
            use_cached_results: If True and results_csv exists, use cached answers but still get contexts
        """
        if not RAGAS_AVAILABLE:
            return {
                "error": "RAGAS not available. Install with: pip install ragas",
                "available": False
            }
        
        questions_df = pd.read_csv(questions_csv)
        
        # Check if we can use cached results
        if use_cached_results and results_csv and Path(results_csv).exists():
            results_df = pd.read_csv(results_csv)
            merged = questions_df.merge(
                results_df,
                on="question",
                how="left",
                suffixes=("_q", "_r")
            )
            print(f"Using cached answers from {results_csv}, but need to get contexts...")
        else:
            merged = questions_df
            print("Running full RAG pipeline for RAGAS evaluation...")
        
        records = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        for idx, row in merged.iterrows():
            question = row["question"]
            expected = row.get("expected_output", "")
            
            # Get answer (from cache or agent)
            if use_cached_results and results_csv and Path(results_csv).exists():
                answer = row.get("answer", "")
                if not answer:
                    # Fallback to agent if answer not in cache
                    print(f"  [{idx+1}/{len(merged)}] Answer not in cache, running agent...")
                    resp = self.agent.answer(question)
                    answer = resp["answer"]
                    contexts = self._extract_contexts(resp.get("sources", []))
                else:
                    # Use cached answer but get contexts
                    print(f"  [{idx+1}/{len(merged)}] Using cached answer, getting contexts...")
                    resp = self.agent.answer(question)
                    contexts = self._extract_contexts(resp.get("sources", []))
            else:
                # Run full agent
                print(f"  [{idx+1}/{len(merged)}] Running agent...")
                resp = self.agent.answer(question)
                answer = resp["answer"]
                contexts = self._extract_contexts(resp.get("sources", []))
            
            records["question"].append(question)
            records["answer"].append(answer)
            records["contexts"].append(contexts)
            records["ground_truth"].append(expected)
        
        from datasets import Dataset
        dataset = Dataset.from_dict(records)
        
        print("Running RAGAS metrics...")
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        
        return {
            "available": True,
            "scores": scores,
            "records": records
        }
    
    def generate_report(
        self,
        evaluation_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAG SYSTEM EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        total_questions = len(evaluation_df)
        exact_matches = evaluation_df["exact_match"].sum()
        avg_keyword_coverage = evaluation_df["keyword_coverage"].mean()
        avg_number_match = evaluation_df["number_match"].mean()
        avg_completeness = evaluation_df["answer_completeness"].mean()
        avg_source_citation = evaluation_df["source_citation"].mean()
        
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Questions: {total_questions}")
        report_lines.append(f"Exact Matches: {exact_matches} ({exact_matches/total_questions*100:.1f}%)")
        report_lines.append(f"Average Keyword Coverage: {avg_keyword_coverage:.1%}")
        report_lines.append(f"Average Number Match: {avg_number_match:.1%}")
        report_lines.append(f"Average Answer Completeness: {avg_completeness:.1%}")
        report_lines.append(f"Average Source Citation Quality: {avg_source_citation:.1%}")
        report_lines.append("")
        
        # Per-question results
        report_lines.append("PER-QUESTION RESULTS")
        report_lines.append("-" * 80)
        
        for _, row in evaluation_df.iterrows():
            qid = row["question_id"]
            report_lines.append(f"\n{qid}:")
            report_lines.append(f"  Question: {row['question'][:80]}...")
            report_lines.append(f"  Exact Match: {row['exact_match']}")
            report_lines.append(f"  Keyword Coverage: {row['keyword_coverage']:.1%}")
            report_lines.append(f"  Number Match: {row['number_match']:.1%}")
            report_lines.append(f"  Completeness: {row['answer_completeness']:.1%}")
            report_lines.append(f"  Source Citation: {row['source_citation']:.1%}")
            
            # Question-specific metrics
            if qid == "EF_1" and "rules_coverage" in row:
                report_lines.append(f"  Rules Coverage: {row['rules_coverage']:.1%} ({row.get('rules_found', 0)}/{row.get('total_rules', 0)})")
            elif qid == "EF_2" and "fact_score" in row:
                report_lines.append(f"  Fact Score: {row['fact_score']:.1%}")
            elif qid == "EF_3" and "is_correct" in row:
                report_lines.append(f"  Calculation Correct: {row['is_correct']}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report_text, encoding='utf-8')
            print(f"Report saved to: {output_path}")
        
        return report_text


def main():
    """Main evaluation entry point."""
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT environment variable.")
    
    # Initialize agent and evaluator
    print("Initializing RAG Agent...")
    agent = RAGAgent(project_id=project_id, dataset_name="rag_dataset")
    evaluator = RAGEvaluator(agent)
    
    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    questions_csv = str(base_dir / "artifacts" / "questions.csv")
    results_csv = str(base_dir / "artifacts" / "questions_results.csv")
    output_csv = str(base_dir / "artifacts" / "evaluation_results.csv")
    report_path = str(base_dir / "artifacts" / "evaluation_report.txt")
    
    print(f"\nEvaluating from: {questions_csv}")
    if Path(results_csv).exists():
        print(f"Using existing results from: {results_csv}")
    
    # Run evaluation (fast mode: uses existing results, skips context extraction)
    print("\nRunning evaluation (fast mode: using existing results)...")
    evaluation_df = evaluator.evaluate_from_csv(
        questions_csv, 
        results_csv, 
        skip_contexts=True  # Skip context extraction for speed
    )
    
    # Save results
    evaluation_df.to_csv(output_csv, index=False)
    print(f"Evaluation results saved to: {output_csv}")
    
    # Generate report
    print("\nGenerating report...")
    report = evaluator.generate_report(evaluation_df, report_path)
    print("\n" + report)
    
    # Optional: Run RAGAS if available (this requires contexts, so it's slower)
    print("\n" + "="*80)
    print("RAGAS Evaluation (optional, requires contexts - will be slower)")
    print("="*80)
    user_input = input("Run RAGAS evaluation? This will re-run agent to get contexts. (y/N): ").strip().lower()
    
    if user_input == 'y' and RAGAS_AVAILABLE:
        print("\nRunning RAGAS evaluation (this will take longer)...")
        ragas_results = evaluator.run_ragas_evaluation(questions_csv, results_csv, use_cached_results=True)
        if ragas_results.get("available"):
            print("\nRAGAS Scores:")
            print(ragas_results["scores"])
        else:
            print("RAGAS not available")
    elif user_input == 'y' and not RAGAS_AVAILABLE:
        print("\nRAGAS not installed. Install with: pip install ragas")
    else:
        print("Skipping RAGAS evaluation.")


if __name__ == "__main__":
    main()
